"""
agents/sql_validator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQL validation layer for DefinitionDrift
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Two-stage validation before any SQL hits the database:

  Stage 1 — AST parse (sqlglot)
    Catches: syntax errors, unmatched parens, invalid keywords,
             broken CTEs, malformed UNION statements.
    Cost: zero — no DB or API call.

  Stage 2 — Schema column check
    Verifies every column reference in the AST exists in the
    known Contoso schema. Catches hallucinated column names
    BEFORE they produce a confusing DB error.
    Cost: zero — uses an in-memory schema dict.

Returns a ValidationResult with:
  valid   bool
  errors  list[str]   — human-readable, shown to user
  fixed_sql str|None  — sqlglot-reformatted SQL (prettier)
  warnings list[str]  — non-blocking issues (e.g. SELECT *)

Usage:
  from agents.sql_validator import validate_sql
  result = validate_sql(sql_string)
  if not result.valid:
      return {"error": result.errors}
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

import sqlglot
import sqlglot.expressions as exp

# ── Contoso schema map ────────────────────────────────────────────────────────
# Maps table_name (lowercase) → set of valid column names (lowercase)
CONTOSO_SCHEMA: dict[str, set[str]] = {
    "factsales": {
        "saleskey", "datekey", "storekey", "productkey", "customerkey",
        "channelkey", "unitcost", "unitprice", "salesquantity",
        "returnquantity", "returnamount", "discountamount",
        "totalcost", "salesamount", "margin",
    },
    "factonlinesales": {
        "onlinesaleskey", "datekey", "storekey", "productkey", "customerkey",
        "promotionkey", "unitcost", "unitprice", "salesquantity",
        "returnquantity", "returnamount", "discountamount",
        "totalcost", "salesamount", "margin",
    },
    "dimproduct": {
        "productkey", "productname", "productlabel", "productdescription",
        "productsubcategorykey", "brandname", "unitcost", "unitprice", "status",
    },
    "dimstore": {
        "storekey", "storename", "storetype", "storemanager",
        "storephone", "sellingareasize", "opendate", "status", "geographykey",
    },
    "dimcustomer": {
        "customerkey", "firstname", "lastname", "birthdate",
        "maritalstatus", "gender", "emailaddress", "annualincome",
        "totalchildren", "educationlevel", "occupation",
        "houseownerflag", "customertype", "geographykey",
    },
    "dimdate": {
        "datekey", "fulldatelabel", "calendaryear", "calendarquarter",
        "calendarmonth", "calendarweek", "daynumberofweek",
        "daynameofweek", "isweekend", "fiscalyear", "fiscalquarter", "fiscalmonth",
    },
    "dimproductcategory": {
        "productcategorykey", "productcategoryname", "productcategorylabel", "description",
    },
    "dimproductsubcategory": {
        "productsubcategorykey", "productsubcategoryname", "productcategorykey",
    },
    "dimgeography": {
        "geographykey", "geographytype", "continentname",
        "cityname", "statename", "regioncountryname",
    },
}

# All valid table names
VALID_TABLES: set[str] = set(CONTOSO_SCHEMA.keys())

# SQL functions that look like columns but aren't
SQL_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "round", "coalesce", "nullif",
    "date", "strftime", "julianday", "abs", "upper", "lower", "length",
    "substr", "replace", "trim", "cast", "iif", "ifnull",
}


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fixed_sql: Optional[str] = None
    parse_ok: bool = False
    schema_ok: bool = False

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "fixed_sql": self.fixed_sql,
            "parse_ok": self.parse_ok,
            "schema_ok": self.schema_ok,
        }


# ── Stage 1: AST parse ────────────────────────────────────────────────────────
def _parse(sql: str) -> tuple[bool, list[str], Optional[sqlglot.Expression]]:
    """Returns (success, errors, ast)."""
    if not sql or not sql.strip():
        return False, ["SQL is empty"], None
    try:
        # sqlglot transpile to SQLite dialect validates syntax
        parsed = sqlglot.parse_one(sql, dialect="sqlite", error_level=sqlglot.ErrorLevel.RAISE)
        return True, [], parsed
    except sqlglot.errors.ParseError as e:
        errors = [f"Syntax error: {str(e)}"]
        return False, errors, None
    except Exception as e:
        return False, [f"Parse failed: {str(e)}"], None


# ── Stage 2: Schema column check ─────────────────────────────────────────────
def _collect_table_aliases(ast: sqlglot.Expression) -> dict[str, str]:
    """
    Builds a map of alias → real_table_name (lowercase).
    Also collects CTE names and SELECT-level column aliases so they
    are not flagged as unknown tables or columns.
    """
    aliases: dict[str, str] = {}
    select_aliases: set[str] = set()

    # Collect CTE names
    for node in ast.walk():
        if isinstance(node, exp.CTE):
            cte_name = node.alias.lower() if node.alias else ""
            if cte_name:
                aliases[cte_name] = "__cte__"

    # Collect SELECT column aliases (e.g. SUM(...) AS net_rev)
    for node in ast.walk():
        if isinstance(node, exp.Alias):
            alias_name = node.alias.lower() if node.alias else ""
            if alias_name:
                select_aliases.add(alias_name)

    # Collect table → alias
    for node in ast.walk():
        if isinstance(node, exp.Table):
            tname = node.name.lower() if node.name else ""
            alias = node.alias.lower() if node.alias else tname
            if tname and tname not in aliases:
                aliases[alias] = tname
                aliases[tname] = tname

    aliases["__select_aliases__"] = "__meta__"  # sentinel
    # Store select_aliases in a way _check_schema can access
    aliases.update({f"__alias__{a}": "__select_alias__" for a in select_aliases})
    return aliases


def _check_schema(ast: sqlglot.Expression) -> tuple[bool, list[str], list[str]]:
    """
    Returns (ok, errors, warnings).
    Checks:
      - All table names exist in CONTOSO_SCHEMA
      - All column references resolve to a known table column
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Build alias map
    aliases = _collect_table_aliases(ast)

    # Check table names (skip CTEs — they are virtual)
    for node in ast.walk():
        if isinstance(node, exp.Table):
            tname = node.name.lower() if node.name else ""
            if not tname:
                continue
            # Skip CTE references
            if aliases.get(tname) == "__cte__":
                continue
            if tname not in VALID_TABLES:
                errors.append(
                    f"Unknown table '{node.name}'. "
                    f"Valid tables: {', '.join(sorted(t.title() for t in VALID_TABLES))}"
                )

    # Check column references
    for node in ast.walk():
        if isinstance(node, exp.Column):
            col_name = node.name.lower() if node.name else ""
            table_ref = node.table.lower() if node.table else None

            # Skip SQL functions that appear as column names
            if col_name in SQL_FUNCTIONS:
                continue

            # Skip star (also handled via exp.Star below)
            if col_name in ("*", ""):
                continue

            # Skip aliases (single-char or short) that we can't resolve
            if not col_name:
                continue

            if table_ref:
                # Resolve alias to real table
                real_table = aliases.get(table_ref)
                # Skip columns from CTEs — they're virtual tables
                if real_table == "__cte__":
                    continue
                if real_table and real_table in CONTOSO_SCHEMA:
                    if col_name not in CONTOSO_SCHEMA[real_table]:
                        errors.append(
                            f"Column '{node.name}' does not exist in "
                            f"'{real_table.title()}'. "
                            f"Available: {', '.join(sorted(CONTOSO_SCHEMA[real_table]))}"
                        )
                elif real_table and real_table not in CONTOSO_SCHEMA:
                    pass  # table already flagged above
            else:
                # Skip if this is a computed alias from a SELECT clause
                if aliases.get(f"__alias__{col_name}") == "__select_alias__":
                    continue
                # No table qualifier — check if col exists in ANY table
                found_in = [
                    t for t, cols in CONTOSO_SCHEMA.items() if col_name in cols
                ]
                if not found_in:
                    errors.append(
                        f"Column '{node.name}' not found in any Contoso table. "
                        "Check for typos or hallucinated column names."
                    )
                elif len(found_in) > 1:
                    warnings.append(
                        f"Column '{node.name}' exists in multiple tables "
                        f"({', '.join(t.title() for t in found_in)}) — "
                        "qualify with table name to avoid ambiguity."
                    )

    # Check for SELECT * (Star expression, separate from Column)
    for node in ast.walk():
        if isinstance(node, exp.Star):
            if "SELECT * warning" not in " ".join(warnings):
                warnings.append("SELECT * detected — prefer explicit column names for clarity")

    ok = len(errors) == 0
    return ok, errors, warnings


# ── Public API ────────────────────────────────────────────────────────────────
def validate_sql(sql: str) -> ValidationResult:
    """
    Full two-stage validation.
    Returns a ValidationResult — check .valid before executing.
    """
    result = ValidationResult(valid=False)

    # Stage 1 — parse
    parse_ok, parse_errors, ast = _parse(sql)
    result.parse_ok = parse_ok
    result.errors.extend(parse_errors)

    if not parse_ok:
        return result  # no point running schema check on broken SQL

    # Reformat SQL (sqlglot makes it prettier and canonical)
    try:
        result.fixed_sql = sqlglot.transpile(sql, read="sqlite", write="sqlite", pretty=True)[0]
    except Exception:
        result.fixed_sql = sql

    # Stage 2 — schema check
    schema_ok, schema_errors, schema_warnings = _check_schema(ast)
    result.schema_ok = schema_ok
    result.errors.extend(schema_errors)
    result.warnings.extend(schema_warnings)

    result.valid = parse_ok and schema_ok
    return result


def validate_and_fix(sql: str) -> tuple[bool, str, list[str]]:
    """
    Convenience wrapper.
    Returns (is_valid, sql_to_use, error_messages).
    sql_to_use is the reformatted SQL if valid, original if not.
    """
    v = validate_sql(sql)
    return v.valid, (v.fixed_sql or sql), v.errors


if __name__ == "__main__":
    print("=== SQL Validator Tests ===\n")

    tests = [
        # (description, sql, expect_valid)
        (
            "Valid net revenue query",
            "SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales",
            True,
        ),
        (
            "Valid join with DimDate",
            """SELECT CalendarYear, SUM(SalesAmount)
               FROM FactSales fs JOIN DimDate dd ON fs.DateKey = dd.DateKey
               WHERE dd.CalendarYear = 2008
               GROUP BY CalendarYear""",
            True,
        ),
        (
            "Hallucinated column",
            "SELECT SUM(Revenue) FROM FactSales",
            False,
        ),
        (
            "Unknown table",
            "SELECT * FROM SalesFact",
            False,
        ),
        (
            "Syntax error",
            "SELECT SUM(SalesAmount FROM FactSales",
            False,
        ),
        (
            "Ambiguous column (no table qualifier)",
            "SELECT SUM(UnitCost) FROM FactSales",
            True,  # valid — but will warn about ambiguity
        ),
        (
            "CTE query",
            """WITH rev AS (
                SELECT SUM(SalesAmount - ReturnAmount) as net_rev FROM FactSales
               )
               SELECT net_rev FROM rev""",
            True,
        ),
    ]

    passed = 0
    for desc, sql, expect in tests:
        result = validate_sql(sql)
        ok = result.valid == expect
        icon = "✅" if ok else "❌"
        print(f"{icon} {desc}")
        print(f"   valid={result.valid}, parse_ok={result.parse_ok}, schema_ok={result.schema_ok}")
        if result.errors:
            for e in result.errors:
                print(f"   ERROR: {e}")
        if result.warnings:
            for w in result.warnings:
                print(f"   WARN:  {w}")
        print()
        if ok:
            passed += 1

    print(f"Results: {passed}/{len(tests)} passed")