"""
scripts/show_hitl_queue.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI tool to view and resolve HITL conflicts without the React UI.
Zero dependency on frontend — works anywhere you can run Python.

Usage:
    python scripts/show_hitl_queue.py            # view all pending
    python scripts/show_hitl_queue.py --resolve  # interactive resolve mode
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from store.db import (
    get_pending_conflicts, resolve_conflict,
    get_all_definitions, upsert_definition
)

COLORS = {
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(text, color): return f"{COLORS[color]}{text}{COLORS['reset']}"

def show_queue():
    conflicts = get_pending_conflicts()
    if not conflicts:
        print(c("\n  ✅ No pending conflicts in the HITL queue.\n", "green"))
        return []

    print(c(f"\n  ⚠️  {len(conflicts)} pending conflict(s) in the HITL queue\n", "yellow"))
    print(c("  " + "─" * 64, "blue"))

    for i, conf in enumerate(conflicts, 1):
        sim_pct = round(conf["similarity"] * 100, 1)
        sim_color = "red" if sim_pct >= 90 else "yellow" if sim_pct >= 80 else "cyan"
        print(f"\n  {c(f'[{i}]', 'bold')} ID: {c(conf['id'], 'cyan')}")
        print(f"      Created:    {conf['created_at']}")
        print(f"      Similarity: {c(f'{sim_pct}%', sim_color)}")
        print(f"\n      {c('Incoming question:', 'bold')}")
        print(f"        {c(conf['question_a'], 'yellow')}")
        print(f"\n      {c('Existing definition:', 'bold')}")
        print(f"        Name: {c(conf['def_b'] or 'N/A', 'green')}")
        print(f"        Desc: {conf['question_b']}")
        print(c("  " + "─" * 64, "blue"))

    return conflicts


def interactive_resolve(conflicts):
    if not conflicts:
        return

    print(c("\n  RESOLVE MODE — Keyboard shortcuts:", "bold"))
    print("    a = approve existing definition (approve_b)")
    print("    n = create new definition from incoming question (approve_a)")
    print("    m = merge both into a new canonical definition")
    print("    r = reject (keep both, no canonical chosen)")
    print("    s = skip this conflict\n")

    for conf in conflicts:
        sim_pct = round(conf["similarity"] * 100, 1)
        print(c(f"\n  Conflict {conf['id'][:8]}  ({sim_pct}% match)", "cyan"))
        print(f"  Incoming: {c(conf['question_a'], 'yellow')}")
        print(f"  Existing: {c(conf['def_b'], 'green')} — {conf['question_b']}")

        while True:
            choice = input(c("  Action [a/n/m/r/s]: ", "bold")).strip().lower()

            if choice == "s":
                print(c("  Skipped.", "blue"))
                break

            elif choice == "a":
                resolved = resolve_conflict(conf["id"], "approve_b", conf["def_b"])
                print(c(f"  ✅ Resolved → kept '{conf['def_b']}' as canonical.", "green"))
                break

            elif choice == "n":
                name_key = conf["question_a"][:40].lower().replace(" ", "_")
                name_key = "".join(c_char for c_char in name_key if c_char.isalnum() or c_char == "_")
                confirm = input(f"  Create new definition '{name_key}'? [y/N]: ").strip().lower()
                if confirm == "y":
                    upsert_definition(
                        name=name_key,
                        description=conf["question_a"],
                        approved=True,
                        reason=f"approved via CLI from conflict {conf['id']}"
                    )
                    resolve_conflict(conf["id"], "approve_a", conf["question_a"])
                    print(c(f"  ✅ Created and approved definition '{name_key}'.", "green"))
                    break

            elif choice == "m":
                print(f"  Incoming: {conf['question_a']}")
                print(f"  Existing: {conf['question_b']}")
                merged = input("  Enter merged definition text: ").strip()
                if merged:
                    upsert_definition(
                        name=conf["def_b"] or "merged",
                        description=merged,
                        approved=True,
                        reason=f"merged via CLI conflict {conf['id']}"
                    )
                    resolve_conflict(conf["id"], "merge", merged)
                    print(c(f"  ✅ Merged and saved as '{conf['def_b'] or 'merged'}'.", "green"))
                    break

            elif choice == "r":
                resolve_conflict(conf["id"], "rejected", "discarded")
                print(c("  ✅ Rejected — both definitions kept independently.", "blue"))
                break

            else:
                print(c("  Invalid choice. Use a/n/m/r/s.", "red"))

    print(c("\n  Queue processing complete.\n", "green"))


def main():
    parser = argparse.ArgumentParser(description="DefinitionDrift HITL Queue CLI")
    parser.add_argument("--resolve", action="store_true", help="Enter interactive resolve mode")
    args = parser.parse_args()

    print(c("\n  DefinitionDrift — HITL Queue", "bold"))
    conflicts = show_queue()

    if args.resolve and conflicts:
        interactive_resolve(conflicts)
    elif not args.resolve and conflicts:
        print(c("  Tip: run with --resolve to interactively resolve conflicts.\n", "blue"))

    # also show current approved definitions
    defs = get_all_definitions(approved_only=True)
    print(c(f"  Approved definitions: {len(defs)}", "green"))
    for d in defs:
        print(f"    • {c(d['name'], 'cyan')} (v{d['version']}) — {d['description'][:60]}...")
    print()


if __name__ == "__main__":
    main()