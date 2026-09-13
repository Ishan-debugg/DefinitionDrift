import sys; sys.path.insert(0, ".")
from store.db import init_db, get_all_definitions
init_db()
defs = get_all_definitions()
print(f"Total defs: {len(defs)}")
for d in defs:
    print(f"  name={d['name']}, approved={d['approved']}, sql_expr={str(d.get('sql_expr','NONE'))[:80]}")
