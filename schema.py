import sqlite3
c = sqlite3.connect('data/contoso.db')
for row in c.execute("SELECT sql FROM sqlite_master WHERE type='table'"):
    print(row[0])
