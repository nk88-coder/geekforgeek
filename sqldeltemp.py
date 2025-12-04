import sqlite3

conn = sqlite3.connect("transactions.db")
cur = conn.cursor()

cur.execute("DELETE FROM transactions;")   # delete all rows

conn.commit()
conn.close()

print("All rows deleted!")
