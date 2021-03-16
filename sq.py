import sqlite3

conn = sqlite3.connect('data.db')

cursor = conn.cursor()

cursor.execute("""CREATE TABLE VAL (Age TEXT,Gender TEXT)""")

conn.commit()

cursor.close()

conn.close()