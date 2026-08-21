import sqlite3

conn = sqlite3.connect('agent_memory.db')

# CREATE A memory TABLE

#cursor=conn.cursor()

#cursor.execute("""
# CREATE TABLE IF NOT EXISTS memory(
#  id INTEGER PRIMARY KEY AUTOINCREMENT,
#  role TEXT NOT NULL,
#  content TEXT NOT NULL,
#  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
# )
#""")
def save_excute(role,content):
 conn = sqlite3.connect('agent_memory.db')
 conn.execute("INSERT INTO memory(role,content) VALUES (?,?)",(role,content))
 conn.commit()
 conn.close()
 print("it has been saved in the data basem successfuly")

def load_memory():
 conn = sqlite3.connect('agent_memory.db')
 rows =conn.execute("SELECT role , content FROM memory ORDER BY id").fetchall()
 conn.close()
 return rows

def reset_memory():
  conn = sqlite3.connect('agent_memory.db')
  conn.execute("DELETE FROM memory")
  conn.commit()
  conn.close()
