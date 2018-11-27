import sqlite3


class DBConn:
    def __init__(self):
        self.db = sqlite3.connect('platedb')
        cursor = self.db.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS cars(id integer primary key autoincrement,tag text unique, make text, model text,permit text, lot text,valid integer)''')
        self.db.commit()

    def __del__(self):
        self.db.close()

    def insert(self,tag,make,model,permit,lot,valid):
        cursor = self.db.cursor()
        cursor.execute('''INSERT OR REPLACE INTO cars(tag,make,model,permit,lot,valid)
                            VALUES(?,?,?,?,?,?)''',(tag, make, model, permit, lot, valid))
        self.db.commit()

    #returns 5 element tuple
    def select(self,input):
         cursor = self.db.cursor()
         cursor.execute('''SELECT make,model,permit,lot,valid FROM cars WHERE tag=?''',(input,))
         data = cursor.fetchone()
         return data
