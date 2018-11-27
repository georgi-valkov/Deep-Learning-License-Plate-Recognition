import sqlite3


class DBConn:
    def __init__(self):
        self.db = sqlite3.connect('platedb')
        cursor = self.db.cursor()
        cursor.execute(
            '''CREATE TABLE cars(id integer primary key,tag text unique ,make text, model text, valid integer)''')
        self.db.commit()

    def __del__(self):
        self.db.close()

    def insert(self,tag,make,model,valid):
        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO cars(tag, make, model, valid)
                            VALUES(?,?,?,?)''',(tag,make,model,valid))
        self.db.commit()

    def select(self,input):
         cursor = self.db.cursor()
         cursor.execute('''SELECT make, model, valid FROM cars WHERE tag=?''',(input,))
         data = cursor.fetchone()
         return data
