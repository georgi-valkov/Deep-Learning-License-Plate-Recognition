import sqlite3


class DBConn:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
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
        input = input.replace(' ', '')
        partial_match_indicator = None # 0 - no partial match, 1 - partial match, 2 not in db
        cursor = self.db.cursor()
        cursor.execute('''SELECT tag,make,model,permit,lot,valid FROM cars WHERE tag=?''', (input,))
        data = cursor.fetchone()
        partial_match_indicator = 0
        if data is None:
            s = set()
            for i in range(len(input)):
                cursor.execute("SELECT tag FROM cars WHERE tag LIKE ?", (input[i] + '%',))
                data = cursor.fetchall()
                for element in data:
                    s.add(element[0])
            for tag in s:
                if _non_overlap(input, tag) <= 2:
                    input = tag

                    cursor.execute('''SELECT tag,make,model,permit,lot,valid FROM cars WHERE tag=?''', (input,))
                    data = cursor.fetchone()
                    partial_match_indicator = 1
                    return data, partial_match_indicator
            return None, 2
        else:

            return data, partial_match_indicator

    # Counts non overlapping chars in two strings
def _non_overlap(string1, string2):
    count = 0
    for i in range(min(len(string1), len(string2))):
        if string1[i] != string2[i]:
            count = count + 1
    return count