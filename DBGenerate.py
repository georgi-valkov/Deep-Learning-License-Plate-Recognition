from DbConn import DBConn
import csv

def main():
    conn = DBConn()
    with open('plates.txt') as csvfile:
        dr = csv.DictReader(csvfile)
        for row in dr:
            conn.insert(row['tag'],row['make'],row['model'],row['valid'])

    del conn
