from DbConn import DBConn
import csv
conn = DBConn()
#tag,make,model,permit,lot,valid
with open('plates.csv') as csvfile:
    dr = csv.DictReader(csvfile)
    for row in dr:
        #print(row['tag'],row['make'],row['model'],row['permit'],row['lot'],row['valid'])
        conn.insert(row['tag'],row['make'],row['model'],row['permit'],row['lot'],row['valid'])
    del conn
