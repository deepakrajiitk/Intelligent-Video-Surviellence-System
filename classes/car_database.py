import mysql.connector
from classes.mysql_convertor import NumpyMySQLConverter

class Car_Database:
    def __init__(self):
        # connecting to database
        self.mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="tracker_car_db"
        )
        self.mydb.set_converter_class(NumpyMySQLConverter)
        self.mycursor = self.mydb.cursor()
        self.attributes = ["red", "yellow", "green", "teal", "blue", "pink", "white", "black", "gray"]
        self.table_name = "car_table"
        self.threshold = 0.5
        # table is created one time only
        # self.create_main_table()

    def create_main_table(self):
        # query format
        table_creation_query = "create table if not exists "+ self.table_name+" (attributes text NOT NULL)";
        self.mycursor.execute(table_creation_query);
        query = "insert into " +self.table_name+ " (attributes) values (%s)";
        for attribute in self.attributes:
            self.add_to_main_table_column("attributes", attribute)
        
    def add_to_main_table_column(self, columnName, value):
        query = "insert into " + self.table_name + " (" + columnName + ") " + " values (%s)"
        self.mycursor.execute(query, [value])
        self.mydb.commit();
    
    def main_table_insert(self, video_id):
        column_name = video_id
        check_query = "SELECT * FROM information_schema.COLUMNS WHERE TABLE_NAME = '{}' AND COLUMN_NAME = '{}'".format(self.table_name, column_name)
        self.mycursor.execute(check_query)
        result = self.mycursor.fetchone()
        if result is None:
            query1 = "ALTER TABLE {} ADD COLUMN {} INT".format(self.table_name, column_name)
            self.mycursor.execute(query1)

        for i in range(len(self.attributes)):
            query2 = "UPDATE {} SET {} = (SELECT COUNT(*) FROM {} WHERE {} > {}) WHERE attributes = '{}'".format(self.table_name, column_name, column_name, self.attributes[i], self.threshold, self.attributes[i])
            self.mycursor.execute(query2)

        self.mydb.commit()
    
    def create_video_table(self, table_name):
        table_creation_query = "CREATE TABLE if not exists "+table_name+" \
        (date VARCHAR(255) NOT NULL, \
        video_id VARCHAR(255) NOT NULL, \
        car_id VARCHAR(255) NOT NULL, \
        timeframe VARCHAR(255) NOT NULL, \
        red float NOT NULL, \
        yellow float NOT NULL, \
        green float NOT NULL, \
        teal float NOT NULL, \
        blue float NOT NULL, \
        pink float NOT NULL, \
        white float NOT NULL, \
        black float NOT NULL, \
        gray float NOT NULL)"
        self.mycursor.execute(table_creation_query);
        self.mydb.commit()
    
    def video_table_insert(self, table_name, values):
        table_insertion_query = "insert into " +table_name+" (date, video_id, car_id, timeframe, red, yellow, green, teal, \
            blue, pink, white, black, gray) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.mycursor.execute(table_insertion_query, values)
        self.mydb.commit()