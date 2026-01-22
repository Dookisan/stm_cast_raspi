import mysql.connector
import requests
import time
import datetime
import openpyxl
import pandas as pd
from sqlalchemy import create_engine


# -----------------------------
# Einstellungen
# -----------------------------
EXCEL_FILE = "Wetterdaten.xlsx" # Excel-Datei
# -----------------------------

class DATABASE(object):
    
    def __init__(self):
        try: 
            self.db_connection = self.get_connection()
        except Exception as ex:
                print("Connection could not be made due to the following error: \n", ex)
        print(f"Connection created successfully.")


    def get_connection(self, user = "Elias_", password = "STMCAST", host = "192.168.4.1", port = 3306, database = "weather_data"):
        return create_engine(
            url="mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}".format(
                user, password, host, port, database
            )
    )

    def write_to_training_db(self,data):
        # 1. Daten vom STM32 holen

        now = datetime.datetime.now()
        datum = now.strftime("%Y/%m/%d")
        uhrzeit = now.strftime("%H:%M:")
        observation_time_WEATHERSTATION =  now.strftime("%Y/%m/%d %H:%M")
                            
        # 2. In MySQL einfügen
        mydb = mysql.connector.connect(
                    host="192.168.4.1",
                    user="Elias_",
                    password="STMCAST",
                    database="weather_data"
                )
        mycursor = mydb.cursor() # Objekt mit dem man SQL Befehle ausführen kann.

        sql_api = "INSERT INTO api_data (observation_time, temperature, pressure, humidity) VALUES (%s, %s, %s, %s)"
        val_api = (
                    data["observation_time"],
                    data["temperature"],
                    data["pressure"],
                    data["humidity"]
                )
                
        mycursor.execute(sql_api, val_api)

        sql_weatherstation = "INSERT INTO weatherstation_data (observation_time, temperature, pressure, humidity) VALUES (%s, %s, %s, %s)"
        val_weather = (
                        observation_time_WEATHERSTATION,
                        data["weatherstation_temp"],
                        data["weatherstation_press"],
                        data["weatherstation_hum"]
                )
                
        mycursor.execute(sql_weatherstation, val_weather)

        mydb.commit() 
        print(mycursor.rowcount, "Neue Wetterdaten in Datenbank eingefügt.")

        mycursor.close()
        mydb.close()

    def write_to_prediction_temp_db(self,data):
        mydb = mysql.connector.connect(
                    host="192.168.4.1",
                    user="Elias_",
                    password="STMCAST",
                    database="weather_data"
                )
        mycursor = mydb.cursor() # Objekt mit dem man SQL Befehle ausführen kann.

        sql_predictions = ("INSERT INTO stmcast_predictions_temp (HOUR_0, HOUR_1, HOUR_2, HOUR_3, HOUR_4, HOUR_5, HOUR_6, HOUR_7, HOUR_8, HOUR_9, HOUR_10, HOUR_11, HOUR_12, HOUR_13, HOUR_14, HOUR_15, HOUR_16, HOUR_17, HOUR_18, HOUR_19, HOUR_20, HOUR_21, HOUR_22, HOUR_23) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        val_predictions = (
                            data["00:00"],
                            data["01:00"],
                            data["02:00"],
                            data["03:00"],
                            data["04:00"],
                            data["05:00"],
                            data["06:00"],
                            data["07:00"],
                            data["08:00"],
                            data["09:00"],
                            data["10:00"],
                            data["11:00"],
                            data["12:00"],
                            data["13:00"],
                            data["14:00"],
                            data["15:00"],
                            data["16:00"],
                            data["17:00"],
                            data["18:00"],
                            data["19:00"],
                            data["20:00"],
                            data["21:00"],
                            data["22:00"],
                            data["23:00"]
                        )
        mycursor.execute(sql_predictions, val_predictions)#

        mydb.commit() 
        print(mycursor.rowcount, "Neue Wetterdaten in predictions DB eingefügt.")

        mycursor.close()
        mydb.close()

    def write_to_prediction_hum_db(self,data):
        mydb = mysql.connector.connect(
                    host="192.168.4.1",
                    user="Elias_",
                    password="STMCAST",
                    database="weather_data"
                )
        mycursor = mydb.cursor() # Objekt mit dem man SQL Befehle ausführen kann.

        sql_predictions = ("INSERT INTO stmcast_predictions_hum (HOUR_0, HOUR_1, HOUR_2, HOUR_3, HOUR_4, HOUR_5, HOUR_6, HOUR_7, HOUR_8, HOUR_9, HOUR_10, HOUR_11, HOUR_12, HOUR_13, HOUR_14, HOUR_15, HOUR_16, HOUR_17, HOUR_18, HOUR_19, HOUR_20, HOUR_21, HOUR_22, HOUR_23) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        val_predictions = (
                            data["00:00"],
                            data["01:00"],
                            data["02:00"],
                            data["03:00"],
                            data["04:00"],
                            data["05:00"],
                            data["06:00"],
                            data["07:00"],
                            data["08:00"],
                            data["09:00"],
                            data["10:00"],
                            data["11:00"],
                            data["12:00"],
                            data["13:00"],
                            data["14:00"],
                            data["15:00"],
                            data["16:00"],
                            data["17:00"],
                            data["18:00"],
                            data["19:00"],
                            data["20:00"],
                            data["21:00"],
                            data["22:00"],
                            data["23:00"]
                        )
        mycursor.execute(sql_predictions, val_predictions)#

        mydb.commit() 
        print(mycursor.rowcount, "Neue Wetterdaten in predictions DB eingefügt.")

        mycursor.close()
        mydb.close()



