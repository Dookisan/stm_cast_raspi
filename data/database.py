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
    
    def __init__(self,data):
        self.data = data
        self.write_to_db()
        try: 
            self.db_connection = self.get_connection()
        except Exception as ex:
                print("Connection could not be made due to the following error: \n", ex)
        print(f"Connection created successfully.")


    def get_connection(self, user = "Elias_", password = "STMCAST", host = "10.0.0.30", port = 3306, database = "weather_data"):
        return create_engine(
            url="mysql+mysqlconnector://{0}:{1}@{2}:{3}/{4}".format(
                user, password, host, port, database
            )
    )

    def write_to_db(self):
        # 1. Daten vom STM32 holen

        data = self.data

        now = datetime.datetime.now()
        datum = now.strftime("%Y-%m-%d")
        uhrzeit = now.strftime("%H:%M:%S")
        observation_time_WEATHERSTATION =  now.strftime("%Y-%m-%d %H:%M:%S")
                            
        # 2. In MySQL einfügen
        mydb = mysql.connector.connect(
                    host="10.0.0.30",
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
        mydb.commit() # commit -> Methode um Änderungen durchzuführen
        print(mycursor.rowcount, "Neue Wetterdaten in Datenbank eingefügt.")

        mycursor.close()
        mydb.close()

    
