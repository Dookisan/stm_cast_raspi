import uvicorn
import requests
import pandas as pd
from src import controller
import smbus2
import time

class i2c_com(object):
    def __init__(self):
        self.i2c_addr = 0x21
        self.i2c_bus = smbus2.SMBus(1)

    def generateTelematry(self,data):
        self.i2c_bus.write_byte_data(self.i2c_addr,0x00,data)
        incoming_data = self.i2c_bus.read_byte(self.i2c_addr)
        return incoming_data
    
    def readArray(self, num_bytes=24):
        """Liest ein Array mit mehreren Bytes vom I2C-Gerät"""
        from smbus2 import i2c_msg
        
        # Kommando 0x02 senden - STM32 empfängt und bereitet Daten vor
        write = i2c_msg.write(self.i2c_addr, [0x02])
        self.i2c_bus.i2c_rdwr(write)
        
        time.sleep(0.1)  # Längere Pause für STM32 Verarbeitung
        
        # Ganzes Array direkt lesen (ohne Register-Adresse)
        read = i2c_msg.read(self.i2c_addr, num_bytes)
        self.i2c_bus.i2c_rdwr(read)
        
        return list(read)
    
    def writeArray(self, data_array):
        """Schreibt ein Array zum I2C-Gerät"""
        from smbus2 import i2c_msg
        
        # Array direkt senden
        write = i2c_msg.write(self.i2c_addr, data_array)
        self.i2c_bus.i2c_rdwr(write)

if __name__ == "__main__":
    i2c = i2c_com()
    """
    url = "http://10.0.0.42/json_data"
    response = requests.get(url)
    print("Status Code:", response.status_code)
    print("Antwort:", response.text)
    df = pd.read_json(url,lines = True)
    print(f"dataframe: {df.shape}")
    temp = df.loc[:,"weatherstation_temp"].copy()
    value = temp[0].astype(int)
    print(f"Temperature: {temp}")
    """
    
    while(True):
        try:
            value = 23
            # Nur Array lesen - enthält alle Daten
            #print(f"data i2c array: {i2c.readArray()}")
            incoming = i2c.generateTelematry(value)
            print(f"data i2c byte: {incoming}")
            
        except Exception as e:
            print(f"Fehler beim Lesen des I2C-Arrays: {e}")
        finally:
            time.sleep(0.9)  # Kurze Pause - STM32 wartet eh im Loop

    
    

    #uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)

