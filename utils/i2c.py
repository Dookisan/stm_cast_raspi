import pandas as pd
from src import controller
import smbus2
import time
import numpy as np
import struct

"""
I2C Communication module for STM32 telemetry data exchange.
created_by: Elias Schebath
"""
class i2c_com(object):
    def __init__(self):
        self.i2c_addr = 0x21
        self.i2c_bus = smbus2.SMBus(1)

    def write_byte(self, data):
        """Schreibt 1 Byte an Register 0x0 (konvertiert NumPy-Typen automatisch)"""
        # Konvertiere zu normalem Python int (falls NumPy-Typ)

        for byte in data:
            byte_value = int(byte) & 0xFF  # Begrenze auf 0-255
            self.i2c_bus.write_byte_data(self.i2c_addr, 0x0, byte_value)

    def read_byte(self, num_bytes=24):
        while(1):
            try:
                incoming_data = self.i2c_bus.read_byte(self.i2c_addr)
                print(f"Received data: {incoming_data} ")
                break
            except Exception as e:
                time.sleep(0.01)
                continue
            finally:
                return incoming_data

    def writeArray(self, data_array):
        """Schreibt ein Array zum I2C-Gerät"""
    
        # Array direkt senden
        byte_array =  struct.pack('<3f', *data_array)

        write = self.i2c_bus.write_i2c_block_data(self.i2c_addr,0,byte_array)
        
    
    def readArray(self, num_bytes=8):
        """Liest ein Array mit mehreren Bytes vom I2C-Gerät"""
        from smbus2 import i2c_msg
        
        # Ganzes Array direkt lesen
        read = i2c_msg.read(self.i2c_addr, num_bytes)
        self.i2c_bus.i2c_rdwr(read)
        data = np.array(list(read))
        print(f"Received array data: {data} ")
        return data