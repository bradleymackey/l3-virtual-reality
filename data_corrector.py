# data_corrector.py
# Bradley Mackey
# Virtual Reality coursework 18/19

####################
# Python 3.7.x
# ensure that the `IMUData.csv` file is located in the same directory as this script
###################

import os.path
import numpy as np
import csv


def get_imu_data():
    """retrieves the IMU data from the `IMUData.csv` file"""
    data = []
    with open('IMUData.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        for row in reader:
            data.append(row)
    print(f"Successfully retrieved {len(data)} items from IMUData.csv.")
    return data



def main():
    get_imu_data()


if __name__=="__main__":
    main()