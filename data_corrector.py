# data_corrector.py
# Bradley Mackey
# Virtual Reality coursework 18/19

####################
# Python 3.7.x
# ensure that the `IMUData.csv` file is located in the same directory as this script
###################

import csv
import numpy as np

def get_sanitized_imu_data():
    """retrieves the IMU data from the `IMUData.csv` file"""
    data = []
    with open('IMUData.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        for row in reader:
            data.append(row)
    data = np.array(data)
    print(f"> Successfully retrieved {len(data)} items from IMUData.csv.")
    for row in range(len(data)):
        # rotational rate to radians
        for col in range(1,4):
            data[row,col] = float(data[row,col]) * (np.pi/180) 
        # normalize accelerometer
        acc_range = range(4,7)
        acc_len = np.linalg.norm(data[row,acc_range])
        print("before:",data[row,acc_range])
        for col in acc_range:
            data[row,col] = 0.0 if acc_len==0.0 else float(data[row,col])/acc_len
        print("after:",data[row,acc_range])
        # normalize magnetometer
        mag_range = range(7,10)
        mag_len = np.linalg.norm(data[row,mag_range])
        for col in mag_range:
            data[row,col] = 0.0 if mag_len==0.0 else float(data[row,col])/mag_len
    print("> Successfully normalized data.")
    return data 

def main():
    get_sanitized_imu_data()

if __name__=="__main__":
    main()