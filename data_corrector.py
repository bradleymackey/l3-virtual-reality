# data_corrector.py
# Bradley Mackey
# Virtual Reality coursework 18/19

####################
# tested on Python 3.7.x
# ensure that the `IMUData.csv` file is located in the same directory as this script
###################

import csv
import numpy as np
import math

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
        for col in acc_range:
            data[row,col] = 0.0 if acc_len==0.0 else float(data[row,col])/acc_len
        # normalize magnetometer
        mag_range = range(7,10)
        mag_len = np.linalg.norm(data[row,mag_range])
        for col in mag_range:
            data[row,col] = 0.0 if mag_len==0.0 else float(data[row,col])/mag_len
    print("> Successfully normalized data.")
    return data 

def euler_to_qtrn(angles):
    """converts a numpy euler angle array [x y z] to a quaternion array [x y z w]"""
    (yaw, pitch, roll) = (angles[0], angles[1], angles[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.array([qx, qy, qz, qw])

def qtrn_to_euler(qtrn):
    """converts a numpy quaternion array [x y z w] to an euler angle array [x y z]"""
    (x, y, z, w) = (qtrn[0], qtrn[1], qtrn[2], qtrn[3])
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([yaw, pitch, roll])

def qtrn_conj(qtrn):
    """computes the conjugate of a quaternion, passed as a numpy array [x y z w]"""
    (x, y, z, w) = (qtrn[0], qtrn[1], qtrn[2], qtrn[3])
    return np.array([-x,-y,-z,w])

def qtrn_mult(qtrn_1, qtrn_2):
    """computes the product of 2 quaternions, each [x y z w]"""
    (a_x, a_y, a_z, a_w) = (qtrn_1[0], qtrn_1[1], qtrn_1[2], qtrn_1[3])
    (b_x, b_y, b_z, b_w) = (qtrn_2[0], qtrn_2[1], qtrn_2[2], qtrn_2[3])
    w = a_w*b_w - a_x*b_x - a_y*b_y - a_z*b_z
    x = a_w*b_x + a_x*b_w + a_y*b_z - a_z*b_y
    y = a_w*b_y - a_x*b_z + a_y*b_w + a_z*b_x
    z = a_w*b_z + a_x*b_y - a_y*b_x + a_z*b_w
    return np.array([x,y,z,w])

def main():
    get_sanitized_imu_data()
    q = euler_to_qtrn(np.array([0.2,1.12,2.31]))
    

if __name__=="__main__":
    main()