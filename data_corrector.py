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

# the sample rate, in Hz
IMU_SAMPLE_RATE = 256

# PROBLEM 1:

def get_sanitized_imu_data():
    """retrieves the IMU data from the `IMUData.csv` file"""
    print(">>> Data Extraction <<<")
    data = []
    with open('IMUData.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        for row in reader:
            data.append(row)
    data = np.array(data, dtype=np.float32)
    print(f"> Successfully retrieved {len(data)} items from IMUData.csv.")
    for row in range(len(data)):
        # rotational rate to radians
        for col in range(1,4):
            data[row,col] = data[row,col] * (np.pi/180) 
        # normalize accelerometer
        acc_range = range(4,7)
        acc_len = np.linalg.norm(data[row,acc_range])
        for col in acc_range:
            data[row,col] = 0.0 if acc_len==0.0 else data[row,col]/acc_len
        # normalize magnetometer
        mag_range = range(7,10)
        mag_len = np.linalg.norm(data[row,mag_range])
        for col in mag_range:
            data[row,col] = 0.0 if mag_len==0.0 else data[row,col]/mag_len
    print("> Successfully normalized data.")
    return data 

def reading_to_qtrn(angles):
    """converts a numpy euler rotation at {IMU_SAMPLE_RATE}Hz [x y z] to a quaternion array [w i j k]"""
    sample_time = float(1/IMU_SAMPLE_RATE)
    rot_angle = np.linalg.norm(angles) * sample_time
    rot_axis = np.repeat(1/np.linalg.norm(angles),3) * angles
    (x, y, z) = (rot_axis[0], rot_axis[1], rot_axis[2])
    w = np.cos(rot_angle/2)
    i = np.sin(rot_angle/2) * x
    j = np.sin(rot_angle/2) * y
    k = np.sin(rot_angle/2) * z
    return np.array([w, i, j, k])

def qtrn_to_euler(qtrn):
    """converts a numpy quaternion array [w i j k] to an euler angle array (axis of rotation followed by the angle rotated by) [x y z theta]"""
    (w, i, j, k) = (qtrn[0], qtrn[1], qtrn[2], qtrn[3])
    angle = 2 * np.arctan2(np.linalg.norm(qtrn[1:]), w)
    if angle==0:
        # angle is 0? all values are 0
        return np.repeat(0.0,4)
    axis_x = i/np.sin(angle/2)
    axis_y = j/np.sin(angle/2)
    axis_z = k/np.sin(angle/2)
    return np.array([axis_x, axis_y, axis_z, angle])

def qtrn_conj(qtrn):
    """computes the conjugate of a quaternion, passed as a numpy array [w i j k]"""
    (w, i, j, k) = (qtrn[0], qtrn[1], qtrn[2], qtrn[3])
    return np.array([w,-i,-j,-k])

def qtrn_mult(qtrn_1, qtrn_2):
    """computes the product of 2 quaternions, each [w i j k]"""
    (a_w, a_x, a_y, a_z) = (qtrn_1[0], qtrn_1[1], qtrn_1[2], qtrn_1[3])
    (b_w, b_x, b_y, b_z) = (qtrn_2[0], qtrn_2[1], qtrn_2[2], qtrn_2[3])
    w = a_w*b_w - a_x*b_x - a_y*b_y - a_z*b_z
    x = a_w*b_x + a_x*b_w + a_y*b_z - a_z*b_y
    y = a_w*b_y - a_x*b_z + a_y*b_w + a_z*b_x
    z = a_w*b_z + a_x*b_y - a_y*b_x + a_z*b_w
    return np.array([w,x,y,z])

# PROBLEM 2:

def dead_reckoning(imu_data):
    """implementation of dead-reckoning filter - estimating position only using the gyro
    sanitized IMU data should be input, progress will be reported"""
    # we start at the identity quaternion
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    print(">>> Dead Reckoning <<<")
    print("> Start position:",curr_pos)
    gyro_range = range(1,4)
    for point in imu_data:
        point_qtrn = reading_to_qtrn(point[gyro_range])
        curr_pos = qtrn_mult(curr_pos, point_qtrn)
    print("> End position:",curr_pos)
    return curr_pos

# EXECUTION:

def main():
    data = get_sanitized_imu_data()
    dead_reckoning(data)
    # print(data.dtype)
    # print("input:",data[0,1:4])
    # q = reading_to_qtrn(data[0,1:4])
    # print("quat:",q)
    # r = qtrn_to_euler(q)
    # print("euler:",r)


if __name__=="__main__":
    main()