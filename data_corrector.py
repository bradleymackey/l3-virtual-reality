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

def euler_to_qtrn(euler):
    """converts euler angles (axis + rotation) [x y z theta] to a quaternion [w i j k]"""
    (x, y, z) = euler[:3]
    angle = euler[3]
    w = np.cos(angle/2)
    i = np.sin(angle/2) * x
    j = np.sin(angle/2) * y
    k = np.sin(angle/2) * z
    return np.array([w, i, j, k])

def reading_to_qtrn(angles):
    """converts a numpy euler rotation at {IMU_SAMPLE_RATE}Hz [x y z] to a quaternion array [w i j k]"""
    sample_time = float(1/IMU_SAMPLE_RATE)
    rot_angle = np.linalg.norm(angles) * sample_time
    rot_axis = np.repeat(1/np.linalg.norm(angles),3) * angles
    return euler_to_qtrn(np.append(rot_axis,rot_angle))

def qtrn_to_euler(qtrn):
    """converts a numpy quaternion array [w i j k] to an euler angle array (axis of rotation followed by the angle rotated by) [x y z theta]"""
    (w, i, j, k) = qtrn
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
    (w, i, j, k) = qtrn
    return np.array([w, -i, -j, -k])

def qtrn_mult(qtrn_a, qtrn_b):
    """computes the product of 2 quaternions, each [w i j k]"""
    (a_w, a_x, a_y, a_z) = qtrn_a
    (b_w, b_x, b_y, b_z) = qtrn_b
    w = a_w*b_w - a_x*b_x - a_y*b_y - a_z*b_z
    i = a_w*b_x + a_x*b_w + a_y*b_z - a_z*b_y
    j = a_w*b_y - a_x*b_z + a_y*b_w + a_z*b_x
    k = a_w*b_z + a_x*b_y - a_y*b_x + a_z*b_w
    return np.array([w, i, j, k])

# PROBLEM 2:

def gyro_dead_reckoning(imu_data):
    """implementation of dead-reckoning filter - estimating position only using the gyro
    sanitized IMU data should be input, progress will be reported"""
    # we start at the identity quaternion
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    print(">>> Dead Reckoning (Gyro) <<<")
    print("> Start orientation:",curr_pos)
    gyro_range = range(1,4)
    for point in imu_data:
        point_qtrn = reading_to_qtrn(point[gyro_range])
        curr_pos = qtrn_mult(curr_pos, point_qtrn)
    print("> End orientation:",curr_pos)
    return curr_pos

# PROBLEM 3:

def gyro_acc_positioning(imu_data):
    """computes current position using data both from the gyroscope and accelerometer"""
    ALPHA = 0.01
    print(">>> Tilt Correction <<<")
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    print("> Start orientation:",curr_pos)
    gyro_range = range(1,4)
    acc_range = range(4,7)
    ref_vector = np.array([0.,1.,0.], dtype=np.float32)
    for point in imu_data:
        ### calculate initial position only using the gyro
        gyro_qtrn = reading_to_qtrn(point[gyro_range])
        curr_pos = qtrn_mult(curr_pos, gyro_qtrn)
        #print("gyro pos:",curr_pos)
        ### convert acc data to the global frame
        acc_qtrn = reading_to_qtrn(point[acc_range])
        glob_acc_qtrn = qtrn_mult(qtrn_mult(qtrn_conj(gyro_qtrn), acc_qtrn), gyro_qtrn)
        ### calculate the tilt error
        # x = index 1, z = index 3 (index 0 is w, which relates to angle, and we don't care about this at the moment)
        tilt_error_axis = np.array([glob_acc_qtrn[3], 0.0, glob_acc_qtrn[1]])
        acc_vector = qtrn_to_euler(glob_acc_qtrn)
        #print("acc_vector:",acc_vector)
        cos_ang = np.dot(ref_vector, acc_vector[:3])
        tilt_error_angle = np.arccos(cos_ang)
        #print("error axis:",tilt_error_axis)
        #print("error angle:",tilt_error_angle)
        ### Repair tilt using the comp filter
        comp_filter = euler_to_qtrn(np.append(tilt_error_axis,-ALPHA*tilt_error_angle))
        #print("comp filter:",comp_filter)
        curr_pos = qtrn_mult(comp_filter, curr_pos)
        #print("new position",curr_pos)
        #print()
    print("> End orientation:",curr_pos)
    return curr_pos

# PROBLEM 4:

def gyro_acc_mag_positioning(imu_data):
    """corrects for tilt and yaw using the accelerometer and magnetometer"""
    ALPHA = 0.001
    print(">>> Tilt and Yaw Correction <<<")
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    print("> Start orientation:",curr_pos)
    gyro_range = range(1,4)
    acc_range = range(4,7)
    mag_range = range(7,10)
    ref_vector = np.array([0.,1.,0.], dtype=np.float32)

    # take reference measurements for yaw correction
    m_ref = reading_to_qtrn(imu_data[0,mag_range])
    # transform m_ref to the global frame
    m_ref = qtrn_mult(qtrn_mult(qtrn_conj(curr_pos), m_ref), curr_pos)

    for point in imu_data:
        ### calculate initial position only using the gyro
        gyro_qtrn = reading_to_qtrn(point[gyro_range])
        curr_pos = qtrn_mult(curr_pos, gyro_qtrn)
        ### convert acc data to the global frame
        acc_qtrn = reading_to_qtrn(point[acc_range])
        acc_qtrn = qtrn_mult(qtrn_mult(qtrn_conj(gyro_qtrn), acc_qtrn), gyro_qtrn)
        ### calculate the tilt error
        # x = index 1, z = index 3 (index 0 is w, which relates to angle, and we don't care about this at the moment)
        tilt_error_axis = np.array([acc_qtrn[3], 0.0, acc_qtrn[1]])
        acc_vector = qtrn_to_euler(acc_qtrn)
        cos_ang = np.dot(ref_vector, acc_vector[:3])
        tilt_error_angle = np.arccos(cos_ang)
        ### Repair tilt using the comp filter
        comp_filter = euler_to_qtrn(np.append(tilt_error_axis,-ALPHA*tilt_error_angle))
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        ### convert mag data to the global frame
        mag_qtrn = reading_to_qtrn(point[mag_range])
        mag_qtrn = qtrn_mult(qtrn_mult(qtrn_conj(curr_pos), acc_qtrn), curr_pos)
        # calculate yaw difference
        yaw_angle_meas = np.arctan2(mag_qtrn[1], mag_qtrn[3])
        yaw_angle_real = np.arctan2(m_ref[1], m_ref[3])
        yaw_diff = yaw_angle_meas - yaw_angle_real
        # repair yaw drift using complementary filter
        comp_filter = euler_to_qtrn(np.append([0,1,0],-ALPHA*yaw_diff))
        curr_pos = qtrn_mult(comp_filter, curr_pos)
    print("> End orientation:",curr_pos)
    return curr_pos

# MAIN:
def main():
    imu_data = get_sanitized_imu_data()
    print()
    end_bad = gyro_dead_reckoning(imu_data)
    print()
    end_better = gyro_acc_positioning(imu_data)
    print()
    end_best = gyro_acc_mag_positioning(imu_data)
    print()

if __name__=="__main__":
    main()