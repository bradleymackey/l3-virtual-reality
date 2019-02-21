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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# PROBLEM 1:

def get_raw_imu_data():
    """retrieves raw IMU data from the `IMUData.csv` file"""
    print(">>> Data Extraction <<<")
    data = []
    with open('IMUData.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip header
        for row in reader:
            data.append(row)
    data = np.array(data, dtype=np.float32)
    print(f"> Successfully retrieved {len(data)} items from IMUData.csv.")
    return data

def sanitize_imu_data(data):
    """retrieves the IMU data from the `IMUData.csv` file"""
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

def euler_to_qtrn(axis, angle):
    """converts euler angles (axis-angle repr.) ([x y z], angle) to a quaternion [a b c d]"""
    (x, y, z) = axis
    a = np.cos(angle/2)
    b = np.sin(angle/2) * x
    c = np.sin(angle/2) * y
    d = np.sin(angle/2) * z
    return np.array([a, b, c, d])

def reading_to_qtrn(reading, prev_sample_time):
    """converts a numpy euler rotation sample [time x y z] and prev sample time to a quaternion array [a b c d]"""
    sample_time = float(reading[0] - prev_sample_time)
    sample_time = sample_time if sample_time>0 else (1/256) # first reading assumes 256Hz
    rot_angle = np.linalg.norm(reading[1:]) * sample_time
    rot_axis = np.repeat(1/np.linalg.norm(reading[1:]),3) * reading[1:]
    return euler_to_qtrn(rot_axis, rot_angle)

def qtrn_to_euler(qtrn):
    """converts a numpy quaternion array [a b c d] to an euler angle array (axis of rotation followed by the angle rotated by) ([x y z], theta)"""
    (a, b, c, d) = qtrn
    angle = 2 * np.arccos(a)
    if angle==0.0:
        # identity quaternion/rotation (= no rotation)
        return (np.array([1, 0, 0]), 0)
    divisor = 1/np.sqrt(1-(a**2))
    x, y, z = b/divisor, c/divisor, d/divisor
    axis = np.array([x, y, z]) 
    norm_axis = axis / np.sqrt(np.sum(axis**2))
    return (norm_axis, angle)

def qtrn_conj(qtrn):
    """computes the conjugate of a quaternion, passed as a numpy array [a b c d]"""
    (a, b, c, d) = qtrn
    return np.array([a, -b, -c, -d])

def qtrn_mult(qtrn_1, qtrn_2):
    """computes the product of 2 quaternions, each [a b c d]"""
    (a_1, b_1, c_1, d_1) = qtrn_1
    (a_2, b_2, c_2, d_2) = qtrn_2
    a = a_1*a_2 - b_1*b_2 - c_1*c_2 - d_1*d_2
    b = a_1*b_2 + b_1*a_2 + c_1*d_2 - d_1*c_2
    c = a_1*c_2 - b_1*d_2 + c_1*a_2 + d_1*b_2
    d = a_1*d_2 + b_1*c_2 - c_1*b_2 + d_1*a_2
    return np.array([a, b, c, d])

# PROBLEM 2:

def gyro_dead_reckoning(imu_data):
    """implementation of dead-reckoning filter - estimating position only using the gyro
    sanitized IMU data should be input, progress will be reported"""
    # we start at the identity quaternion
    curr_pos = np.array([1, 0, 0, 0], dtype=np.float32)
    print(">>> Dead Reckoning (Gyro) <<<")
    print("> Start orientation:",curr_pos)
    start_axis, angle = qtrn_to_euler(curr_pos)
    print("start axis",start_axis,"angle",angle)
    gyro_range = range(0,4)

    # time,x,y,z
    gyro_data = []

    prev_sample_time = 0.0
    for point in imu_data:

        ### UPDATE POSITION
        # update the position using the gyroscope
        delta_qtrn = reading_to_qtrn(point[gyro_range], prev_sample_time)
        prev_sample_time = point[0]
        curr_pos = qtrn_mult(delta_qtrn, curr_pos)

        gyro_data.append([prev_sample_time] + curr_pos.tolist())

    print("> End orientation:",curr_pos)
    end_axis, angle = qtrn_to_euler(curr_pos)
    print("end axis",end_axis,"angle",angle)
    print("end check:",np.linalg.norm(curr_pos))
    return gyro_data

# PROBLEM 3:

def gyro_acc_positioning(imu_data):
    """computes current position using data both from the gyroscope and accelerometer"""
    ALPHA_ACC = 0.001

    print(">>> Tilt Correction <<<")
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    print("> Start orientation:",curr_pos)
    gyro_range = range(0,4)
    acc_range = range(4,7)
    ref_vector = np.array([0,1,0], dtype=np.float32)
    end_axis, angle = qtrn_to_euler(curr_pos)
    print("start axis",end_axis,"angle",angle)

    # time,x,y,z
    gyro_data = []

    prev_sample_time = 0.0
    for point in imu_data:

        ### UPDATE POSITION
        ### calculate initial position only using the gyro
        delta_qtrn = reading_to_qtrn(point[gyro_range], prev_sample_time)
        prev_sample_time = point[0]
        curr_pos = qtrn_mult(delta_qtrn, curr_pos)

        ### PITCH/TILT CORRECTION
        ### convert acc data to the global frame
        # (this is the inverse of the head orientation!)
        # therefore, we compute q^-1•p•q rather than q•p•q^-1
        acc_qtrn = np.insert(point[acc_range], 0, 0)
        acc_qtrn = qtrn_mult(qtrn_mult(qtrn_conj(curr_pos), acc_qtrn), curr_pos)
        ### calculate the tilt error
        # x = index 1, z = index 3 (we disregard the first element, it is not needed for rotating a simple point as we just rotated acc to the global frame)
        tilt_error_axis = np.array([acc_qtrn[3], 0.0, -acc_qtrn[1]])
        cos_ang = np.dot(ref_vector, acc_qtrn[1:])
        # account for any rounding errors
        cos_ang = cos_ang if cos_ang > -1 else -1
        cos_ang = cos_ang if cos_ang < 1 else 1
        tilt_error_angle = np.arccos(cos_ang)
        ### Repair tilt using the comp filter
        comp_filter = euler_to_qtrn(tilt_error_axis, -ALPHA_ACC*tilt_error_angle)
        # fix our current estimated position using acceleration data
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        gyro_data.append([prev_sample_time] + curr_pos.tolist())


    print("> End orientation:",curr_pos)
    end_axis, angle = qtrn_to_euler(curr_pos)
    print("end axis",end_axis,"angle",angle)

    print("end check:",np.linalg.norm(curr_pos))
    return gyro_data

# PROBLEM 4:

def gyro_acc_mag_positioning(imu_data):
    """corrects for tilt and yaw using the accelerometer and magnetometer"""
    ALPHA_ACC = 0.001
    ALPHA_YAW = 0.001

    print(">>> Tilt and Yaw Correction <<<")
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    print("> Start orientation:",curr_pos)
    gyro_range = range(0,4)
    acc_range = range(4,7)
    mag_range = range(7,10)
    ref_vector = np.array([0,1,0], dtype=np.float32)

    # take reference measurement for yaw correction
    m_ref = np.insert(imu_data[0,mag_range], 0, 0)
    # transform m_ref to the global frame
    # (this is the inverse of the head orientation!)
    # therefore, we compute q^-1•p•q rather than q•p•q^-1
    m_ref = qtrn_mult(qtrn_mult(qtrn_conj(curr_pos), m_ref), curr_pos)

    # time,x,y,z
    gyro_data = []

    prev_sample_time = 0.0
    for point in imu_data:

        ### UPDATE POSITION
        ### calculate initial position only using the gyro
        delta_qtrn = reading_to_qtrn(point[gyro_range], prev_sample_time)
        prev_sample_time = point[0]
        curr_pos = qtrn_mult(delta_qtrn, curr_pos)

        ### PITCH/TILT CORRECTION
        ### convert acc data to the global frame
        # (this is the inverse of the head orientation!)
        # therefore, we compute q^-1•p•q rather than q•p•q^-1
        acc_qtrn = np.insert(point[acc_range], 0, 0)
        acc_qtrn = qtrn_mult(qtrn_mult(qtrn_conj(curr_pos), acc_qtrn), curr_pos)
        ### calculate the tilt error
        # x = index 1, z = index 3 (we disregard the first element, it is not needed for rotating a simple point as we just rotated acc to the global frame)
        tilt_error_axis = np.array([acc_qtrn[3], 0.0, -acc_qtrn[1]])
        cos_ang = np.dot(ref_vector, acc_qtrn[1:])
        # account for any rounding errors
        cos_ang = cos_ang if cos_ang > -1 else -1
        cos_ang = cos_ang if cos_ang < 1 else 1
        tilt_error_angle = np.arccos(cos_ang)
        ### Repair tilt using the comp filter
        comp_filter = euler_to_qtrn(tilt_error_axis, -ALPHA_ACC*tilt_error_angle)
        # fix our current estimated position using acceleration data
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        ### YAW CORRECTION
        ### convert mag data to the global frame
        # (this is the inverse of the head orientation!)
        # therefore, we compute q^-1•p•q rather than q•p•q^-1
        mag_qtrn = np.insert(point[mag_range], 0, 0)
        mag_qtrn = qtrn_mult(qtrn_mult(qtrn_conj(curr_pos), mag_qtrn), curr_pos)
        # calculate yaw difference
        # x = index 1, z = index 3 (first element is just 0, which we don't care about - it was only used to make up the length for a quaternion)
        yaw_angle_meas = np.arctan2(mag_qtrn[1], mag_qtrn[3])
        yaw_angle_real = np.arctan2(m_ref[1], m_ref[3])
        yaw_diff = yaw_angle_meas - yaw_angle_real
        # repair yaw drift using complementary filter
        tilt_yaw_axis = np.array([0, 1, 0])
        comp_filter = euler_to_qtrn(tilt_yaw_axis, -ALPHA_YAW*yaw_diff)
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        gyro_data.append([prev_sample_time] + curr_pos.tolist())

    print("> End orientation:",curr_pos)
    print("end check:",np.linalg.norm(curr_pos))
    return gyro_data


# PROBLEM 5:

def save_unmodified_figs(raw_data):
    """saves figures for the unmodified data"""
    time_data = []
    x_data = []
    y_data = []
    z_data = []
    for point in raw_data:
        time_data.append(point[0])
        x_data.append(point[1])
        y_data.append(point[2])
        z_data.append(point[3])
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Rate (degs^-1)")
    plt.title("Raw Gyroscope Readings")
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    plt.savefig("gyro-unaltered.png")

    time_data = []
    x_data = []
    y_data = []
    z_data = []
    for point in raw_data:
        time_data.append(point[0])
        x_data.append(point[4])
        y_data.append(point[5])
        z_data.append(point[6])
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (ms^2)")
    plt.title("Raw Accelerometer Readings")
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    plt.savefig("acc-unaltered.png")

    time_data = []
    x_data = []
    y_data = []
    z_data = []
    for point in raw_data:
        time_data.append(point[0])
        x_data.append(point[7])
        y_data.append(point[8])
        z_data.append(point[9])
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetic Flux (G)")
    plt.title("Raw Magnetometer Readings")
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    plt.savefig("mag-unaltered.png")

def save_gyro_fig(name, title, data):
    """saves gyro data for a given filename and data"""
    time_data = []
    x_data = []
    y_data = []
    z_data = []
    for point in data:
        time_data.append(point[0])
        x_data.append(point[1]/(np.pi/180))
        y_data.append(point[2]/(np.pi/180))
        z_data.append(point[3]/(np.pi/180))
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Rate (degs^-1)")
    plt.title(title)
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='lower left', shadow=True, fontsize='small')
    plt.savefig(f"{name}.png")

def test():
    axis = [0.2,1.12,2.31]
    q = euler_to_qtrn(axis, 1.32)
    print("q:",q)
    r = qtrn_to_euler(q)
    print("r:",r)

# MAIN:
def main():
    test()
    raw_data = get_raw_imu_data()
    save_unmodified_figs(raw_data)
    imu_data = sanitize_imu_data(raw_data)

    gyro_data = gyro_dead_reckoning(imu_data)
    save_gyro_fig("gyro_dead", "Dead-Reckoning (Gyro)", gyro_data)

    acc_data = gyro_acc_positioning(imu_data)
    save_gyro_fig("gyro_acc", "Tilt Correction (Gyro + Acc)", acc_data)
    
    mag_data = gyro_acc_mag_positioning(imu_data)
    save_gyro_fig("gyro_acc_mag", "Tilt & Yaw Correction (Gyro + Acc + Mag)", mag_data)

if __name__=="__main__":
    main()