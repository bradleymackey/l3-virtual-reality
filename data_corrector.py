# data_corrector.py
# Bradley Mackey
# Virtual Reality coursework 18/19

####################
# tested on Python 3.7.2
# ensure that the `IMUData.csv` file is located in the same directory as this script
# to see the 3d visualisations, go down to the `main()` function and un-comment them (this may take a very long time to run though - which is why they are initially commented out)
# (my computer was only able to render at 1/40 speed)
###################

import csv
import numpy as np
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

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

def axis_angle_to_qtrn(axis, angle):
    """converts axis-angle repr. ([x y z], angle) to a quaternion [a b c d]"""
    (x, y, z) = axis
    a = np.cos(angle/2)
    b = np.sin(angle/2) * x
    c = np.sin(angle/2) * y
    d = np.sin(angle/2) * z
    return np.array([a, b, c, d])

def euler_to_qtrn(angles):
    """converts euler angles [x, y, z] to a quaternion"""
    (x, y, z) = angles
    c_1 = np.cos(y/2)
    c_2 = np.cos(x/2)
    c_3 = np.cos(z/2)
    s_1 = np.sin(y/2)
    s_2 = np.sin(x/2)
    s_3 = np.sin(z/2)
    c1c2 = c_1*c_2
    s1s2 = s_1*s_2
    a = c1c2*c_3 - s1s2*s_3
    b = c1c2*s_3 + s1s2*c_3
    c = s_1*c_2*c_3 + c_1*s_2*s_3
    d = c_1*s_2*c_3 - s_1*c_2*s_3
    return np.array([a, b, c, d])


def reading_to_qtrn(reading, prev_sample_time):
    """converts a numpy euler rotation sample [time x y z] and prev sample time to a quaternion array [a b c d]"""
    sample_time = float(reading[0] - prev_sample_time)
    sample_time = sample_time if sample_time>0 else (1/256) # first reading assumes 256Hz
    rot_angle = np.linalg.norm(reading[1:]) * sample_time
    rot_axis = np.repeat(1/np.linalg.norm(reading[1:]),3) * reading[1:]
    return axis_angle_to_qtrn(rot_axis, rot_angle)


def qtrn_to_euler(qtrn):
    """converts a numpy quaternion array [a b c d] to a raw euler angle array [x y z]"""
    (a, b, c, d) = qtrn
    x = np.arcsin(2*b*c + 2*d*a)
    y = np.arctan2(2*c*a-2*b*d, 1 - 2*(c**2) - 2*(d**2))
    z = np.arctan2(2*b*a-2*c*d , 1 - 2*(b**2) - 2*(d**2))
    return np.array([x, y, z])

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

    print("> Completed")
    #print("> End orientation:",curr_pos)
    #print("end check:",np.linalg.norm(curr_pos))
    return gyro_data

# PROBLEM 3:

def gyro_acc_positioning(imu_data):
    """computes current position using data both from the gyroscope and accelerometer"""
    ALPHA_ACC = 0.01

    print(">>> Tilt Correction <<<")
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    gyro_range = range(0,4)
    acc_range = range(4,7)

    # z is clearly the up vector based on the raw accelerometer readings
    up_vec = np.array([0,0,1])
    print("true up:",up_vec)
    
    # [[time,x,y,z]]
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
        acc_qtrn = np.insert(point[acc_range], 0, 0)
        #inv_curr = qtrn_conj(curr_pos) / np.repeat(np.linalg.norm(curr_pos)**2,4)
        acc_qtrn = qtrn_mult(qtrn_mult(curr_pos, acc_qtrn), qtrn_conj(curr_pos))
        acc_vec = acc_qtrn[1:]
        #acc_vec = np.repeat(1/np.linalg.norm(acc_vec),3) * acc_vec
        ### calculate the tilt error
        x, y = acc_vec[0], acc_vec[1]
        tilt_error_axis = np.array([-y, x, 0])
        tilt_error_axis = np.repeat(1/np.linalg.norm(tilt_error_axis),3) * tilt_error_axis
        cos_ang = np.dot(up_vec, acc_vec)
        # possible rounding errors
        cos_ang = cos_ang if cos_ang > -1 else -1
        cos_ang = cos_ang if cos_ang < +1 else +1
        tilt_error_angle = np.arccos(cos_ang)
        #print("error angle:",tilt_error_angle)
        ### Repair tilt using the comp filter
        comp_filter = axis_angle_to_qtrn(tilt_error_axis, -ALPHA_ACC*tilt_error_angle)
        # fix our current estimated position using acceleration data
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        gyro_data.append([prev_sample_time] + curr_pos.tolist())

    print("> Completed")
    #print("> End orientation:",curr_pos)
    #print("end check:",np.linalg.norm(curr_pos))
    return gyro_data

# PROBLEM 4:

def gyro_acc_mag_positioning(imu_data):
    """corrects for tilt and yaw using the accelerometer and magnetometer"""
    ALPHA_ACC = 0.001
    ALPHA_YAW = 0.001

    print(">>> Tilt and Yaw Correction <<<")
    curr_pos = np.array([1,0,0,0], dtype=np.float32)
    gyro_range = range(0,4)
    acc_range = range(4,7)
    mag_range = range(7,10)

    # z is clearly the up vector based on the raw accelerometer readings
    up_vec = np.array([0,0,1])
    print("true up:",up_vec)

    # take reference measurement for yaw correction
    m_ref = np.insert(imu_data[0,mag_range], 0, 0)
    # transform m_ref to the global frame
    m_ref = qtrn_mult(qtrn_mult(curr_pos, m_ref), qtrn_conj(curr_pos))

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
        acc_qtrn = np.insert(point[acc_range], 0, 0)
        #inv_curr = qtrn_conj(curr_pos) / np.repeat(np.linalg.norm(curr_pos)**2,4)
        acc_qtrn = qtrn_mult(qtrn_mult(curr_pos, acc_qtrn), qtrn_conj(curr_pos))
        acc_vec = acc_qtrn[1:]
        acc_vec = np.repeat(1/np.linalg.norm(acc_vec),3) * acc_vec
        ### calculate the tilt error
        x, y = acc_vec[0], acc_vec[1]
        tilt_error_axis = np.array([-y, x, 0])
        tilt_error_axis = np.repeat(1/np.linalg.norm(tilt_error_axis),3) * tilt_error_axis
        cos_ang = np.dot(up_vec, acc_vec)
        # possible rounding errors
        cos_ang = cos_ang if cos_ang > -1 else -1
        cos_ang = cos_ang if cos_ang < +1 else +1
        tilt_error_angle = np.arccos(cos_ang)
        ### Repair tilt using the comp filter
        comp_filter = axis_angle_to_qtrn(tilt_error_axis, -ALPHA_ACC*tilt_error_angle)
        # fix our current estimated position using acceleration data
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        ### YAW CORRECTION
        ### convert mag data to the global frame
        # (this is the inverse of the head orientation!)
        # therefore, we compute q^-1•p•q rather than q•p•q^-1
        mag_qtrn = np.insert(point[mag_range], 0, 0)
        mag_qtrn = qtrn_mult(qtrn_mult(curr_pos, mag_qtrn), qtrn_conj(curr_pos))
        # calculate yaw difference
        # z is the up axis, so that is what we yaw around (angle around z was not corrected during tilt correction) 
        # - in our XYZ representation though, WE still see X as the tilt on the graph so the fact that this yaw correction is not applied around the Z axis until this point explains why X drifts in the tilt correction!
        meas_x, meas_y = mag_qtrn[1], mag_qtrn[2]
        yaw_angle_meas = np.arctan2(meas_y, meas_x)
        real_x, real_y = m_ref[1], m_ref[2]
        yaw_angle_real = np.arctan2(real_y, real_x)
        yaw_diff = yaw_angle_meas - yaw_angle_real
        # repair yaw drift using complementary filter
        tilt_yaw_axis = np.array([0, 0, 1])
        comp_filter = axis_angle_to_qtrn(tilt_yaw_axis, -ALPHA_YAW*yaw_diff)
        curr_pos = qtrn_mult(comp_filter, curr_pos)

        gyro_data.append([prev_sample_time] + curr_pos.tolist())

    print("> Completed")
    #print("> End orientation:",curr_pos)
    #print("end check:",np.linalg.norm(curr_pos))
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
    plt.ylabel("Angular Rate (deg/s)")
    plt.title("Raw Gyroscope Readings")
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    plt.grid()
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
    plt.ylabel("Acceleration (g)")
    plt.title("Raw Accelerometer Readings")
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='upper left', shadow=True, fontsize='small')
    plt.grid()
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
    plt.grid()
    plt.savefig("mag-unaltered.png")

def save_gyro_fig(name, title, data):
    """saves gyro data for a given filename and data"""
    time_data = []
    x_data = []
    y_data = []
    z_data = []
    for point in data:
        time_data.append(point[0])
        # convert the rotation quaternion to euler angles
        x, y, z = qtrn_to_euler(point[1:5])
        x_data.append(x/(np.pi/180))
        y_data.append(y/(np.pi/180))
        z_data.append(z/(np.pi/180))
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel("Time (s)")
    plt.ylabel("Euler Angle (degs)")
    plt.title(title)
    plt.grid()
    ax.plot(time_data, x_data, 'r-', label="X")
    ax.plot(time_data, y_data, 'g-', label="Y")
    ax.plot(time_data, z_data, 'b-', label="Z")
    legend = ax.legend(loc='lower left', shadow=True, fontsize='small')
    plt.savefig(f"{name}.png")

def animated_3d_plot(data):
    """produces a 3d animated plot of the given imu data"""
    plt.clf()

    origin = [0,0,0]
    x_basis = [1,0,0]
    y_basis = [0,1,0]
    z_basis = [0,0,1]
    soa = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    red = (1,0,0)
    green = (0,1,0)
    blue = (0,0,1)
    colors = [red, green, blue, red, red, green, green, blue, blue]

    X, Y, Z, U, V, W = zip(*soa)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, colors=colors)

    current_position = 0

    ax.set_title("Dead Reckoning")
    
    ax.set_xlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylim([-1, 1])
    ax.set_ylabel('Y')
    ax.set_zlim([-1, 1])
    ax.set_zlabel('Z')

    def update_position(point):
        x, y, z = qtrn_to_euler(point[1:5])
        print("TIME:",point[0])
        print(x,y,z)
        new_x, new_y, new_z = [0] + x_basis, [0] + y_basis, [0] + z_basis
        #print(new_x, new_y, new_z)
        axis_angles = [(x_basis, x), (y_basis, y), (z_basis, z)]
        for axis, angle in axis_angles:
            # reset the real part of the quaternion, so we can get an accurate rotation again
            new_x[0], new_y[0], new_z[0]  = 0, 0, 0
            point_qtrn = axis_angle_to_qtrn(axis, angle)
            #print(point_qtrn)
            new_x, new_y, new_z = list(map(lambda qtrn: qtrn_mult( qtrn_mult(point_qtrn, qtrn), qtrn_conj(point_qtrn)), [new_x, new_y, new_z]))
        # get just the coordinates from quaternion
        new_x, new_y, new_z = list(new_x[1:]), list(new_y[1:]), list(new_z[1:])
        #print(new_x, new_y, new_z)
        soa = np.array([origin + new_x, origin + new_y, origin + new_z])
        #print(soa)
        X, Y, Z, U, V, W = zip(*soa)
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylim([-1, 1])
        ax.set_ylabel('Y')
        ax.set_zlim([-1, 1])
        ax.set_zlabel('Z')
        ax.quiver(X, Y, Z, U, V, W, colors=colors)
    

    # Creating the Animation object
    move_ani = animation.FuncAnimation(fig, update_position, frames=data,
                                        interval=1)
    plt.show()


# MAIN:
def main():
    
    # get the raw data from the csv file
    raw_data = get_raw_imu_data()
    # save the raw data from the csv
    save_unmodified_figs(raw_data)
    # sanitise the data so that we can work with it effectively
    imu_data = sanitize_imu_data(raw_data)

    gyro_data = gyro_dead_reckoning(np.array(imu_data, copy=True))
    save_gyro_fig("gyro_dead", "Dead-Reckoning (Gyro)", gyro_data)
    #animated_3d_plot(gyro_data)

    acc_data = gyro_acc_positioning(np.array(imu_data, copy=True))
    save_gyro_fig("gyro_acc", "Tilt Correction (Gyro + Acc)", acc_data)
    #animated_3d_plot(acc_data)

    mag_data = gyro_acc_mag_positioning(imu_data)
    save_gyro_fig("gyro_acc_mag", "Tilt & Yaw Correction (Gyro + Acc + Mag)", mag_data)
    #animated_3d_plot(mag_data)

if __name__=="__main__":
    main()