
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import time
import numpy as np

def read_data(text):
    x = []
    y = []
    z = []
    w = []
    with open(text,'r') as file:
        rad = csv.DictReader(file)
        # xt = rad[' pos.x']
        for r in rad:
            tmp = r[' pos.x']
            tmp = float(tmp.strip())
            x.append(tmp)
            tmp = r[' pos.y']
            tmp = float(tmp.strip())
            y.append(tmp)

            # tmp = r[' quat.z']
            # tmp = float(tmp.strip())
            # z.append(tmp)
            # tmp = r[' quat.w']
            # tmp = float(tmp.strip())
            # w.append(tmp)


    return x, y

def main():

    root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/record_pf2train_gp/results_2020_11_7___12_18_9'

    xt, yt = read_data(root + '/true_pose.csv')
    xm, ym = read_data(root + '/measured_pose.csv')
    # xp, yp = read_data(root + '/particle_poses.csv')


    xt_arr = np.asarray(xt)
    yt_arr = np.asarray(yt)
    xm_arr = np.asarray(xm)
    ym_arr = np.asarray(ym)

    dist = np.sqrt((xt_arr[1:] - xm_arr)**2 + (yt_arr[1:] - ym_arr)**2)
    # dist = np.sqrt((xt_arr - xm_arr[1:] )**2 + (yt_arr - ym_arr[1:])**2)
    MSE = np.mean(dist**2)
    err_max = np.amax(dist)
    print("MSE     =", MSE)
    print("Err_max =", err_max)
    # xt, yt, zt, wt = read_data('true_pose.csv')
    # xm, ym, zm, wm = read_data('measured_pose.csv')
    # xp, yp, zm, wm = read_data('particle_poses.csv')


    """ XY Plot """
    """
    # print('{:.2f},   {:.2f}'.format(xt,yt))
    # plt.figure(figsize=(10,4))
    plt.plot(xt,yt, 'b')
    plt.plot(xm,ym, 'g')
    # plt.scatter(xp, yp, alpha=0.5)
    plt.xlabel('x axis (m)')
    plt.ylabel('y axis (m)')
    plt.legend(['True pose', 'Measured pose'])
    # plt.legend(['True pose', 'Measured pose', 'Particle pose'])
    plt.axis('auto')
    plt.tight_layout()
    # plt.savefig(root+'/true_and_meas.png')
    plt.show()
    """


    """ XY Plot with error """

    # plt.subplot(2, 1, 2)
    # plt.plot(dist)
    # plt.axis('auto')

    fig = plt.figure(figsize=(6,6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:2, :])
    plt.plot(xt,yt, 'b')
    plt.plot(xm,ym, 'g')
    plt.xlabel('x axis (m)')
    plt.ylabel('y axis (m)')
    plt.legend(['True pose', 'Measured pose'])
    ax2 = fig.add_subplot(gs[2, :])
    plt.plot(dist, 'r')
    plt.xlabel('Ping number')
    plt.ylabel('Error (m)')

    plt.tight_layout()
    plt.savefig(root+'/true_and_meas_with_err.png')
    plt.show()


main()