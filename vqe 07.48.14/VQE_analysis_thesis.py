import os
import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from analysis_tools import *
#from skopt import gp_minimize
np.random.seed(1)


zi_freq = np.array([0.00532, -0.00645, 0.00389, 0.00050, -0.00134, 0.00179, ])
iz_freq = np.array([0.00075, 0.00664, -0.00020, -0.00677, 0.00351, 0.00086, ])
zz_freq = np.array([-0.50940, -0.50893, -0.50876, 0.50468, -0.50937, -0.50977, -0.50941, -0.50288, -0.51020, -0.50722, -0.50746, -0.50942, -0.50760, -0.50763, -0.51109, -0.50520]) #-0.38725, -0.44150, -0.50520])
zz = np.average(zz_freq)
print("average zz:", zz)

if 0:
    month = '18-Jun'
    file = '18-13.42.32'
    errors = (np.array([1., 2., 4.]) * 1040. + 480.) / 1520. # Checked

    batch_1 = [32 + i * 2 for i in range(20)]
    batch_2 = [83 + i * 2 for i in range(20)]
    batch_3 = [134 + i * 2 for i in range(20)]
    batch_4 = [185 + i * 2 for i in range(20)]
    batch_5 = [241 + i * 2 for i in range(20)]
    batch_6 = [292 + i * 2 for i in range(5)]

    file_nums = batch_1 + batch_2 + batch_3 + batch_4 + batch_5

if 1: # This looks good
    month = '18-May'
    file = '11-18.50.45'
    errors = (np.array([1., 2., 4., 8., 16.]) * 1040. + 660.) / 1700.

    batch_1 = [19 + i * 10 for i in range(4)]
    batch_2 = [65 + i * 10 for i in range(4)]
    batch_3 = [111 + i * 10 for i in range(4)]
    batch_4 = [157 + i * 10 for i in range(3)]
    batch_5 = [213]


    file_nums = batch_1 + batch_2 + batch_3 + batch_4 + batch_5

if 0:
    month = '18-Jun'
    file = '10-10.43.57'
    errors = (np.array([1., 2.]) * 1200. + 660.) / 1860. # Checked

    batch_1 = [21 + i * 14 for i in range(4)]
    batch_2 = [82 + i * 14 for i in range(4)]
    batch_3 = [143 + i * 14 for i in range(4)]
    batch_4 = [204 + i * 14 for i in range(4)]
    batch_5 = [265 + i * 14 for i in range(4)]
    batch_6 = [326 + i * 14 for i in range(4)]
    batch_7 = [387 + i * 14 for i in range(4)]
    batch_8 = [448 + i * 14 for i in range(4)]
    batch_9 = [509 + i * 14 for i in range(4)]
    batch_10 = [570 + i * 14 for i in range(4)]
    batch_11 = [631 + i * 14 for i in range(4)]
    batch_12 = [692 + i * 14 for i in range(4)]
    batch_13 = [753 + i * 14 for i in range(4)]
    batch_14 = [814 + i * 14 for i in range(4)]
    batch_15 = [875 + i * 14 for i in range(4)]
    batch_16 = [936 + i * 14 for i in range(4)]
    batch_17 = [997 + i * 14 for i in range(4)]
    batch_18 = [1058 + i * 14 for i in range(2)]

    file_nums = batch_1 + batch_2 + batch_3 + batch_4 + batch_5 + \
                batch_6 + batch_7 + batch_8 + batch_9 + batch_10 + \
                batch_11 + batch_12 + batch_13 + batch_14 + batch_15 + \
                batch_16 + batch_17 + batch_18

if 0:
    month = '18-May'
    file = '11-08.34.46'
    errors = (np.array([1., 2., 4., 8., 16.]) * 1040. + 660.) / 1700.

    batch_1 = [18 + i * 4 for i in range(4)]
    batch_2 = [39 + i * 4 for i in range(4)]
    batch_3 = [66 + i * 4 for i in range(4)]
    batch_4 = [87 + i * 4 for i in range(4)]
    batch_5 = [114 + i * 4 for i in range(4)]
    batch_6 = [135 + i * 4 for i in range(4)]

    file_nums = batch_1 + batch_2 + batch_3 + batch_4 + batch_5 + batch_6

# vqe_analysis(file_nums, month, file, zz, errors)


y, y_ave, y_std, x, error_num = import_raw_data(file_nums, month, file)
fit, fit_std, x_fit = full_gp(x, y_ave, y_std, zz)


if 0: # For raw data
    y_min, y_argmin, mit_min, mit_argmin, optimal_theta = energy(y_ave, x, errors, zz)
    y_eb, mit_eb = error_bars(y_argmin, y_std)

else: # For Gaussian Processed data
    y_min, y_argmin, mit_min, mit_argmin, optimal_theta = energy(fit, x_fit, errors, zz)
    y_eb, mit_eb = error_bars(y_argmin, fit_std)

plot_paulis(x, x_fit, y_ave, y_std, fit, fit_std, zz)
plot_energy_curve(y_min, y_eb, mit_min, mit_eb)


"""
for i in range(len(optimal_thetas)):
    print(len(optimal_thetas))
    print("distance:",r_values[i])
    print("optimal theta:",optimal_thetas[i])
    
for i in paulis:
    for j in range(5):
        for k in range(5):
            plt.plot(thetas, i[j][:,k])
        plt.show()
"""

"""
def f(x,noise_level=None):
    index = np.where(thetas==x)
    print(index)
    print(zi_ave.T[0][index] + np.random.randn() * zi_std.T[0][index])
    return zi_ave.T[0][index] + np.random.randn() * zi_std.T[0][index]

res = gp_minimize(f,                  # the function to minimize
                  [(thetas[0], thetas[-1])],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=123)   # the random seed
"""


"""
os.chdir("D:\\oqclab-0.4_logs\\18-May\\11-04.29.24")
start = 24
for i in range(4):
    file_num = start + 3 * i
    if file_num is start:
        directory = "0000%s-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run\\000001-state_tomography_2q_partial_ss.run"%start
        directory_q0 = "0000%s - set_threshold.run" % start+1
        directory_q1 = "0000%s - set_threshold.run" % start+2
        os.chdir(directory)
        q0 = np.load('raw_data_q0.npy')
        q1 = np.load('raw_data_q1.npy')
        os.chdir('..\\..\\..')
        os.chdir(directory_q0)
        q0_base = np.load('results')
        os.chdir('..')
        os.chdir(directory_q1)
        q1_base = np.load('results')


    else:
        if file_num < 100:
            file_num_str_e1 = '0' + '%i' % file_num
            directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run\\000001-state_tomography_2q_partial_ss.run"
            directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"
        else:
            file_num_str_e1 = '%i' % file_num
            directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run"
            directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"

        os.chdir(directory)
        zis.append(np.load('zi.npy')[:,:5])
        izs.append(np.load('iz.npy')[:,:5])
        xxs.append(np.load('xx.npy')[:,:5])
        yys.append(np.load('yy.npy')[:,:5])
        zzs.append(np.load('zz.npy')[:,:5])
        theta_range = np.load('theta_range')
        theta_num = np.load('theta_num')
        error_num = np.load('err_num')
        os.chdir('..')
        os.chdir('..')
os.chdir('..')

"""