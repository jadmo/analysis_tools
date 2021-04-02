import os
import numpy as np
from matplotlib import pylab as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# class vqe_analysis():
#
#     def __init__(self, file_nums, month, file, zz, errors):
#
#         self.vqe_parameters()
#         self.zz = zz
#
#         os.chdir('../../../..')
#         self.import_raw_data(file_nums, month, file)
#         self.full_gp()
#
#         self.y_min, self.y_argmin, self.mit_min, self.mit_argmin, self.optimal_theta = energy(self.y_ave, self.x, errors, self.zz)
#         self.y_eb, self.mit_eb = error_bars(self.y_argmin, self.y_std)
#
#     def vqe_parameters(self):
#         file1 = open("r-theta-energy.txt", "r")
#         line1 = file1.readlines()
#         self.r_values = []
#         self.energies = []
#         for x in line1:
#             self.r_values.append(float(x.split(' ')[0]))
#             self.energies.append(float(x.split(' ')[4]))
#         file1.close()
#
#         file2 = open("dataH2N.txt", "r")
#         line2 = file2.readlines()
#         g0 = [];
#         g1 = [];
#         g2 = [];
#         g3 = [];
#         g4 = [];
#         g5 = []
#
#         self.gs = [g0, g1, g2, g3, g4, g5]
#         for x in line2:
#             for i in range(6):
#                 self.gs[i].append(float(x.split(' ')[i + 1]))
#         file2.close()
#
#     def exponential(self, x, a, b, c):
#         return a * np.exp(-b * x) + c
#
#     def linear(self, x, a, b):
#         return a * x + b
#
#     def import_raw_data(self, file_nums, month, file):
#         zis = [];
#         izs = [];
#         zzs = [];
#         xxs = [];
#         yys = []
#         os.chdir("oqclab-0.4_logs/" + month + '/' + file)
#         for i, file_num in enumerate(file_nums):
#
#             if i is 0:
#                 directory = "0000%s-H2_simulation_echo_2d_sweep_qst.__init__/000001-H2_simulation_echo_2d_sweep_qst.run" % file_num
#             else:
#                 if file_num < 100:
#                     file_num_str_e1 = '0' + '%i' % file_num
#                     directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run"  # /000001-H2_simulation_echo_2d_sweep_qst.run"
#                     # directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"
#                 else:
#                     file_num_str_e1 = '%i' % file_num
#                     directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run"  # /000001-H2_simulation_echo_2d_sweep_qst.run"
#                     # directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"
#
#             os.chdir(directory)
#             zis.append(np.load('zi.npy')[:, :5])
#             izs.append(np.load('iz.npy')[:, :5])
#             xxs.append(np.load('xx.npy')[:, :5])
#             yys.append(np.load('yy.npy')[:, :5])
#             zzs.append(np.load('zz.npy')[:, :5])
#
#             if i is 0:
#                 os.chdir('../..')
#
#             elif i is len(file_nums) - 1:
#                 pass
#
#             else:
#                 os.chdir('..')
#
#         theta_range = np.load('theta_range', allow_pickle=True)
#         theta_num = np.load('theta_num', allow_pickle=True)
#         self.error_num = np.load('err_num', allow_pickle=True)
#         os.chdir('..')
#
#         N = len(file_nums)  # 100 - (clean * 2)
#
#         self.y = np.array([zis, izs, xxs, yys, zzs])
#         self.y_ave = np.average(self.y, axis=1)
#         self.y_std = np.std(self.y, axis=1) / np.sqrt(N)
#         self.x = np.linspace(-theta_range, theta_range, theta_num)
#
#     def gp_regression(self, x, y_ave, y_std, points):
#         # Training set
#         X = np.atleast_2d(x).T
#         y = y_ave
#         dy = y_std
#         noise = np.random.normal(0, dy)
#         y += noise
#         # Test set
#         x = np.atleast_2d(np.linspace(x[0], x[-1], points)).T
#         kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#         gp = GaussianProcessRegressor(kernel=kernel, alpha=(dy / y) ** 2,
#                                       n_restarts_optimizer=10)
#         gp.fit(X, y)
#
#         fit, fit_std = gp.predict(x, return_std=True)
#         if 0:
#             fig = plt.figure()
#             plt.errorbar(X, y, dy, fmt='r.', markersize=10, label=u'Observations')
#             plt.plot(x, fit, 'b-', label=u'Prediction')
#             plt.fill(np.concatenate([x, x[::-1]]),
#                      np.concatenate([fit - 1.9600 * fit_std,
#                                      (fit + 1.9600 * fit_std)[::-1]]),
#                      alpha=.5, fc='b', ec='None', label='95% confidence interval')
#             plt.xlabel('$x$')
#             plt.ylabel('$f(x)$')
#             plt.ylim(-1, 1)
#             plt.legend(loc='upper left')
#
#         return fit, fit_std
#
#     def full_gp(self):
#         s = np.shape(self.y_ave)
#         points = 1000
#         s = (s[0], points, s[2])
#         self.fit = np.zeros(s)
#         self.fit_std = np.zeros(s)
#
#         for i in range(s[0]):
#             for j in range(s[2]):
#                 self.fit[i, :, j], self.fit_std[i, :, j] = self.gp_regression(self.x * 2 * self.zz, self.y_ave[i, :, j], self.y_std[i, :, j],
#                                                                points)
#
#         self.x_fit = np.linspace(self.x[0], self.x[-1], 1000)
#
#     def fit_and_plot(self, data_ave, thetas, errors, zz_freq, lin_plot=0, exp_plot=0, data_plot=0, show=0):
#         points = 2
#
#         errors_ext = np.insert(self.errors, 0, 0)
#         lin = []
#         exp = []
#         for i in range(len(data_ave)):
#             popt_l, pcov_l = curve_fit(self.linear, self.errors[0:points], data_ave[i][0:points])
#             lin.append(popt_l[1])
#             if lin_plot:
#                 plt.plot(errors_ext, self.linear(errors_ext, popt_l[0], popt_l[1]))
#             if exp_plot:
#                 popt_e, pcov_e = curve_fit(self.exponential, self.errors, data_ave[i], p0=[popt_l[1], -popt_l[0] / popt_l[1],
#                                                                                  0])  # ,bounds=([-2., -1., -1], [2., 1., 1.]))
#                 exp.append(popt_e[0] + popt_e[2])
#                 plt.plot(errors_ext, self.exponential(errors_ext, popt_e[0], popt_e[1], popt_e[2]))
#             else:
#                 exp.append(popt_l[1])
#             if data_plot:
#                 plt.plot(self.errors, data_ave[i])
#             if lin_plot or exp_plot or data_plot:
#                 plt.show()
#
#         if show:
#             for j in range(len(data_ave[0])):
#                 plt.plot(thetas * 2. * zz_freq, data_ave[:, j])
#             plt.plot(thetas * 2. * zz_freq, lin_extrapolated)
#             if exp_plot:
#                 plt.plot(thetas * 2. * zz_freq, exp_extrapolated)
#             plt.xlabel(r'$Rotation\ angle\ /\ 2 \pi $')
#             plt.ylabel('Eigenvalue')
#             plt.show()
#
#         return lin, exp
#
#     def fit_all(self, pauli_ave, thetas, errors, zz_freq):
#         s = np.shape(pauli_ave)
#         s = (s[0], s[1])
#         lin_extrapolated = np.zeros(s)
#         exp_extrapolated = np.zeros(s)
#         print(np.shape(lin_extrapolated))
#         for i in range(5):
#             lin_extrapolated[i], exp_extrapolated[i] = fit_and_plot(pauli_ave[i], thetas, errors, zz_freq,
#                                                                     lin_plot=0,
#                                                                     exp_plot=0, data_plot=0, show=0)
#
#         return lin_extrapolated, exp_extrapolated
#
#     def energy(self, y_ave, x, errors, zz_freq):
#         fig = plt.figure(0)
#         extrap_lin, extrap_exp = fit_all(y_ave, x, errors, zz_freq)
#
#         y_min = []
#         y_argmin = []
#
#         for j in range(2):
#             min_datum = []
#             argmin_datum = []
#             mit_min = []
#             mit_argmin = []
#             optimal_theta = []
#             for i in range(len(self.gs[0])):
#                 data_energy = self.gs[0][i] + self.gs[1][i] * y_ave[0, :, j] + self.gs[2][i] * y_ave[1, :, j] + \
#                               self.gs[3][i] * y_ave[4,:, j] + self.gs[4][i] * y_ave[2, :, j] + self.gs[5][i] * y_ave[3, :, j]
#                 extrap_energy = self.gs[0][i] + self.gs[1][i] * np.array(extrap_lin[0]) + self.gs[2][i] * np.array(extrap_lin[1]) + \
#                                 self.gs[3][i] * np.array(extrap_lin[4]) + self.gs[4][i] * np.array(extrap_lin[2]) + self.gs[5][
#                                     i] * np.array(extrap_lin[3])
#
#                 min_datum.append(np.amin(data_energy))
#                 argmin_datum.append(np.argmin(data_energy))
#
#                 mit_min.append(np.amin(extrap_energy))
#                 mit_argmin.append(np.argmin(extrap_energy))
#                 if j is 0:
#                     plt.plot(x * 2. * zz_freq, data_energy)
#
#                 # print("error:", 2 ** (j))
#                 # print("distance:", r_values[i])
#                 # print("optimal theta:", thetas[np.argmin(data_energy)] * 2. * zz_freq)
#                 # print("minimum value:", np.amin(data_energy))
#                 # print("linear extrapolated")
#                 # print("distance:", r_values[i])
#                 # print("optimal theta:", thetas[np.argmin(extrap_energy)] * 2. * zz_freq)
#                 # print("minimum value:", np.amin(extrap_energy))
#
#                 optimal_theta.append(x[np.argmin(extrap_energy)] * 2. * zz_freq)
#
#             y_min.append(min_datum)
#             y_argmin.append(argmin_datum)
#         fig.show()
#         return y_min, y_argmin, mit_min, mit_argmin, optimal_theta
#
#     def error_bars(self, y_argmin, y_std):
#
#         y_eb = []
#         for j in range(2):
#             error_bars = []
#             for i in range(len(self.gs[0])):
#                 min = y_argmin[j][i]
#                 error_bar = np.sqrt((self.gs[1][i] * y_std[0][min, j]) ** 2 + (self.gs[2][i] * y_std[1][min, j]) ** 2 + (
#                         self.gs[3][i] * y_std[4][min, j]) ** 2 + (self.gs[4][i] * y_std[2][min, j]) ** 2 + (
#                         self.gs[5][i] * y_std[3][min, j]) ** 2)
#                 error_bars.append(error_bar)
#
#             y_eb.append(error_bars)
#
#         mit_eb = []
#         for i in range(len(self.gs[0])):
#             lin_error_bar = np.sqrt((2 * np.array(y_eb[0])[i]) ** 2 + (np.array(y_eb[1])[i]) ** 2)
#             mit_eb.append(lin_error_bar)
#
#         return y_eb, mit_eb
#
#     def plot_paulis(self, x, x_fit, y_ave, y_std, fit, fit_std, zz_freq):
#         fig = plt.figure(0)
#
#         labels = ['ZI', 'IZ', 'XX', 'YY', 'ZZ']
#         colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#         for i in range(5):
#             subplot = fig.add_subplot(5, 1, i + 1)
#             for j in range(np.shape(y_ave)[2]):
#                 subplot.errorbar(x * 2. * zz_freq, y_ave[i, :, j], yerr=y_std[i, :, j], color=colors[j], fmt='.')
#                 subplot.errorbar(x_fit * 2. * zz_freq, fit[i, :, j], yerr=fit_std[i, :, j], color=colors[j])
#             subplot.set(xlabel=r'$Rotation\ angle\ /\ 2 \pi $', ylabel=labels[i])
#             subplot.set_ylim(-1.1, 1.1)
#
#         plt.show()
#
#         # for i in range(5):
#         #     for j in range(np.shape(pauli_ave)[2]):
#         #         plt.errorbar(thetas,pauli_ave[i, :, j], yerr=pauli_std[i, :, j], fmt='o')
#         #         plt.errorbar(thetas_gp, y_preds[i, :, j], yerr=sigmas[i, :, j])
#         #     plt.xlabel('Rotation angle')
#         #     plt.ylabel(labels[i])
#         #     plt.ylim(-1.1, 1.1)
#         #     plt.show()
#
#     def plot_energy_curve(self, y_min, y_eb, mit_min, mit_eb):
#
#         fig = plt.figure(0)
#
#         plt.plot(self.r_values, self.energies)  # Theory curve
#         for j in range(2):
#             plt.errorbar(self.r_values, y_min[j], marker='o', markersize=1, linestyle='None', yerr=y_eb[j],
#                          label="Error %s times" % 2 ** j)
#
#         plt.errorbar(self.r_values, mit_min, marker='o', markersize=1, linestyle='None', yerr=mit_eb,
#                      label="Linear extrapolation")
#         plt.legend()
#         plt.xlabel(r'Bond distance (Angstrom)', fontsize=25)
#         plt.ylabel(r'Total Energy (Hatree)', fontsize=25)
#         plt.show()
#

def vqe_parameters():
    file1 = open("r-theta-energy.txt", "r")
    line1 = file1.readlines()
    r_values = []
    energies = []
    for x in line1:
        r_values.append(float(x.split(' ')[0]))
        energies.append(float(x.split(' ')[4]))
    file1.close()

    file2 = open("dataH2N.txt", "r")
    line2 = file2.readlines()
    g0 = []; g1 = []; g2 = []; g3 = []; g4 = []; g5 = []

    gs = [g0,g1,g2,g3,g4,g5]
    for x in line2:
        for i in range(6):
            gs[i].append(float(x.split(' ')[i+1]))
    file2.close()

    gs = [g0,g1,g2,g3,g4,g5]

    return r_values, energies, gs

r_values, energies, gs = vqe_parameters()

def import_raw_data(file_nums, month, file):
    zis = []; izs = []; zzs = []; xxs = []; yys = []
    path = os.getcwd()
    os.chdir('../../../..')
    print('path', path)
    os.chdir("oqclab-0.4_logs/" + month + '/' + file)
    for i, file_num in enumerate(file_nums):

        if i is 0:
            directory = "0000%s-H2_simulation_echo_2d_sweep_qst.__init__/000001-H2_simulation_echo_2d_sweep_qst.run" % file_num
        else:
            if file_num < 100:
                file_num_str_e1 = '0' + '%i' % file_num
                directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run" # /000001-H2_simulation_echo_2d_sweep_qst.run"
                # directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"
            elif file_num < 1000:
                file_num_str_e1 = '%i' % file_num
                directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run" #/000001-H2_simulation_echo_2d_sweep_qst.run"
                # directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"
            else:
                file_num_str_e1 = '%i' % file_num
                directory = "00" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.run" #/000001-H2_simulation_echo_2d_sweep_qst.run"
                # directory = "000" + file_num_str_e1 + "-H2_simulation_echo_2d_sweep_qst.__init__\\000001-H2_simulation_echo_2d_sweep_qst.run"

        os.chdir(directory)
        zis.append(np.load('zi.npy')[:, :5])
        izs.append(np.load('iz.npy')[:, :5])
        xxs.append(np.load('xx.npy')[:, :5])
        yys.append(np.load('yy.npy')[:, :5])
        zzs.append(np.load('zz.npy')[:, :5])

        if i is 0:
            os.chdir('../..')

        elif i is len(file_nums)-1:
            pass

        else:
            os.chdir('..')

    theta_range = np.load('theta_range', allow_pickle=True)
    theta_num = np.load('theta_num', allow_pickle=True)
    error_num = np.load('err_num', allow_pickle=True)
    os.chdir('..')

    N = len(file_nums) # 100 - (clean * 2)

    y = np.array([zis, izs, xxs, yys, zzs])
    y_ave = np.average(y, axis=1)
    y_std = np.std(y, axis=1) / np.sqrt(N)
    x = np.linspace(-theta_range,theta_range,theta_num)

    return y, y_ave, y_std, x, error_num

def gp_regression(x, y_ave, y_std, points):
    # Training set
    X = np.atleast_2d(x).T
    y = y_ave
    dy = y_std
    noise = np.random.normal(0, dy)
    y += noise
    # Test set
    x = np.atleast_2d(np.linspace(x[0], x[-1], points)).T
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=(dy / y) ** 2,
                                  n_restarts_optimizer=10)
    gp.fit(X, y)

    y_pred, sigma = gp.predict(x, return_std=True)
    if 0:
        fig = plt.figure()
        plt.errorbar(X, y, dy, fmt='r.', markersize=10, label=u'Observations')
        plt.plot(x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                 (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-1, 1)
        plt.legend(loc='upper left')

    return y_pred, sigma

def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def linear(x, a, b):
    return a * x + b

def full_gp(x, y_ave, y_std, zz):
    s = np.shape(y_ave)
    points = 1000
    s = (s[0], points, s[2])
    fit = np.zeros(s)
    fit_std = np.zeros(s)

    for i in range(s[0]):
        for j in range(s[2]):
            fit[i, :, j], fit_std[i, :, j] = gp_regression(x * 2 * zz, y_ave[i, :, j], y_std[i, :, j], points)

    x_fit = np.linspace(x[0], x[-1], 1000)

    return fit, fit_std, x_fit

def extrapolation(y, x, errors, zz, lin_plot=0, exp_plot=0, data_plot=0, show=0):
    points = 2

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    errors_ext = np.insert(errors,0,0)
    print(errors_ext)
    lin = []
    exp = []
    for i in range(len(y)):
        popt_l, pcov_l = curve_fit(linear, errors[0:points], y[i][0:points])
        lin.append(popt_l[1])
        if lin_plot:
            plt.plot(errors_ext, linear(errors_ext, popt_l[0], popt_l[1]))
        if exp_plot:
            popt_e, pcov_e = curve_fit(exponential, errors, y[i], p0=[popt_l[1], -popt_l[0] / popt_l[1], 0])  # ,bounds=([-2., -1., -1], [2., 1., 1.]))
            exp.append(popt_e[0] + popt_e[2])
            plt.plot(errors_ext, exponential(errors_ext, popt_e[0], popt_e[1], popt_e[2]))
        else:
            exp.append(popt_l[1])
        if data_plot:
            for j in range(5):
                plt.scatter(errors[j], y[i][j], color=colors[j])
        if lin_plot or exp_plot or data_plot:
            plt.show()

    if show:
        for j in range(len(y[0])):
            plt.plot(x * 2. * zz, y[:, j])
        plt.plot(x * 2. * zz, lin_extrapolated)
        if exp_plot:
            plt.plot(x * 2. * zz, exp_extrapolated)
        plt.xlabel(r'$Rotation\ angle\ /\ 2 \pi $')
        plt.ylabel('Eigenvalue')
        plt.show()

    return lin, exp

def error_mitigation(y, thetas, errors, zz_freq):
    s = np.shape(y)
    s = (s[0], s[1])
    lin_extrapolated = np.zeros(s)
    exp_extrapolated = np.zeros(s)
    for i in range(5):
        lin_extrapolated[i], exp_extrapolated[i] = extrapolation(y[i], thetas, errors, zz_freq, lin_plot=1,
                                                                 exp_plot=1, data_plot=1, show=0)

    return lin_extrapolated, exp_extrapolated

def energy(y_ave, x, errors, zz_freq):
    fig = plt.figure(0)
    extrap_lin, extrap_exp = error_mitigation(y_ave, x, errors, zz_freq)

    y_min = []
    y_argmin = []

    for j in range(2):
        min_datum = []
        argmin_datum = []
        mit_min = []
        mit_argmin = []
        optimal_theta = []
        for i in range(len(gs[0])):
            data_energy = gs[0][i] + gs[1][i] * y_ave[0, :, j] + gs[2][i] * y_ave[1, :, j] + gs[3][i] * y_ave[4, :, j] + gs[4][i] * y_ave[2, :, j] + gs[5][i] * y_ave[3, :, j]
            extrap_energy = gs[0][i] + gs[1][i] * np.array(extrap_lin[0]) + gs[2][i] * np.array(extrap_lin[1]) + gs[3][i] * np.array(extrap_lin[4]) + gs[4][i] * np.array(extrap_lin[2]) + gs[5][i] * np.array(extrap_lin[3])

            min_datum.append(np.amin(data_energy))
            argmin_datum.append(np.argmin(data_energy))

            mit_min.append(np.amin(extrap_energy))
            mit_argmin.append(np.argmin(extrap_energy))
            if j is 0:
                plt.plot(x * 2. * zz_freq, data_energy)

            # print("error:", 2 ** (j))
            # print("distance:", r_values[i])
            # print("optimal theta:", thetas[np.argmin(data_energy)] * 2. * zz_freq)
            # print("minimum value:", np.amin(data_energy))
            # print("linear extrapolated")
            # print("distance:", r_values[i])
            # print("optimal theta:", thetas[np.argmin(extrap_energy)] * 2. * zz_freq)
            # print("minimum value:", np.amin(extrap_energy))

            optimal_theta.append(x[np.argmin(extrap_energy)] * 2. * zz_freq)

        y_min.append(min_datum)
        y_argmin.append(argmin_datum)
    plt.ylabel(r'Total Energy (Hatree)', fontsize=20)
    plt.xlabel(r'$Rotation\ angle\ /\ 2 \pi $', fontsize=20)
    plt.show()
    return y_min, y_argmin, mit_min, mit_argmin, optimal_theta

def error_bars(y_argmin, y_std):

    y_eb = []
    for j in range(2):
        error_bars = []
        for i in range(len(gs[0])):
            min = y_argmin[j][i]
            error_bar = np.sqrt((gs[1][i] * y_std[0][min, j]) ** 2 + (gs[2][i] * y_std[1][min, j]) ** 2 + (gs[3][i] * y_std[4][min, j]) ** 2 + (gs[4][i] * y_std[2][min, j]) ** 2 + (gs[5][i] * y_std[3][min, j]) ** 2)
            error_bars.append(error_bar)

        y_eb.append(error_bars)

    mit_eb = []
    for i in range(len(gs[0])):
        lin_error_bar = np.sqrt((2 * np.array(y_eb[0])[i]) ** 2 + (np.array(y_eb[1])[i]) ** 2)
        mit_eb.append(lin_error_bar)

    return y_eb, mit_eb

def plot_paulis(x, x_fit, y_ave, y_std, fit, fit_std, zz_freq):
    fig = plt.figure(0)

    labels = [r'$\langle ZI \rangle$', r'$\langle IZ \rangle$', r'$\langle XX \rangle$', r'$\langle YY \rangle$', r'$\langle ZZ \rangle$']
    error_labels = ['Total duration = 1,700 ns',
                    'Total duration = 2,740 ns',
                    'Total duration = 4,820 ns',
                    'Total duration = 8,980 ns',
                    'Total duration = 17,300 ns']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(5):
        subplot = fig.add_subplot(5, 1, i+1)
        for j in range(np.shape(y_ave)[2]):
            subplot.errorbar(x * 2. * zz_freq, y_ave[i, :, j], yerr=y_std[i, :, j], color=colors[j], fmt='.')
            subplot.errorbar(x_fit * 2. * zz_freq, fit[i, :, j], yerr=fit_std[i, :, j], color=colors[j]) #, label=error_labels[j])
        # subplot.set(xlabel=r'$Rotation\ angle\ /\ 2 \pi $', ylabel=labels[i], fontsize=20)
        subplot.set_xlabel(r'$Rotation\ angle\ /\ 2 \pi $', fontsize=20)
        subplot.set_ylabel(labels[i], fontsize=20)
        subplot.set_ylim(-1.2, 1.2)
    # plt.legend()
    plt.show()

    # for i in range(5):
    #     for j in range(np.shape(pauli_ave)[2]):
    #         plt.errorbar(thetas,pauli_ave[i, :, j], yerr=pauli_std[i, :, j], fmt='o')
    #         plt.errorbar(thetas_gp, y_preds[i, :, j], yerr=sigmas[i, :, j])
    #     plt.xlabel('Rotation angle')
    #     plt.ylabel(labels[i])
    #     plt.ylim(-1.1, 1.1)
    #     plt.show()

def plot_energy_curve(y_min, y_eb, mit_min, mit_eb):

    fig = plt.figure(0)

    plt.plot(r_values, energies)  # Theory curve

    error_labels = ['Total duration = 1,700 ns', 'Total duration = 2,740 ns']
    for j in range(2):
        plt.errorbar(r_values, y_min[j], marker='o', markersize=1, linestyle='None', yerr=y_eb[j],
                     label=error_labels[j])

    plt.errorbar(r_values, mit_min, marker='o', markersize=1, linestyle='None', yerr=mit_eb,
                 label="Error mitigated result")
    plt.legend()
    plt.xlabel(r'Bond distance (Angstrom)', fontsize=20)
    plt.ylabel(r'Total Energy (Hatree)', fontsize=20)
    plt.show()



