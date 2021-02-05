# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from .operation import *
from .data import *
from .plot import *
from .theory import *

def hamiltonian_tomography_exp(directory, file_name, file_list, init_list, include_individual=False, save=False):
    # print('0')
    label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    if init_list is None:
        init = ['-i, 1', '-i, 0', '-i, -i', '-i, +i', '-i, +', '-i, -', '1, -i', '0, -i', '-i, -i', '+i, -i', '+, -i', '-, -i']
    else:
        init = init_list

    joint = load_oqclab(directory, file_name + 'e_joint.npy', file_list)
    indiv = load_oqclab(directory, file_name + 'e_indiv.npy', file_list)
    theta = load_oqclab(directory, file_name + 'theta.npy', file_list)

    dtheta = theta[0][1]-theta[0][0]

    x = []
    y = []
    z = []
    for i in range(len(file_list)):
        print('Initial state:', init[i])

        ### Data cooking

        if include_individual:
            all_data = [indiv[i], joint[i]]
            readout_label = ['Before correction', 'After correction']
        else:
            all_data = [joint[i]]
            readout_label = ['Joint readout']

        fit_dim_1 = []
        for j in range(4):
            fit_dim_2 = []
            for k in range(4):
                blochs = []
                fit_blochs = []
                fit_dim_3 = []
                for l, result in enumerate(all_data): # enumerate([indiv[i], joint[i]])
                    den_mat = 0.25 * np.tensordot(pauli(2), result, (0, 0))
                    den_mat_4_4 = transform_to_4_4(den_mat)
                    blochs.append(bloch_vector(den_mat_4_4))
                    fit_blochs.append(lsq_fit(exp_func, theta[i], bloch_vector(den_mat_4_4)))
                    fit_dim_3.append(Fit_1D_Freq_Gaussian_Estimation(den_mat_4_4[j,k], dtheta, gaussian_window_std=0.2))
                fit_dim_2.append(fit_dim_3)
            fit_dim_1.append(fit_dim_2)

        ### Results

        bloch = concatenate_newdim(blochs)
        fit_bloch = concatenate_newdim(fit_blochs)
        fit_ham = fit_dim_1 #add_dim(fit_dim_1, 3)
        tomo = concatenate_newdim(all_data)

        if save is False:
            save_xyz_1 = False
            save_xyz_2 = False
            save_sphere_1 = False
            save_sphere_2 = False
            save_4_4 = False
            save_blochvector = False
        else:
            save_xyz_1 = save + '_xyz_1_{}'.format(i)
            save_xyz_2 = save + '_xyz_2_{}'.format(i)
            save_sphere_1 = save + '_blochsphere_1_{}'.format(i)
            save_sphere_2 = save + '_blochsphere_2_{}'.format(i)
            save_4_4 = save + '_4_4_{}'.format(i) + save
            save_blochvector = save + '_blochvector_{}'.format(i)

        # 1st qubit
        data_y = transform_to_3_1(tomo, [4,8,12])
        fit_y = [[Fit_1D_Freq_Gaussian_Estimation(data[:,k], dtheta, gaussian_window_std=0.2) for k in range(len(all_data))] for data in data_y]
        plot_3_1(theta[i], data_y, fit_y=fit_y, y_lim=[-1.4, 1.4], label=readout_label, x_label=r'Time ($\mu s$)', y_label='Population', save=save_xyz_1)
        plot_sphere(theta[i], data_y, fit_y=None, label=readout_label, save=save_sphere_1)

        x.append(fit_y[0][-1]['Frequency'])
        y.append(fit_y[1][-1]['Frequency'])
        z.append(fit_y[2][-1]['Frequency'])

        # data = [data_y[0,:,1], data_y[1,:,1], data_y[2,:,1]]
        # fit = Fit_3D_Rabi(data, dtheta, freq_est=None, use_freq_bound=True)
        # Plot_3D_Rabi(data, fit['Time'], fit)


        # 2nd qubit
        data_y = transform_to_3_1(tomo, [1,2,3])
        fit_y = [[Fit_1D_Freq_Gaussian_Estimation(data[:,k], dtheta, gaussian_window_std=0.2) for k in range(len(all_data))] for data in data_y]
        plot_3_1(theta[i], data_y, fit_y=fit_y, y_lim=[-1.4, 1.4], label=readout_label, x_label=r'Time ($\mu s$)', y_label='Population', save=save_xyz_2)
        plot_sphere(theta[i], data_y, fit_y=None, label=readout_label, save=save_sphere_2)


        # data = [data_y[0,:,1], data_y[1,:,1], data_y[2,:,1]]
        # fit = Fit_3D_Rabi(data, dtheta, freq_est=None, use_freq_bound=True)
        # Plot_3D_Rabi(data, fit['Time'], fit)

        # State tomography
        t1 = time.time()
        plot_4_4(theta[i], tomo, fit_y=None, label=readout_label, x_label=r'Time ($\mu s$)', y_label=label, save=save_4_4)
        t2 = time.time()
        print(t2 - t1)

        # Bloch vector
        plot_1(theta[i], bloch, fit_y=None, y_lim=[0, 2.0], label=readout_label, x_label=r'Time ($\mu s$)', y_label='Bloch vector length', save=save_blochvector)

    return theta, tomo, bloch

def simultaneous_ramsey_exp(directory, file_name, file_list, include_individual=False, save=False):
    # print('0')
    label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

    joint = load_oqclab(directory, file_name + 'e_joint.npy', file_list)
    indiv = load_oqclab(directory, file_name + 'e_indiv.npy', file_list)
    theta = load_oqclab(directory, file_name + 'theta.npy', file_list)

    dtheta = theta[0][1]-theta[0][0]

    x = []
    y = []
    z = []
    for i in range(len(file_list)):

        ### Data cooking

        if include_individual:
            all_data = [indiv[i], joint[i]]
            readout_label = ['Before correction', 'After correction']
        else:
            all_data = [joint[i]]
            readout_label = ['Joint readout']

        fit_dim_1 = []
        for j in range(4):
            fit_dim_2 = []
            for k in range(4):
                blochs = []
                fit_blochs = []
                fit_dim_3 = []
                for l, result in enumerate(all_data): # enumerate([indiv[i], joint[i]])
                    den_mat = 0.25 * np.tensordot(pauli(2), result, (0, 0))
                    den_mat_4_4 = transform_to_4_4(den_mat)
                    blochs.append(bloch_vector(den_mat_4_4))
                    fit_blochs.append(lsq_fit(exp_func, theta[i], bloch_vector(den_mat_4_4)))
                    fit_dim_3.append(Fit_1D_Freq_Gaussian_Estimation(den_mat_4_4[j,k], dtheta, gaussian_window_std=0.2))
                fit_dim_2.append(fit_dim_3)
            fit_dim_1.append(fit_dim_2)

        ### Results

        bloch = concatenate_newdim(blochs)
        fit_bloch = concatenate_newdim(fit_blochs)
        fit_ham = fit_dim_1 #add_dim(fit_dim_1, 3)
        tomo = concatenate_newdim(all_data)

        if save is False:
            save_xyz_1 = False
            save_xyz_2 = False
            save_sphere_1 = False
            save_sphere_2 = False
            save_4_4 = False
            save_blochvector = False
        else:
            save_xyz_1 = save + '_xyz_1_{}'.format(i)
            save_xyz_2 = save + '_xyz_2_{}'.format(i)
            save_sphere_1 = save + '_blochsphere_1_{}'.format(i)
            save_sphere_2 = save + '_blochsphere_2_{}'.format(i)
            save_4_4 = save + '_4_4_{}'.format(i) + save
            save_blochvector = save + '_blochvector_{}'.format(i)

        # 1st qubit
        data_y = transform_to_3_1(tomo, [4,8,12])
        fit_y = [[Fit_1D_Freq_Gaussian_Estimation(data[:,k], dtheta, gaussian_window_std=0.2) for k in range(len(all_data))] for data in data_y]
        plot_3_1(theta[i], data_y, fit_y=fit_y, y_lim=[-1.4, 1.4], label=readout_label, x_label=r'Time ($\mu s$)', y_label='Population', save=save_xyz_1)
        plot_sphere(theta[i], data_y, fit_y=None, label=readout_label, save=save_sphere_1)

        x.append(fit_y[0][-1]['Frequency'])
        y.append(fit_y[1][-1]['Frequency'])
        z.append(fit_y[2][-1]['Frequency'])

        # data = [data_y[0,:,1], data_y[1,:,1], data_y[2,:,1]]
        # fit = Fit_3D_Rabi(data, dtheta, freq_est=None, use_freq_bound=True)
        # Plot_3D_Rabi(data, fit['Time'], fit)


        # 2nd qubit
        data_y = transform_to_3_1(tomo, [1,2,3])
        fit_y = [[Fit_1D_Freq_Gaussian_Estimation(data[:,k], dtheta, gaussian_window_std=0.2) for k in range(len(all_data))] for data in data_y]
        plot_3_1(theta[i], data_y, fit_y=fit_y, y_lim=[-1.4, 1.4], label=readout_label, x_label=r'Time ($\mu s$)', y_label='Population', save=save_xyz_2)
        plot_sphere(theta[i], data_y, fit_y=None, label=readout_label, save=save_sphere_2)


        # data = [data_y[0,:,1], data_y[1,:,1], data_y[2,:,1]]
        # fit = Fit_3D_Rabi(data, dtheta, freq_est=None, use_freq_bound=True)
        # Plot_3D_Rabi(data, fit['Time'], fit)

        # State tomography
        plot_4_4(theta[i], tomo, fit_y=None, label=readout_label, x_label=r'Time ($\mu s$)', y_label=label, save=save_4_4)

        # Bloch vector
        plot_1(theta[i], bloch, fit_y=None, y_lim=[0, 2.0], label=readout_label, x_label=r'Time ($\mu s$)', y_label='Bloch vector length', save=save_blochvector)

    return

def vqe_exp(directory, file_name, file_list, init_list, offset=None, tomo_modi=False, plot_qst=False, plot_bloch=False, plot_energy=False, plot_error_mitigation=False, plot_everything=False):
    # print('1')
    label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    vqe_pauli_label = ['ZI', 'IZ', 'ZZ', 'XX', 'YY']
    init = ['-i, 1', '-i, 0', '-i, -i', '-i, +i', '-i, +', '-i, -', '1, -i', '0, -i', '-i, -i', '+i, -i', '+, -i', '-, -i']

    if init_list=='original':
        joint = load_oqclab(directory, file_name + 'e.npy', file_list)
        theta_range = 0.52 # load_oqclab(directory, file_name + 'theta_range', file_list)
        theta_num = 31 # load_oqclab(directory, file_name + 'theta_num', file_list)
        theta = [np.linspace(-theta_range, theta_range, theta_num)]*len(init_list)

    else:
        joint = load_oqclab(directory, file_name + 'e_joint.npy', file_list)
        indiv = load_oqclab(directory, file_name + 'e_indiv.npy', file_list)
        theta = load_oqclab(directory, file_name + 'theta.npy', file_list)

    tomos = []
    pauli_4_4 = []
    pauli_vqe = []
    for i in range(len(file_list)):
        print(init_list)
        ### Data cooking
        if init_list=='+i, -':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
            else:
                #sign = [1, -1, -1, -1, 1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
                sign = [-1, 1, -1, -1, 1]; ind = [[2,0], [0,1], [2,1], [1,3], [3,2]]
        elif init_list=='-i, +':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
            else:
                #sign = [-1, 1, -1, 1, -1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
                sign = [1, -1, -1, 1, -1]; ind = [[2,0], [0,1], [2,1], [1,3], [3,2]]
        elif init_list=='-, +i':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
            else:
                sign = [-1, 1, -1, -1, 1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
        elif init_list=='+, -i':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
            else:
                sign = [1, -1, -1, 1, -1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
        elif init_list=='+, +i':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
            else:
                sign = [1, 1, 1, -1, -1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
        elif init_list=='-, -i':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
            else:
                sign = [-1, -1, 1, 1, 1]; ind = [[1,0], [0,2], [1,2], [3,1], [2,3]]
        elif init_list=='+i, +':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
            else:
                sign = [1, 1, 1, 1, 1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
        elif init_list=='-i, -':
            if tomo_modi:
                sign = [1, -1, -1, -1, 1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
            else:
                sign = [-1, -1, 1, -1, -1]; ind = [[2,0], [0,1], [2,1], [3,2], [1,3]]
        elif init_list=='original':
            if tomo_modi:
                sign = [1, 1, 1, 1, 1]; ind = [[3,0], [0,3], [3,3], [1,1], [2,2]]
            else:
                sign = [1, 1, 1, 1, 1]; ind = [[3,0], [0,3], [3,3], [1,1], [2,2]]
        else:
            print('oops')


        tomo = np.flip(joint[i], axis=1)
        tomos.append(tomo)
        tomo_4_4 = transform_to_4_4(tomo)
        pauli_4_4.append(tomo_4_4)

        tomo_vqe = np.array([sign[0]*tomo_4_4[ind[0][0],ind[0][1]],
                             sign[1]*tomo_4_4[ind[1][0],ind[1][1]],
                             sign[2]*tomo_4_4[ind[2][0],ind[2][1]],
                             sign[3]*tomo_4_4[ind[3][0],ind[3][1]],
                             sign[4]*tomo_4_4[ind[4][0],ind[4][1]]])

        pauli_vqe.append(tomo_vqe)

        den_mat = 0.25 * np.tensordot(pauli(2), joint[i], (0, 0))
        den_mat_4_4 = transform_to_4_4(den_mat)
        num_plots = len(den_mat_4_4[0][0][0])
        fit_dim_1 = []
        for j in range(4):
            fit_dim_2 = []
            for k in range(4):
                fit_dim_3 = []
                for l in range(num_plots):
                    # print(Fit_1D_Freq_Gaussian_Estimation(den_mat_4_4[j,k,:,l], theta[0][1]-theta[0][0], gaussian_window_std=0.2))
                    fit_dim_3.append(Fit_1D_Freq_Gaussian_Estimation(den_mat_4_4[j,k,:,l], theta[0][1]-theta[0][0], gaussian_window_std=0.2))
                fit_dim_2.append(fit_dim_3)
            fit_dim_1.append(fit_dim_2)

        fit = fit_dim_1
        bloch = bloch_vector(den_mat_4_4)


        label_name = []
        for j in range(len(tomo[0][0])):
            label_name.append('Error {}'.format(j+1))
        label_name.append('Error mitigated')
        ### Results


        ### Display results

        print('Initial state:', init_list[0])

        # State tomography
        print('Data:', i+1)
        if plot_everything:
            if plot_qst:
                print('Data:', i+1, 'QST')
                plot_4_4(theta[i], tomo, fit_y=None, title=label, label=label_name)
            if plot_energy:
                print('Data:', i+1, 'Energy curve')
                plot_energy_curve(theta[i], tomo_vqe, label=label_name)

            # # Bloch vector
            if plot_bloch:
                plot_1(theta[i], bloch, fit_y=None, y_lim=[0, 2.0], label=label_name)

                # 1st qubit
                data_y = transform_to_3_1(tomo, [1,2,3])
                plot_sphere(theta[i], data_y, fit_y=None, label=label_name)

                # 2nd qubit
                data_y = transform_to_3_1(tomo, [4,8,12])
                plot_sphere(theta[i], data_y, fit_y=None, label=label_name)

    tomos = concatenate_newdim(tomos)
    pauli_4_4s = concatenate_newdim(pauli_4_4)
    pauli_vqes = concatenate_newdim(pauli_vqe)


    tomo_ave = np.average(tomos, axis=3)
    tomo_4_4_ave = np.average(pauli_4_4s, axis=4)
    pauli_vqe_ave = np.average(pauli_vqes, axis=3)

    # Plot averaged result
    if plot_qst:
        print('Averaged QST')
        plot_4_4(theta[0], tomo_4_4_ave, fit_y=None, title=label, label=label_name)
    if plot_energy:
        print('Averaged Energy curve')
        plot_energy_curve(theta[0], pauli_vqe_ave, label=label_name)
    if plot_bloch:
        # 1st qubit
        data_y = transform_to_3_1(tomo_ave, [1,2,3])
        plot_sphere(theta[0], data_y, fit_y=None, label=label_name)

        # 2nd qubit
        data_y = transform_to_3_1(tomo_ave, [4,8,12])
        plot_sphere(theta[0], data_y, fit_y=None, label=label_name)


    # Fit error mitigation results
    if plot_error_mitigation is not None:
        x = plot_error_mitigation[0]
        fit_x = np.linspace(0, 5.0, 101)

        s = np.shape(pauli_vqe_ave)
        intercept_dim1 = []
        for i in range(s[0]):
            intercept_dim2 = []
            for j in range(s[1]):
                if plot_error_mitigation[1] is 'linear':
                    result = lsq_fit(lin_func, x, pauli_vqe_ave[i,j])
                    intercept_dim2.append(result['popt'][1])
                    fit_y = result
                    #fit_y = lin_func(fit_x, result['popt'][0], result['popt'][1])

                elif plot_error_mitigation[1] is 'exponential':
                    result = lsq_fit(exp_func, x, pauli_vqe_ave[i,j])
                    intercept_dim2.append(result['popt'][0] + result['popt'][2])
                    # fit_y = exp_func(fit_x, result['popt'][0], result['popt'][1], result['popt'][2])
                    fit_y = result

                if plot_everything:
                    y = add_dim(pauli_vqe_ave[i,j], 2)
                    # plot_1(x, y, [fit_y], x_lim=[0, 5], y_lim=[-1.1, 1.1], label=[vqe_pauli_label[i]])

            intercept_dim1.append(intercept_dim2)
        error_mitigated_result = add_dim(np.array(intercept_dim1), 3)

        full_result = np.concatenate([pauli_vqe_ave, error_mitigated_result], 2)
        print(np.shape(full_result))

        print('Error mitigated result')
        plot_energy_curve(theta[0], error_mitigated_result, label=label_name)

        print('Full result')
        plot_energy_curve(theta[0], full_result, label=label_name)
        for i in range(5):
            plot_1(theta[0], full_result[i], fit_y=None, y_lim=[-1.1, 1.1], label=label_name)

    return tomo_4_4_ave, pauli_vqe_ave, theta

def vqe_exp_multi(directories, file_name, file_list, init_list, offset=None, tomo_modi=False, plot_qst=False, plot_bloch=False, plot_energy=False, plot_error_mitigation=False, plot_everything=False):
    # print('1')
    label = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    vqe_pauli_label = ['ZI', 'IZ', 'ZZ', 'XX', 'YY']
    init = ['-i, 1', '-i, 0', '-i, -i', '-i, +i', '-i, +', '-i, -', '1, -i', '0, -i', '-i, -i', '+i, -i', '+, -i', '-, -i']

    tomo_4_4s = []
    pauli_vqes = []
    for i, directory in enumerate(directories):
        tomo_4_4, pauli_vqe, theta = vqe_exp(directory, file_name, file_list, init_list[i], tomo_modi=tomo_modi, plot_qst=plot_qst, plot_bloch=False, plot_energy=plot_energy, plot_error_mitigation=plot_error_mitigation, plot_everything=plot_everything)
        tomo_4_4s.append(tomo_4_4)
        pauli_vqes.append(pauli_vqe)

    label_name = []
    for j in range(len(tomo_4_4s[0][0][0][0])):
        label_name.append('Error {}'.format(j+1))
    label_name.append('Error mitigated')
    ### Results

    pauli_vqe = concatenate_newdim(pauli_vqes)
    pauli_vqe_ave = np.average(pauli_vqe, 3)
    tomo_4_4 = concatenate_newdim(tomo_4_4s)
    tomo_4_4_ave = np.average(tomo_4_4, 4)


    # Plot averaged result
    if plot_qst:
        print('Averaged QST')
        plot_4_4(theta[0], tomo_4_4_ave, fit_y=None, title=label, label=label_name)
    if plot_energy:
        print('Averaged Energy curve')
        plot_energy_curve(theta[0], pauli_vqe_ave, label=label_name)

    # Fit error mitigation results
    if plot_error_mitigation is not None:
        x = plot_error_mitigation[0]
        fit_x = np.linspace(0, 5.0, 101)

        s = np.shape(pauli_vqe_ave)
        intercept_dim1 = []
        for i in range(s[0]):
            intercept_dim2 = []
            for j in range(s[1]):
                if plot_error_mitigation[1] is 'linear':
                    result = lsq_fit(lin_func, x, pauli_vqe_ave[i,j])
                    intercept_dim2.append(result['popt'][1])
                    fit_y = result
                    #fit_y = lin_func(fit_x, result['popt'][0], result['popt'][1])

                elif plot_error_mitigation[1] is 'exponential':
                    result = lsq_fit(exp_func, x, pauli_vqe_ave[i,j])
                    intercept_dim2.append(result['popt'][0] + result['popt'][2])
                    # fit_y = exp_func(fit_x, result['popt'][0], result['popt'][1], result['popt'][2])
                    fit_y = result

                if plot_everything:
                    y = add_dim(pauli_vqe_ave[i,j], 2)
                    plot_1(x, y, [fit_y], x_lim=[0, 5], y_lim=[-1.1, 1.1], label=[vqe_pauli_label[i]])

            intercept_dim1.append(intercept_dim2)
        error_mitigated_result = add_dim(np.array(intercept_dim1), 3)

        full_result = np.concatenate([pauli_vqe_ave, error_mitigated_result], 2)
        print(np.shape(full_result))

        print('Error mitigated result')
        plot_energy_curve(theta[0], error_mitigated_result, label=label_name)

        print('Full result')
        plot_energy_curve(theta[0], full_result, label=label_name)
        for i in range(5):
            plot_1(theta[0], full_result[i], fit_y=None, y_lim=[-1.1, 1.1], label=label_name)

    return tomo_4_4_ave, pauli_vqe_ave, theta

