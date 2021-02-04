# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import re
import time
from scipy import optimize
from scipy.stats import norm

def file_num(x):

    num = "{x:.0f}"

    if x < 10:
        num = "00{x:.0f}"
    elif x < 100:
        num = "0{x:.0f}"

    return num.format(x=x)

def load_data(file):

    data = np.load(file)

    return data

def load_oqclab(directory, file_name, file_list):

    batch = []
    for i in file_list:
        file = directory + '/000' + file_num(i) + file_name
        batch.append(load_data(file))

    return batch

def transform_to_4_4(data):
    # This function assumes that the first dimension of the input data is 16 or 4*4.
    # The outputs of the data becomes 4*4 in the first two axes

    s = np.shape(data)

    if s[0] == 16:
        s = (4,4) + s[1:]
        reshaped_data = np.reshape(data, s)

    elif s[0] == 4:
        reshaped_data = data

    return reshaped_data

def transform_to_3_1(data, indices):

    reshaped_data = np.array([data[indices[0]], data[indices[1]], data[indices[2]]])

    return reshaped_data

def add_dim(data, target_dim):

    s = np.shape(data)
    # print(s)
    if len(s) == target_dim:
        reshaped_data = data

    elif len(s) == target_dim-1:
        s = s + (1, )
        reshaped_data = np.reshape(data, s)

    return reshaped_data

def swap_axes(data):
    # Swap last two axes in the data
    # For example, assume 4*4*d1*d2 as an input, it outputs 4*4*d2*d1

    s = np.shape(data)
    swapped_data = np.swapaxes(data, s[-2], s[-1])

    return swapped_data

def concatenate_newdim(data):

    reshaped_data = []
    for d in data:
        reshaped_data.append(d)

    len_shape = len(np.shape(reshaped_data))
    s_tup = tuple(range(1, len_shape, 1)) + (0,)

    reshaped_data = np.transpose(reshaped_data, s_tup)

    return np.array(reshaped_data)