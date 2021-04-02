import numpy as np
from qutip import *
import time

def to_4_4(data):
    # This function assumes that the first dimension of the input data is 16 or 4*4.
    # The outputs of the data becomes 4*4 in the first two axes

    s = np.shape(data)

    if s[0] == 16:
        s = (4,4) + s[1:]
        reshaped_data = np.reshape(data, s)

    elif s[0] == 4:
        reshaped_data = data

    return reshaped_data

def to_n_1(data, indices):

    return np.array([data[i] for i in indices])

def swap_axes(data):
    # Swap last two axes in the data
    # For example, assume 4*4*d1*d2 as an input, it outputs 4*4*d2*d1

    s = np.shape(data)
    swapped_data = np.swapaxes(data, s[-2], s[-1])

    return swapped_data


"""
These files can be replaced by np.stack([data], -1) and np.stack([data1, data2], -1)

def add_dim(data, target_dim):

    s = np.shape(data)
    # print(s)
    if len(s) == target_dim:
        reshaped_data = data

    elif len(s) == target_dim-1:
        s = s + (1, )
        reshaped_data = np.reshape(data, s)

    return reshaped_data

def concatenate_newdim(data):

    reshaped_data = []
    for d in data:
        reshaped_data.append(d)

    len_shape = len(np.shape(reshaped_data))
    s_tup = tuple(range(1, len_shape, 1)) + (0,)

    reshaped_data = np.transpose(reshaped_data, s_tup)

    return np.array(reshaped_data)
"""
