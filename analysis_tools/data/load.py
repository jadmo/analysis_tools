import numpy as np
from qutip import *
import time

def file_num_oqclab(x):

    num = "{x:.0f}"

    if x < 10:
        num = "00000{x:.0f}"
    elif x < 100:
        num = "0000{x:.0f}"
    elif x < 1000:
        num = "000{x:.0f}"
    elif x < 1000:
        num = "00{x:.0f}"
    elif x < 1000:
        num = "0{x:.0f}"
    elif x < 1000:
        num = "{x:.0f}"

    return num.format(x=x)

def load_oqclab(directory, file_name, file_list):

    batch = []
    for i in file_list:
        file = directory + '/000' + file_num_oqclab(i) + file_name
        batch.append(np.load(file))

    return batch