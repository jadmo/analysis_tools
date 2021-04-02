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

def oqclab_files(date, experiment, file_num):

    batch = []
    for i in file_num:
        file = 'Data/oqclab-0.4_logs/' + date + file_num_oqclab(i) + '-' + experiment
        batch.append(np.load(file, allow_pickle=True))

    if len(batch)==1:
        batch = batch[0]

    return batch
