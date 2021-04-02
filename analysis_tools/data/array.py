import numpy as np


class Array():

    def __init__(self, input):
        self.shape = np.shape(input)
        self.output = input

    def to_4_4(self):

        s = np.shape(self.output)

        try:
            if s[0] == 16:
                s = (4, 4) + s[1:]
                reshaped_data = np.reshape(self.output, s)

            elif s[0] == 4 and s[1] == 4:
                reshaped_data = data

        except:
            print('The input data is not suitable for transforming into 4 by 4 for first two dimenstions.')

        self.output = reshaped_data

    # def shape(self, input):
    #     return np.shape(input)

if __name__ == "__main__":
    input = 'd' # np.zeros((4,4,4,4))
    array = Array(input)
    array.to_4_4