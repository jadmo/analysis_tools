import unittest
from analysis_tools.data.array import Array

import numpy as np

class TDDforArray(unittest.TestCase):

    def setUp(self):
        self.test_input = np.zeros((16, 4, 4))
        self.array = Array(self.test_input)

    def test_shape_attribute_correct_result(self):
        result = self.array.shape
        print('processed:', result)
        print('original:', np.shape(input))
        self.assertEqual(result, (16, 4, 4))

    def test_if_output_is_np_ndarray(self):
        result = type(self.array.output)
        print('output type:', type(self.array.output))
        self.assertIs(result, np.ndarray)

    def test_to_4_4_method_correct_result(self):
        self.array.to_4_4()
        result = np.shape(self.array.output)

        self.assertEqual(result, (4, 4, 4, 4))

    def test_to_4_4_method_returns_error_message_if_input_type_is_not_correct(self):
        self.assertRaises(TypeError, self.array.to_4_4, 'd')

if __name__ == "__main__":
    unittest.main()