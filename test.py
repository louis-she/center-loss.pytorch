import unittest

import torch
import numpy

from loss import get_center_delta

class TestCenterLossFunctions(unittest.TestCase):

    def test_get_center_delta(self):
        features = torch.tensor( (( 1,2,3), (4,5,6), (7,8,9)) ).float()
        centers = torch.tensor( ((1,1,1), (2,2,2), (3,3,3), (5,5,5) )).float()
        targets = torch.tensor((1, 3, 1))
        result = get_center_delta(features, centers, targets)

        # size should match
        self.assertTrue(result.size() == centers.size())

        # for class 1
        result_for_class_one = ((features[0] + features[2]) - 2 * centers[1]) / 3
        self.assertEqual(3, torch.sum(result[1] == result_for_class_one).item())

        # for class 3
        result_for_class_three = (features[1] - centers[3]) / 2
        self.assertEqual(3, torch.sum(result[3] == result_for_class_three).item())

        # others should all be zero
        sum_others = torch.sum(result[(0,2), :]).item()
        self.assertEqual(0, sum_others)


if __name__ == '__main__':
    unittest.main()