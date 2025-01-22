import unittest
from maihem_code.main.in_out.output import merge_detection_metrics

class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_count_dict = {"elephant" : 3,
                          "tortoise" : 5,
                          "otter": 7}
        test_area_dict = {"elephant": 12345,
                          "tortoise": 5678,
                          "otter": 9876}
        test_merged_dict = merge_detection_metrics(test_count_dict, test_area_dict)
        expected_merged_dict = {'elephant' : {'count': 3, 'total area': 12345},
                                'tortoise': {'count': 5, 'total area': 5678},
                                'otter': {'count': 7, 'total area': 9876}}
        self.assertEqual(test_merged_dict, expected_merged_dict)  # add assertion here


if __name__ == '__main__':
    unittest.main()
