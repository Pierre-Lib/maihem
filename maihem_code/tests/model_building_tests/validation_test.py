import unittest
from maihem_code.main.model_building.validation import validate_model

class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_validation = validate_model(model_path= 'test_model/weights/best.pt',
                                         save_path = 'test_model/Validation',
                                         save_name = 'test_validation')
        test_mAP75_box = test_validation['map75_box']
        test_mAP75_seg = test_validation['map75_seg']
        expected_mAP75_box = 0.7050959860383944
        expected_mAP75_seg = 0.5353737638161722
        self.assertEqual(test_mAP75_box, expected_mAP75_box)
        self.assertEqual(test_mAP75_seg, expected_mAP75_seg)


if __name__ == '__main__':
    unittest.main()
