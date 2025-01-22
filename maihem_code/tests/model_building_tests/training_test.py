import unittest
from maihem_code.main.model_building.training import train_model

class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_hyperparameters = d = {'epochs': 20,
                                    'batch_size': 16,
                                    'img_size': 640,
                                    'device': 'cpu',
                                    'seed': 16}
        test_model_metrics = train_model('coco8-seg.yaml',
                                         test_hyperparameters,
                                         '.',
                                         'test_model',
                                         'yolo11n-seg')
        test_model_seg_mAP50 = test_model_metrics.seg.map50
        expected_mAP50 = 0.8496081399536289
        self.assertEqual(test_model_seg_mAP50, expected_mAP50)  # add assertion here


if __name__ == '__main__':
    unittest.main()
