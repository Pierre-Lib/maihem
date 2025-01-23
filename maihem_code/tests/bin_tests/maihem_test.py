import unittest


from maihem_code.main.bin.maihem import model_training_and_validation
from maihem_code.main.bin.maihem import detections_and_calculations

class MyTestCase(unittest.TestCase):
    def test_training_and_validation(self):
        test_bin_metrics = model_training_and_validation(dataset_path = 'coco8-seg.yaml',
                                                         epochs = 20,
                                                         seed = 1612)
        expected_mAP50_box = 0.9396968794610389
        expected_mAP75_seg = 0.5392530224525044
        self.assertEqual(test_bin_metrics['map50_box'], expected_mAP50_box)
        self.assertEqual(test_bin_metrics['map75_seg'], expected_mAP75_seg)

    def test_detections_and_calculations(self):
        test_merged_detection_metrics = detections_and_calculations(model_path = 'yolo11n-seg.pt',
                                                                    yaml_file_path = '../coco8-seg_test.yaml',
                                                                    image_path = '../test_img.jpg',
                                                                    lesion_names = ['person', 'elephant'])
        expected_output = {'person':
                               {'test_img.jpg': {'count': 1, 'total area': 4437.090087890625}},
                           'elephant':
                               {'test_img.jpg': {'count': 2, 'total area': 21790.920776367188}}}
        self.assertEqual(test_merged_detection_metrics, expected_output)


if __name__ == '__main__':
    unittest.main()
