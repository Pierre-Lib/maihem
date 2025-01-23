import unittest

from maihem_code.main.model_usage.detections import detection_segmentation

class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_detection = detection_segmentation(model_path = '../model_building_tests/test_model/weights/best.pt',
                                                image_path = 'test_img.jpg',
                                                save_path = './test_predictions',
                                                conf_threshold = 0.25)
        detection = test_detection[0]
        box = detection.boxes[0]
        class_id = int(box.cls)
        bbox = box.xyxy.cpu().numpy()[0]
        bbox_0 = bbox[0]
        test_dict = {'class' : class_id,
                     'box_0' : bbox_0}
        expected_dict = {'class' : 20,
                         'box_0' : 233.49911499023438}

        self.assertEqual(test_dict['class'], expected_dict['class'])
        self.assertEqual(test_dict['box_0'], expected_dict['box_0'])


if __name__ == '__main__':
    unittest.main()
