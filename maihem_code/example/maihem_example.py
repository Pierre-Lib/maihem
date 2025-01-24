import os
from maihem_code.main.bin.maihem import model_training_and_validation, detections_and_calculations

cwd = os.getcwd()
print(f'Beginning example model training. The model will be saved at {cwd}/maihem_example')
example_metrics = model_training_and_validation(dataset_path = 'coco8-seg.yaml',
                                                epochs = 20,
                                                model_name = 'maihem_example')

print(f'Commencing example model usage on images from the example_pictures folder.\n'
      f'')

example_calculations = detections_and_calculations(model_path = './maihem_example/weights/best.pt',
                                                   yaml_file_path = 'coco8-seg.yaml',
                                                   image_path = './example_pictures',
                                                   lesion_names = ['dog', 'cat'])

print(f'The number of cats and dogs, and the total area for each in the test images are as follows:\n')
print(example_calculations)

