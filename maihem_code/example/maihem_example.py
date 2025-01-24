from maihem_code.main.bin.maihem import model_training_and_validation, detections_and_calculations

print(f'Beginning example model training. The model will be saved at maihem_code/example/maihem_example')
example_metrics = model_training_and_validation(dataset_path = 'coco8-seg.yaml',
                                                epochs = 20,
                                                model_path = 'maihem_code/example',
                                                model_name = 'maihem_example')

print('\nCommencing example model usage on images from the example_pictures folder.')

example_calculations = detections_and_calculations(model_path = 'maihem_code/example/maihem_example/weights/best.pt',
                                                   yaml_file_path = 'maihem_code/example/coco8-seg.yaml',
                                                   image_path = 'maihem_code/example/example_pictures',
                                                   output_path = 'maihem_code/example',
                                                   lesion_names = ['dog', 'cat'])

print(f'Images with overlayed detections are saved in maihem_code/example/predict.\n'
      f'Detections data are saved in maihem_code/example/predict/detections.json')
print(f'The number of dogs and cats, and the total area for each in the test images are as follows:\n')
print(f'Dogs: {example_calculations['dog']}\nCats: {example_calculations['cat']}')
print(f'These measures are also saved in maihem_code/example/predict/measures.json')


