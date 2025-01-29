from maihem_code.main.bin.maihem import model_training_and_validation, detections_and_calculations
from pathlib import Path

current_dir = Path(__file__).parent
print(current_dir)


print(f'Beginning example model training. The model will be saved at {current_dir}/maihem_example')
example_metrics = model_training_and_validation(dataset_path = current_dir / 'coco8-seg.yaml',
                                                epochs = 1,
                                                model_path = current_dir,
                                                model_name = 'maihem_example')

print('\nCommencing example model usage on images from the example_pictures folder.')

example_calculations = detections_and_calculations(model_path = current_dir / 'maihem_example/weights/best.pt',
                                                   yaml_file_path = current_dir / 'coco8-seg.yaml',
                                                   image_path = current_dir / 'example_pictures',
                                                   output_path = current_dir,
                                                   lesion_names = ['dog', 'cat'])

print(f"Images with overlayed detections are saved in {current_dir}/predict.\n"
      f"Detections data are saved in {current_dir}/predict/detections.json")
print(f"The number of dogs and cats, and the total area for each in the test images are as follows:\n")
print(f"Dogs: {example_calculations['dog']}\nCats: {example_calculations['cat']}")
print(f"These measures are also saved in {current_dir}/predict/measures.json")


