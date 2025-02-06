import os
from PIL import Image
from shapely.geometry import Polygon
import numpy as np
import json 
import re 
# Assuming x is your JSON data
# Define the folder where cropped images will be saved
root = '/home/amur/Amur/vrdOCR/datasets/TextCaps/train_val_images/train_images'
output_dir = '/home/amur/Amur/vrdOCR/datasets/TextCaps/val_crops'
os.makedirs(output_dir, exist_ok=True)

# Define the file where image paths and labels will be written
output_txt = '/home/amur/Amur/vrdOCR/datasets/TextCaps/val.txt'

json_file_path = '/home/amur/Amur/vrdOCR/datasets/TextCaps/train_val_images/TextOCR_0.1_val.json'  # Replace with the path to your JSON file
with open(json_file_path, 'r') as f:
    x = json.load(f)

special_characters_pattern = re.compile(r'[®©™¶•⅛℠℗ℵℒℲ♔]+')

# Open the output text file in append mode
with open(output_txt, 'a') as label_file:
    # Iterate over the annotations in x['anns']
    for ann_key, annotation in x['anns'].items():
        image_id = annotation['image_id']
        points = annotation['points']
        utf8_string = annotation['utf8_string']
        if utf8_string == '.' or special_characters_pattern.search(utf8_string):  # Skip annotations with empty strings
            continue
        # Load the image using its ID from x['imgs']
        img_info = x['imgs'][image_id]
        img_path = img_info['file_name']
        img_path = os.path.join(root, img_path.replace('train/', ''))
        # Open the image using Pillow
        img = Image.open(img_path)
        
        # Create a polygon from the points
        polygon = Polygon(np.array(points).reshape(-1, 2))
        
        # Get the bounding box of the polygon (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = polygon.bounds
        
        # Crop the image using the bounding box
        cropped_img = img.crop((minx, miny, maxx, maxy))
        
        # Generate a unique filename for the cropped image
        unique_filename = f"{image_id}_{ann_key}.jpg"
        save_path = os.path.join(output_dir, unique_filename)
        
        # Save the cropped image
        cropped_img.save(save_path)
        # Write the path and label to the text file
        label_file.write(f"{save_path}\t{utf8_string}\n")

print("Cropping complete and text file generated.")
