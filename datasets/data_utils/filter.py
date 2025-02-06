import os
from PIL import Image
import cv2 

from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the dictionary file
dictionary_file = "/home/amur/Amur/vrdOCR/utils/en_dict.txt"

# Load dictionary of allowed characters
with open(dictionary_file, 'r') as f:
    allowed_characters = set(f.read().splitlines())

# Function to check if all characters in the label are in the allowed dictionary
def is_label_valid(label):
    return all(char in allowed_characters for char in label)

# Function to check if PaddleOCR produces a valid output (non-empty)
def is_ocr_valid(image_path):
    result = ocr.ocr(image_path, cls=True)
    text = ''
    if result and result[0]:
        for line in result[0]: 
            text += line[-1][0]

    return bool(text.strip()), text  # Check if the OCR result is not empty

# Path to the annotation file
annotation_file = "/home/amur/Amur/vrdOCR/datasets/TextCaps/val.txt"
filtered_annotation_file = "/home/amur/Amur/vrdOCR/datasets/TextCaps/val_filtered.txt"

# Read the annotation file
with open(annotation_file, 'r') as f:
    annotations = f.readlines()

filtered_annotations = []
skipped = 0
for annotation in annotations:
    # Split each line to get the image path and the label
    image_path, label = annotation.strip().split('\t')

    # Check if the image exists and is not corrupted
    if not os.path.exists(image_path):
        continue

    # Check if image height is greater than width
    with Image.open(image_path) as img:
        width, height = img.size
        if width < 20 or height < 10:
            print("Found small image")
            cv2.imwrite('test.png',cv2.imread(image_path))
            skipped += 1
            continue 

        if height > width:
            print("Found uneven image", skipped)
            cv2.imwrite('test.png',cv2.imread(image_path))
            skipped += 1
            # import pdb; pdb.set_trace()
            continue  # Skip the image if height is greater than width

    # Check if all characters in the label are valid
    if not is_label_valid(label):
        print("Found invalid chars", f'skipped')
        cv2.imwrite('test.png',cv2.imread(image_path))
        skipped += 1
        # import pdb; pdb.set_trace()

        continue  # Skip the annotation if label is invalid

    # Check if OCR result is not empty
    # valid, text = is_ocr_valid(image_path)
    # if not valid:
    #     print(f"Found unreadable image, {text}, {skipped}/{len(annotations)}")")
    #     cv2.imwrite('test.png',cv2.imread(image_path))
    #     skipped += 1
    #     # import pdb; pdb.set_trace()
    #     continue  # Skip the image if OCR returns an empty string

    # Add the valid annotation to the list
    filtered_annotations.append(annotation)

# Write the filtered annotations to a new file
with open(filtered_annotation_file, 'w') as f:
    f.writelines(filtered_annotations)

print(f"Filtered annotations have been saved to {filtered_annotation_file}. (Skipped {skipped}/{len(annotations)} annotations)")
