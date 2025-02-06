# Paths to input files
import os 
root = '/home/amur/Amur/vrdOCR/datasets/mjsynth/mnt/ramdisk/max/90kDICT32px/'

input_file = '/home/amur/Amur/vrdOCR/datasets/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val.txt'  # File with the paths and indices
lexicon_file = '/home/amur/Amur/vrdOCR/datasets/mjsynth/mnt/ramdisk/max/90kDICT32px/lexicon.txt'  # File with the word list

# Read lexicon into a list
with open(lexicon_file, 'r') as lexicon:
    lexicon_lines = lexicon.readlines()

# Process the input file
with open(input_file, 'r') as infile, open('/home/amur/Amur/vrdOCR/datasets/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val_formatted.txt', 'w') as outfile:
    for line in infile:
        parts = line.strip().split()
        image_path = parts[0]
        image_path = os.path.join(root,image_path.replace('./',''))
        label_index = int(parts[1]) - 1  # Assuming indices in the input file are 1-based, adjust accordingly
        label = lexicon_lines[label_index+1].strip()
        # Write to the new file in the required format
        outfile.write(f"{image_path}\t{label}\n")

print("New annotation file created as 'annotation_val_formatted.txt'.")
