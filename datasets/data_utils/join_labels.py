def append_to_file(input_files, output_file):
    with open(output_file, 'w') as outfile:
        # Iterate through each input file
        for file in input_files:
            with open(file, 'r') as infile:
                # Append the content of each file to the output file
                outfile.write(infile.read())

# List of input files to be appended
# input_files = [
#     '/home/amur/Amur/vrdOCR/datasets/TextCaps/train_filtered.txt',
#     '/home/amur/Amur/vrdOCR/datasets/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_train_formatted.txt',
#     '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/train/annotations.txt'
# ]
input_files = [
    '/home/amur/Amur/vrdOCR/datasets/TextCaps/val_filtered.txt',
    '/home/amur/Amur/vrdOCR/datasets/mjsynth/mnt/ramdisk/max/90kDICT32px/annotation_val_formatted.txt',
    '/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/test/annotations.txt'
]
# Output file where the content will be written
output_file = '/home/amur/Amur/vrdOCR/datasets/val_iiit_textcaps_syn90.txt'

# Call the function to append the files
append_to_file(input_files, output_file)
