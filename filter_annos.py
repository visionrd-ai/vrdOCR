import re

# Define the file paths
input_file = "synth_wiki2/annotations.txt"  # Replace with your file path
output_file = "synth_wiki2/annotations_filt.txt"

# Regular expression to match lines with only English characters, numbers, and special characters
pattern = re.compile(r'^[a-zA-Z0-9\s.,;:!?()&\-\'\"]+$')

# Open the input and output files
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Split line by tab to only check the text part for non-English characters
        parts = line.strip().split('\t')
        if len(parts) > 1 and pattern.match(parts[1]):
            outfile.write(line)  # Write the line if it matches the pattern
