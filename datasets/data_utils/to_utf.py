import codecs

input_file = "/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/lexicon.txt"
output_file = "IIIT5k_utf8.txt"

# Convert encoding to UTF-8
with codecs.open(input_file, "r", encoding="ISO-8859-1") as f_in:
    with codecs.open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(line)

print("File successfully converted to UTF-8: ", output_file)
