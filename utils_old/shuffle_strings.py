import random

def shuffle_special_chars(s, num_versions=15):
    special_chars = ['.', ',', ':']  # List of special characters to shuffle
    indices = [i for i, c in enumerate(s) if c in special_chars]  # Get positions of special chars
    
    versions = []
    
    for _ in range(num_versions):
        new_s = list(s)  # Convert string to list for mutable operations
        shuffled_indices = indices.copy()
        random.shuffle(shuffled_indices)  # Shuffle indices
        
        # Swap characters based on shuffled positions
        for orig_idx, new_idx in zip(indices, shuffled_indices):
            new_s[new_idx] = s[orig_idx]
        
        versions.append("".join(new_s))  # Convert list back to string
    
    return versions

def remove_random_special_char(s):
    special_chars = ['.', ',', '/', ':', '-']  # List of special characters to consider
    special_indices = [i for i, c in enumerate(s) if c in special_chars]  # Find all indices of special chars
    
    if not special_indices:
        # If no special characters found, return the original string
        return s

    # Select a random index among the special character positions
    random_index = random.choice(special_indices)
    
    # Remove the selected character by excluding it in the new string
    new_s = s[:random_index] + s[random_index + 1:]
    
    return new_s

# Prepare input strings and number of copies to create
input_strs = [

    ("Factory / Registered Office : Plot No. N.W.Z./I/P-I, Port Qasim Authority, Karachi. Phone: (92-21) 34720041-47 Fax: (92-21) 34720037.", 15),
    (": Plot No. N.W.Z./I/P-I, Port Qasim Authority, Karachi. Phone: (92-21) 34720041-47 Fax: (92-21) 34720037.", 15),
    
    (": 1-B, 1st Floor, Awan Arcade, Nazimuddin Road, Islamabad. Phone:(92-51) 2810300-01, Fax: (92-51) 2810302", 15),
    ("Islamabad Office : 1-B, 1st Floor, Awan Arcade, Nazimuddin Road, Islamabad. Phone:(92-51) 2810300-01, Fax: (92-51) 2810302", 15),

    (": Metro Store, Block-G, Link Road, Model Town, Lahore. Phones:(92-42) 35926465.", 15),
    ("Lahore Office : Metro Store, Block-G, Link Road, Model Town, Lahore. Phones:(92-42) 35926465.", 15),

    ("This Certificate is being issued without any cuttings, alterations or additions.", 2),
    ("Islamabad Office", 1),
    ("Lahore Office", 1),
    ("Factory / Registered Office :", 1),
    
    ("(Authorized Signature)", 1),
    ("Authorized Signature)", 1),
    ("(Authorized Signature", 1),
    ("Authorized Signature", 1),

    ("(Authorised Signature)", 1),
    ("Authorised Signature)", 1),
    ("(Authorised Signature", 1),
    ("Authorised Signature", 1),

]

file_path = 'utils/synth_strings.txt'

# Open the file in 'a+' mode and read existing lines to avoid duplicates
with open(file_path, 'a+') as synth_strings:
    synth_strings.seek(0)  # Move to start to read existing lines
    existing_lines = set(line.strip() for line in synth_strings)  # Store unique lines already in file

    new_lines = set()  # To track unique lines in this execution
    
    for input_str, copies in input_strs:
        randomly_shuffled = shuffle_special_chars(input_str, num_versions=copies)
        randomly_removed = [remove_random_special_char(input_str) for _ in range(copies)]

        # Write each unique shuffled and removed string to the file
        for shuf, rem in zip(randomly_shuffled, randomly_removed):
            if shuf not in existing_lines and shuf not in new_lines:
                synth_strings.write(f'{shuf}\n')
                new_lines.add(shuf)  # Add to the set of new unique lines

            if rem not in existing_lines and rem not in new_lines:
                synth_strings.write(f'{rem}\n')
                new_lines.add(rem)  # Add to the set of new unique lines
