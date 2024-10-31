from trdg.generators import GeneratorFromStrings, GeneratorFromRandom, GeneratorFromWikipedia
import os
import random 

output_dir = 'synth_sents'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

train_annotation_file = os.path.join(train_dir, 'annotations.txt')
val_annotation_file = os.path.join(val_dir, 'annotations.txt')

generator = GeneratorFromStrings(
    ['Factory / Registered Office  : Plot No. N.W.Z./I/P-I, Port Qasim Authority, Karachi. Phone: (92-21) 34720041-47 Fax: (92-21) 34720037.',     
     ': 1-B, 1st Floor, Awan Arcade, Nazimuddin Road, Islamabad. Phone:(92-51) 2810300-01, Fax: (92-51) 2810302',      
     ': Metro Store, Block-G, Link Road, Model Town, Lahore. Phones:(92-42) 35926465.',
     'Certified that the undermentioned vehicle has been sold to:',
     'Document to be furnished by the maker or Authorized Assembler / Sole Distributor in the case of Transport Vehicle other than Motor Cabs.',
     'Document to be furnished by the maker or Authorised Assembler / Sole Distributor in the case of Transport Vehicle other than Motor Cabs.',
     'For INDUS MOTOR COMPANY LIMITED',
     'This Certificate is being issued without any cuttings, alterations or additions.'   
     ],   
    language='/home/amur/.local/lib/python3.8/site-packages/trdg/fonts/amur',
    image_dir='/home/amur/.local/lib/python3.8/site-packages/trdg/images',
    background_type=2,
    distorsion_type=0,
    skewing_angle=0
)
distorted_generator = GeneratorFromStrings(
    generator.strings,
    language=generator.language,
    image_dir=generator.image_dir,
    background_type=generator.background_type,
    distorsion_type=4  
)
skewed_generator = GeneratorFromStrings(
    generator.strings,
    language=generator.language,
    image_dir=generator.image_dir,
    background_type=generator.background_type,
    distorsion_type=0,
    skewing_angle=1
)
num_images = 15000
split_ratio = 0.8
distortion_prob = 0.3
skewing_prob = 0.4

train_limit = int(num_images * split_ratio)

with open(train_annotation_file, 'a') as train_f, open(val_annotation_file, 'a') as val_f:
    for i, (img, lbl) in enumerate(generator):
        # if i <= 46757:
        #     print("skipping", i)
        #     continue 

        if random.random() >= distortion_prob and len(lbl) <= 100:
            img, lbl = next(distorted_generator)
        if random.random() >= skewing_prob:
            img, lbl = next(skewed_generator)

        if i >= num_images:
            break

        if i < train_limit:
            img_name = f'sent-{i:013d}_aug_{i % 100}_{random.randint(0,9999999999999999999999999999)}.jpg'
            img_path = os.path.join(train_dir, img_name)
            img.save(img_path)
            train_f.write(f'{img_path}\t{lbl}\n')
            print(f"Written train image {i}/{train_limit}")
        else:
            img_name = f'sent-{i:013d}_aug_{i % 100}_{random.randint(0,9999999999999999999999999999)}.jpg'
            img_path = os.path.join(val_dir, img_name)
            img.save(img_path)
            val_f.write(f'{img_path}\t{lbl}\n')
            print(f"Written val image {i - train_limit}/{num_images - train_limit}")
