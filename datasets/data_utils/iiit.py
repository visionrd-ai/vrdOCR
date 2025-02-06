from scipy.io import loadmat

train_data = loadmat("/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/trainCharBound.mat")  
train_data = train_data['trainCharBound'][0]

train_file = open('/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/train_converted.txt', 'w')

for train_sample in train_data:
    path = train_sample[0][0]
    label = train_sample[1][0]
    train_file.write(f'{path}\t{label}\n')

train_file.close()


val_data = loadmat("/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/testCharBound.mat")  
val_data = val_data['testCharBound'][0]

val_file = open('/home/amur/Amur/vrdOCR/datasets/IIIT_5k/IIIT5K-Word_V3.0/IIIT5K/val_converted.txt', 'w')

for val_sample in val_data:
    path = val_sample[0][0]
    label = val_sample[1][0]
    val_file.write(f'{path}\t{label}\n')

val_file.close()