from src.vrd_ocr import vrdOCR 
import json 
import torch 
import cv2 
from utils.postprocess import CTCLabelDecode
import imgaug.augmenters as iaa
import numpy as np

# Load configurations
config = json.load(open('data/config.json', 'r'))
# backbone_config = {"scale": 0.95, "conv_kxk_num": 4}
# head_config = {'name': 'MultiHead', 'head_list': [{'CTCHead': {'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True}, 'Head': {'fc_decay': 1e-05}}}, {'NRTRHead': {'nrtr_dim': 384, 'max_text_length': 150}}], 'out_channels_list': {'CTCLabelDecode': 97, 'NRTRLabelDecode': 100}, 'in_channels': 480}
decoder = CTCLabelDecode(character_dict_path='utils/en_dict.txt', use_space_char=True)

backbone_config = {"scale": 0.95, "conv_kxk_num": 4, "freeze_backbone":False}
head_config = {'name': 'MultiHead', 'head_list': [{'CTCHead': {'Neck': {'name': 'svtr', 'dims': 120, 'depth': 2, 'hidden_dims': 120, 'kernel_size': [1, 3], 'use_guide': True}, 'Head': {'fc_decay': 1e-05}}}, {'NRTRHead': {'nrtr_dim': 384, 'max_text_length': 150}}], 'out_channels_list': {'CTCLabelDecode': 97, 'NRTRLabelDecode': 100}, 'in_channels': 512}

model = vrdOCR(backbone_config=backbone_config, head_config=head_config).cuda()

model.load_state_dict(torch.load('runs/resnet_indus_scratch_docs_20241030_110738/resnet_indus_scratch_docs_e3_b2499.pth'))
model.cuda()
model.eval()

# Read and decode image
img_path = '/home/amur/Amur/ForgeryDetectionV1.2/PaddleOCR/bottom_sents/SC-0091564451001-2863334-20240829-131510_stamped.png_0_aug_0.jpg'
img = cv2.imread(img_path)

# Define imgaug transformations: DecodeImage and RecResizeImg
# Resize to the target shape: 3 channels, height 48, width 640
resize_augmenter = iaa.Resize({"height": 48, "width": 640})

# Apply transformations
img_resized = resize_augmenter(image=img)

# Convert to tensor format (CxHxW) and normalize (0-1 range for model input)
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.cuda().unsqueeze(0)  # Add batch dimension and move to GPU

# Run inference
import time
ts = time.time() 
preds = model(img_tensor)
to = time.time()
print(decoder(preds))
cv2.imwrite('test.png',img)
print("Time taken: ", to-ts)
import pdb; pdb.set_trace()

pass
