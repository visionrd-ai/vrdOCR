import os
import cv2
import torch 
from data.imaug.label_ops import LabelEncode
class OCRDataset:
    def __init__(self, input_dir, split, transforms = {}):
        """
        Initializes the OCRDataset.

        Args:
        - input_dir (str): Root directory where images are stored.
        - split (str): Split name, either 'train' or 'val'.
        - label_file (str): Path to the label file with image names and OCR labels.
        """
        self.input_dir = input_dir
        self.split = split
        label_file = os.path.join(self.input_dir, self.split, 'annotations.txt')
        self.image_paths = []
        self.labels = []
        self.transforms = transforms
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_name, label = parts
                    img_path = os.path.join(input_dir, split, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
        self.encoder = LabelEncode(
            max_text_length = 150,
            character_dict_path = 'utils/en_dict.txt',
            use_space_char = False,
            gtc_encode = 'NRTRLabelEncode'
            )
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding label at the given index.

        Args:
        - idx (int): Index of the item.

        Returns:
        - img (ndarray): Loaded image in BGR format.
        - label (str): Corresponding OCR label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        # try:
        encoded_label = self.encoder(label)
        encoded_label = [torch.tensor(val) for val in encoded_label.values()]
        # except:
        #     print(f"\n\nLABEL: {label}\nENC: {encoded_label}")
        #     import pdb; pdb.set_trace()
        # Return individual items (image, label) instead of combining them into a batch
        return [torch.tensor(img)] + encoded_label + [torch.tensor(1.0)]
