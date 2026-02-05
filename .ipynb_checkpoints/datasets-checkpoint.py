import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from configs import NIH_DATASET_ROOT_DIR, NIH_CXR_SINGLE_LABEL_NAMES
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(resize=256, crop=224):
    return A.Compose([
        # A.Resize(height=resize, width=resize, p=1.0),
        
        A.RandomResizedCrop(
                    size=(crop, crop),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),


        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.0,
            hue=0.0,
            p=0.5
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], p=1.0)

def get_valid_transforms(resize=256, crop=224):
    return A.Compose([
        A.Resize(height=resize, width=resize, p=1.0),
        A.CenterCrop(height=crop, width=crop, p=1.0),
        A.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225],
                    max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

def get_test_transforms(resize=256, crop=224):
    return get_valid_transforms(resize, crop)


class NIH_IMG_LEVEL_DS(Dataset):
    def __init__(self, xray_fpaths, labels, transform):
        self.xray_fpaths = xray_fpaths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.xray_fpaths)
    
    def __getitem__(self, index):
        # read image
        image = Image.open(self.xray_fpaths[index]).convert('RGB')
        image = np.array(image)

        # transform image
        transformed = self.transform(image=image)
        transformed_image = transformed['image']
        
        # read label
        label = self.labels[index]
        label = torch.tensor(label).long()    
        
        return {
            'image': transformed_image, 
            'target': label,
            }

def collate_fn_img_level_ds(batch):
    x = batch[0]
    keys = x.keys()
    out = {}
    # declare key
    for key in keys:
        out.update({key:[]})
    # append values
    for i in range(len(batch)):
        for key in keys:
            out[key].append(batch[i][key])
    # stack values
    for key in keys:
        out[key] = torch.stack(out[key])
    
    return out

if __name__ == '__main__':
    train_df = pd.read_csv('./LongTailCXR/nih-cxr-lt_single-label_train.csv')
    train_fpaths = np.array([NIH_DATASET_ROOT_DIR + x for x in train_df['id'].values])
    train_labels = np.stack([np.array(train_df[x]) for x in NIH_CXR_SINGLE_LABEL_NAMES], axis=1).argmax(1) 
    
    train_dataset = NIH_IMG_LEVEL_DS(
                        train_fpaths,
                        train_labels,
                        get_train_transforms(256, 224),
                        )
    data = train_dataset[25]
    image, label = data['image'], data['target']
    plt.imshow(image[0], cmap='gray');plt.axis('off');plt.show();