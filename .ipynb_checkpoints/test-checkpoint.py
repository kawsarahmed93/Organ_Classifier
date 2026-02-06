from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torch import nn
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

import pandas as pd

from datasets import NIH_IMG_LEVEL_DS, get_valid_transforms, collate_fn_img_level_ds
from models import DenseNet121, ConvNeXt_Large, ConvNeXt_Small
from configs import all_configs, DEVICE, NIH_DATASET_ROOT_DIR, TEST_CSV_DIR, NIH_CXR_SINGLE_LABEL_NAMES
from trainer_callbacks import set_random_state

def get_args():
    parser = ArgumentParser(description='test')
    parser.add_argument('--run_config', type=str, default='base_model')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_resize_dim', type=int, default=256)
    parser.add_argument('--image_crop_dim', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=20)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # get configs
    run_config = args.run_config  
    configs = all_configs[run_config]
    weight_saving_path = configs['weight_saving_path']
    
    test_df = pd.read_csv(TEST_CSV_DIR)
    test_fpaths = np.array([NIH_DATASET_ROOT_DIR + x for x in test_df['id'].values])
    test_labels = np.stack([np.array(test_df[x]) for x in NIH_CXR_SINGLE_LABEL_NAMES], axis=1).argmax(1) 

    # get dataloader
    print('Loading Baseline dataloader!')
    test_dataset = NIH_IMG_LEVEL_DS(
                        test_fpaths,
                        test_labels,
                        get_valid_transforms(args.image_resize_dim, args.image_crop_dim),
                        )  
         
    test_loader = DataLoader(
                        test_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.n_workers,
                        drop_last=False,
                        collate_fn=collate_fn_img_level_ds,
                        )  
    
    set_random_state(args.seed)
     
    all_targets = []
    all_preds = []
        
    print('Loading Baseline model!')
    model = ConvNeXt_Large(args.num_classes)
    # model = ConvNeXt_Small(args.num_classes)
    # model = DenseNet121(args.num_classes)
        
    checkpoint = torch.load(weight_saving_path+'/checkpoint_best_f1.pth')
    print('loss score: {:.4f}'.format(checkpoint['val_loss']))
    print('f1 score: {:.4f}'.format(checkpoint['val_f1']))
    model.load_state_dict(checkpoint['Model_state_dict'])
    model = model.to(DEVICE)
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model.eval()                  
    del checkpoint

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for itera_no, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = data['image'].to(DEVICE) 
            targets = data['target'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                out = model(images)
                
            all_targets.append(targets.cpu().data.numpy())              
            y_pred = out['logits'].cpu().detach().clone().float().softmax(1).argmax(1)
            all_preds.append(y_pred.numpy())
            
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    f1 = f1_score(all_targets, all_preds, average='macro')
    # print(f'f1 score: {f1}')
    
    print("Macro-F1:", f1_score(all_targets, all_preds, average='macro'))
    print("Weighted-F1:", f1_score(all_targets, all_preds, average='weighted'))
    print("Micro-F1:", f1_score(all_targets, all_preds, average='micro'))
    print("Accuracy:", accuracy_score(all_targets, all_preds))
    print("Balanced accuracy:", balanced_accuracy_score(all_targets, all_preds))
    print(classification_report(all_targets, all_preds, digits=4))
    
    # print(confusion_matrix(all_targets, all_preds))
    
        
if __name__ == '__main__':
    main()