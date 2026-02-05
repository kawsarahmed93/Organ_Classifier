from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from trainer import ModelTrainer

import pandas as pd
from datasets import NIH_IMG_LEVEL_DS, get_train_transforms, get_valid_transforms, get_test_transforms, collate_fn_img_level_ds
from models import DenseNet121
from configs import all_configs, NIH_DATASET_ROOT_DIR, NIH_CXR_SINGLE_LABEL_NAMES, TRAIN_CSV_DIR, TEST_CSV_DIR
from trainer_callbacks import set_random_state, AverageMeter, PrintMeter
from sklearn.model_selection import train_test_split

import wandb
import torch.nn as nn


def compute_class_weights_sqrt_inv(train_labels, num_classes, device="cuda"):
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (np.sqrt(counts) + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)

def get_args():
    """
    get command line args
    """
    parser = ArgumentParser(description='Classification_Model')
    parser.add_argument('--run_configs_list', type=str, nargs="*", default=['base_model'])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--n_workers', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4690)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--image_resize_dim', type=int, default=256)
    parser.add_argument('--image_crop_dim', type=int, default=224)
    parser.add_argument('--do_grad_accum', type=bool, default=True)
    parser.add_argument('--grad_accum_step', type=int, default=4)
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--perform_interval_validation', type=bool, default=True)
    parser.add_argument('--interval_validation_step', type=int, default=250)
    parser.add_argument('--use_wandb_log', type=bool, default=False)
    parser.add_argument('--use_focal_loss', type=bool, default=True)
    parser.add_argument('--focal_loss_alpha', type=float, default=0.25)
    parser.add_argument('--focal_loss_gamma', type=float, default=2)
    parser.add_argument('--num_classes', type=int, default=20)
    args = parser.parse_args()
    return args

def main():
    """
    main function
    """

    args = get_args()

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        gpu_id = int(str_id)
        if gpu_id >= 0:
            args.gpu_ids.append(gpu_id)
            
    # check if there are duplicate weight saving paths
    unique_paths = np.unique([ x[1]['weight_saving_path'] for x in all_configs.items() ])
    assert len(all_configs.keys()) == len(unique_paths)
    
    for config_name in args.run_configs_list:
        configs = all_configs[config_name]   
        set_random_state(args.seed)
            
        if args.use_wandb_log:
            ## wandb part
            wandb_log_configs = vars(args)
            wandb_log_configs.update(configs)
            wandb.init(
                project="NIH_CXR_DS_RUNS",
                config=wandb_log_configs,
            )
        
        
        # LOADING DATALOADERS
        train_df = pd.read_csv(TRAIN_CSV_DIR)
        train_fpaths = np.array([NIH_DATASET_ROOT_DIR + x for x in train_df['id'].values])
        train_labels = np.stack([np.array(train_df[x]) for x in NIH_CXR_SINGLE_LABEL_NAMES], axis=1).argmax(1)

        train_fpaths_split, val_fpaths, train_labels_split, val_labels = train_test_split(
                                                                                    train_fpaths,
                                                                                    train_labels,
                                                                                    test_size=0.2,
                                                                                    random_state=args.seed,
                                                                                    shuffle=True,
                                                                                    stratify=train_labels  # optional but recommended for classification
                                                                                )
        # ---- WeightedRandomSampler (balance classes in batches) ----
        num_classes = args.num_classes
        
        # counts per class in *train split only*
        class_counts = np.bincount(train_labels_split, minlength=num_classes).astype(np.float32)
        
        # inverse frequency weights (you can also try sqrt inverse)
        class_weights = 1.0 / (class_counts + 1e-6)
        
        # weight for each sample based on its class
        sample_weights = class_weights[train_labels_split]  # shape: [N_train]
        
        # sampler expects a 1D sequence (list/np array/torch tensor)
        sample_weights = torch.from_numpy(sample_weights).double()
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),   # typically same as training set size
            replacement=True                  # important for balancing
        )

        
        class_weights = compute_class_weights_sqrt_inv(train_labels_split, num_classes, device="cuda")

        test_df = pd.read_csv(TEST_CSV_DIR)
        test_fpaths = np.array([NIH_DATASET_ROOT_DIR + x for x in test_df['id'].values])
        test_labels = np.stack([np.array(test_df[x]) for x in NIH_CXR_SINGLE_LABEL_NAMES], axis=1).argmax(1) 
        
        print('Loading Baseline dataloaders!')
        train_dataset = NIH_IMG_LEVEL_DS(
                            train_fpaths_split,
                            train_labels_split,
                            get_train_transforms(args.image_resize_dim, args.image_crop_dim),
                            )
        val_dataset = NIH_IMG_LEVEL_DS(
                            val_fpaths,
                            val_labels,
                            get_valid_transforms(args.image_resize_dim, args.image_crop_dim),
                            )

        test_dataset = NIH_IMG_LEVEL_DS(
                            test_fpaths,
                            test_labels,
                            get_test_transforms(args.image_resize_dim, args.image_crop_dim),
                            )
        
        train_loader = DataLoader(
                                train_dataset,
                                batch_size=args.batch_size,
                                sampler=train_sampler,   # <-- use sampler
                                shuffle=False,           # <-- MUST be False when sampler is set
                                num_workers=args.n_workers,
                                drop_last=True,
                                collate_fn=collate_fn_img_level_ds,
                            )

        val_loader = DataLoader(
                            val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_workers,
                            drop_last=False,
                            collate_fn=collate_fn_img_level_ds,
                            )
        
        
        print('Loading Baseline model!')
        model = DenseNet121(args.num_classes)
            
        trainer_args = {
                'model': model,
                'Loaders': [train_loader, val_loader],
                'metrics': {
                    'loss': AverageMeter,
                    'f1': PrintMeter,
                    },
                'checkpoint_saving_path': configs['weight_saving_path'],
                'lr': args.lr,
                'epochsTorun': configs['epochs'],
                'gpu_ids': args.gpu_ids,
                'do_grad_accum': args.do_grad_accum,
                'grad_accum_step': args.grad_accum_step,
                'fold': None,
                'use_ema': args.use_ema,
                'perform_interval_validation': args.perform_interval_validation,
                'interval_validation_step': args.interval_validation_step,
                'use_wandb_log': args.use_wandb_log,
                ## problem specific parameters ##
                'use_focal_loss': args.use_focal_loss,
                'focal_loss_alpha': class_weights,
                # 'focal_loss_alpha': args.focal_loss_alpha,
                'focal_loss_gamma': args.focal_loss_gamma,
                'num_classes': args.num_classes, # NO FUNCTION in Trainer
                'method': configs['method'],
                }

        trainer = ModelTrainer(**trainer_args)
        trainer.fit()
        
        if args.use_wandb_log:
            wandb.finish()
            
if __name__ == '__main__':
    main()  