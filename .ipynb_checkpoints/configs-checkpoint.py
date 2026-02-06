import torch

DEVICE = torch.device("cuda:0")

NIH_DATASET_ROOT_DIR = '../NIH-CXR/images/'

NIH_CXR_SINGLE_LABEL_NAMES = [
                'Atelectasis',                                 
                'Cardiomegaly',                                
                'Consolidation',                             
                'Edema',                                       
                'Effusion',                                   
                'Emphysema',                                  
                'Fibrosis',                                   
                'Hernia',                                      
                'Infiltration',                                
                'Mass',                                        
                'Nodule'  ,                                     
                'Pleural_Thickening',                          
                'Pneumonia',                                  
                'Pneumothorax',                               
                'Pneumoperitoneum',                           
                'Pneumomediastinum',                        
                'Subcutaneous Emphysema',                     
                'Tortuous Aorta',                              
                'Calcification of the Aorta',                 
                'No Finding']                                  

TRAIN_CSV_DIR = './LongTailCXR/nih-cxr-lt_single-label_train.csv'
# VAL_CSV_DIR = './LongTailCXR/nih-cxr-lt_single-label_test.csv'
TEST_CSV_DIR = './LongTailCXR/nih-cxr-lt_single-label_test.csv'

all_configs = {
    'base_model':{
        'weight_saving_path': '../weights/ConvNeXt_l_run1/',
        'epochs': 100,
        'checkpoint_path': None,
        'method': 'base', 
        },
    }