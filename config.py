#configuration file in which we can modify the training parameters

import torch
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 4  
RESIZE_TO = 768  #change inside custom_utils.py!
NUM_EPOCHS = 100
NUM_WORKERS = 12 

CONFIDENCE_THRESHOLD = 0.75
IOU_THRESHOLD = 0.5
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
PATIENCE = 5  # number of epochs to wait before early stopping if no improvement

# location to save model and plots
OUT_SAVEMODEL = r'.\models'
MODEL_TYPE = 'fasterrcnn_resnet50_fpn'
MODEL_NAME = 'fasterrcnn_resNet50_FinalModel'
#MODEL_TYPE = 'fasterrcnn_mobilenet_v3_large_fpn'
#MODEL_NAME = 'fasterrcnn_mobilenet_1'

# Paths
images_dir_training = r'..\aquarium_pretrain\train\images'
labels_dir_training = r'..\aquarium_pretrain\train\labels'
images_dir_validation = r'..\aquarium_pretrain\valid\images'
labels_dir_validation = r'..\aquarium_pretrain\valid\labels'
images_dir_test = r'..\aquarium_pretrain\test\images'
labels_dir_test = r'..\aquarium_pretrain\test\labels'
yaml_path = r'..\aquarium_pretrain\data.yaml'




