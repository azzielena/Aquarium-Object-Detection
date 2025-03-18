import os
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from config import images_dir_test ,labels_dir_test, yaml_path, DEVICE, OUT_SAVEMODEL, BATCH_SIZE, NUM_WORKERS, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODEL_NAME, MODEL_TYPE
from tester import Tester
from solver import Solver
from custom_utils import CustomDataset, transformsTest
import argparse
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description="Testing configuration")
    
    # Dati per il run e il modello
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Nome del modello da salvare/caricare')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE, help='Nome del modello da salvare/caricare')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Numero di elementi per batch')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='Numero di workers nel data loader')
    parser.add_argument('--confidence_th', type=int, default=CONFIDENCE_THRESHOLD, help='confidence threshold')
    parser.add_argument('--iou_th', type=int, default=IOU_THRESHOLD, help='iou threshold')
   
    # Path e checkpoint
    parser.add_argument('--checkpoint_path', type=str, default=OUT_SAVEMODEL, help='Percorso per salvare il modello addestrato')
    parser.add_argument('--config_dir', type=str, default='./config', help='Directory dei file di configurazione')
    return parser.parse_args()


def main(args):
    model_tag = os.path.splitext(args.model_name)[0]
    test_log_dir = os.path.join("runs", model_tag, "test")
    writer_test = SummaryWriter(log_dir=test_log_dir)
    
    # Upload YAML file to get class names
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    num_classes = data.get("nc", 0) + 1  # + background
    names = data.get("names", {})  # names = {0: 'fish', 1: 'jellyfish', 2: 'shark', ...}

    # Create the dataset and DataLoader for testing
    test_dataset = CustomDataset(images_dir_test, labels_dir_test, transforms=transformsTest)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=test_dataset.collate_fn)

    # Load the model and checkpoint
    model = Solver.get_model(num_classes, model_type=args.model_type)
    model.to(DEVICE)

    checkpoint_path = os.path.join(args.checkpoint_path, args.model_name+".pth") 
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE) 
    model.load_state_dict(checkpoint['model_state_dict']) 

    #test the model
    tester = Tester(args=args, model=model, test_loader=test_loader, writer=writer_test, names=names, device=DEVICE)
    tester.test(DEVICE)



if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
    print('End of testing!')
