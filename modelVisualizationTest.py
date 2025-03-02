import os
import yaml
import torch
from PIL import Image
from torch.utils.data import DataLoader
import argparse

from config import images_dir_test, labels_dir_test, yaml_path, DEVICE
from config import OUT_SAVEMODEL, BATCH_SIZE, NUM_WORKERS, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODEL_NAME, MODEL_TYPE
from custom_utils import CustomDataset, transformsTest
from solver import Solver
from evaluatorV3 import Tester  # it contains the interactive_visualization method

def get_args():
    parser = argparse.ArgumentParser(description="Interactive Visualization Testing")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Nome del modello da salvare/caricare')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE, help='Tipo di modello')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Numero di elementi per batch')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='Numero di workers nel data loader')
    parser.add_argument('--confidence_th', type=float, default=CONFIDENCE_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--iou_th', type=float, default=IOU_THRESHOLD, help='IoU threshold')
    parser.add_argument('--checkpoint_path', type=str, default=OUT_SAVEMODEL, help='Percorso per il checkpoint del modello')
    parser.add_argument('--config_dir', type=str, default='./config', help='Directory dei file di configurazione')
    return parser.parse_args()

def main(args):
    # Upload YAML file to get class names and number of classes
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    num_classes = data.get("nc", 0) + 1
    names = data.get("names", {}) 
    
    # Create the dataset and DataLoader for testing
    test_dataset = CustomDataset(images_dir_test, labels_dir_test, transforms=transformsTest)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=test_dataset.collate_fn)
    
    # Load the model and checkpoint
    model = Solver.get_model(num_classes, model_type=args.model_type)
    model.to(DEVICE)
    checkpoint_path = os.path.join(args.checkpoint_path, args.model_name + ".pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Puts the model into evaluation mode
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images_tensor = [img.to(DEVICE) for img in images]
            predictions = model(images_tensor)
            for i in range(len(images_tensor)):
                # Converts the image to PIL format
                image_np = images_tensor[i].cpu().detach().permute(1, 2, 0).numpy()
                image_np = (image_np * 255).astype("uint8")
                image_pil = Image.fromarray(image_np)

                # Filter predictions based on confidence threshold
                mask = predictions[i]['scores'] > args.confidence_th
                pred_boxes = predictions[i]['boxes'][mask].cpu().numpy()
                pred_scores = predictions[i]['scores'][mask].cpu().numpy()
                pred_labels = predictions[i]['labels'][mask].cpu().numpy()

                # target
                target_boxes = targets[i]['boxes'].cpu().numpy()
                target_labels = targets[i]['labels'].cpu().numpy()
                
                result = {
                    'pred': {
                        'image': image_pil,
                        'boxes': pred_boxes,
                        'labels': pred_labels,
                        'scores': pred_scores
                    },
                    'target': {
                        'image': image_pil, #it uses the same image
                        'boxes': target_boxes,
                        'labels': target_labels
                    }
                }
                all_results.append(result)
    
    # Create Tester instance (writer is not needed for view only)
    tester = Tester(test_loader=None, args=args, names=names, model=model, writer=None, device=DEVICE)
    # Start the interactive visualization using the obtained results
    tester.interactive_visualization(all_results)

if __name__ == '__main__':
    args = get_args()
    main(args)
