import time
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from matplotlib.patches import Rectangle
from torchvision.transforms import functional as Ftv
from random import sample

class Tester(object):
    """Tester for testing the model."""

    def __init__(self, test_loader, args, names, model, writer, device):
        self.confidence_threshold = args.confidence_th
        self.iou_threshold = args.iou_th
        self.args = args
        self.model = model
        self.writer = writer
        self.names = names
        self.test_loader = test_loader
        self.device = device
        # Lists to save calculated mAP and FPS values
        self.map_values = []
        self.map50_values = []
        self.map75_values = []
        self.fps_values = []

    @staticmethod
    def compute_iou(boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) between two boxes [xmin, ymin, xmax, ymax].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def evaluate_results_mAP(self, all_results): #calculate the mAP
        preds = []
        targets = []
        
        for res in all_results:
            # Prepare predictions and targets for TorchMetrics
            pred = {
                "boxes": torch.as_tensor(res['pred']['boxes']),
                "scores": torch.as_tensor(res['pred']['scores']),
                "labels": torch.as_tensor(res['pred']['labels'])
            }
            target = {
                "boxes": torch.as_tensor(res['target']['boxes']),
                "labels": torch.as_tensor(res['target']['labels'])
            }
            preds.append(pred)
            targets.append(target)
        
        # Calculate detection metrics with TorchMetrics (Mean Average Precision)
        map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        map_metric.update(preds, targets)
        detection_metrics = map_metric.compute()
        
        return detection_metrics

    def compute_metrics(self, all_results): #calculate precision recall & f1score
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for res in all_results:
            pred_boxes = res['pred']['boxes']
            pred_labels = res['pred']['labels']
            target_boxes = res['target']['boxes']
            target_labels = res['target']['labels']

            #it ​​is used to keep track of targets already matched to a prediction
            matched = [False] * len(target_boxes) 
            
            tp = 0
            fp = 0
            # For each prediction (box + label), check if it matches a target
            for box, label in zip(pred_boxes, pred_labels):
                found_match = False
                for j, (tb, tl) in enumerate(zip(target_boxes, target_labels)):
                    if matched[j]:
                        continue
                    if int(label) == int(tl) and self.compute_iou(box, tb) >= self.iou_threshold:
                        tp += 1
                        matched[j] = True
                        found_match = True
                        break
                if not found_match:
                    fp += 1
            fn = len(target_boxes) - sum(matched) #number of targets that were not matched
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def interactive_visualization(self, results):
        # current_index as a list to make it mutable inside the callback
        current_index = [0]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        def update_figure():
            for ax in axes:
                ax.clear()
                ax.axis("off")

            # Seleziona il risultato corrente
            index = current_index[0]
            pred = results[index]['pred']
            target = results[index]['target']
            pred_img = np.array(pred['image'])
            target_img = np.array(target['image'])

            # Visualize prediction image
            axes[0].imshow(pred_img)
            axes[0].set_title("Prediction")
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                correct = False
                for tb, tl in zip(target['boxes'], target['labels']):
                    if int(label) == int(tl) and Tester.compute_iou(box, tb) >= self.iou_threshold:
                        correct = True
                        break
                color = 'green' if correct else 'red'
                label_text = self.names[int(label)-1] if label > 0 else "bg"
                x0, y0, x1, y1 = map(int, box)
                rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor=color, facecolor='none')
                axes[0].add_patch(rect)
                axes[0].text(x0, max(y0-5, 0), f"{label_text}:{score:.2f}", color=color,
                             fontsize=14)

            # Visualize target image
            axes[1].imshow(target_img)
            axes[1].set_title("Target")
            for box, label in zip(target['boxes'], target['labels']):
                label_text = self.names[int(label)-1] if int(label) > 0 else "bg"
                x0, y0, x1, y1 = map(int, box)
                rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='blue', facecolor='none')
                axes[1].add_patch(rect)
                axes[1].text(x0, max(y0-5, 0), label_text, color='blue',
                             fontsize=14)

            fig.suptitle(f"Image {index+1} / {len(results)}", fontsize=16)
            plt.tight_layout()
            fig.canvas.draw_idle() #update the image

        def on_key(event):
            if event.key in ['right', 'd']:
                current_index[0] = (current_index[0] + 1) % len(results)
                update_figure()
            elif event.key in ['left', 'a']:
                current_index[0] = (current_index[0] - 1) % len(results)
                update_figure()
            elif event.key in ['escape', 'q']:
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        update_figure()
        plt.show()

    #used to save some random images on tensorboard
    def log_images_with_boxes(self, all_results, step=0, max_images=5):
        max_images = min(max_images, len(all_results))
        chosen_indices = sample(range(len(all_results)), max_images) #random indices

        for idx in chosen_indices:
            img_pil = all_results[idx]['pred']['image'].copy()
            draw = ImageDraw.Draw(img_pil)
            for box, label, score in zip(all_results[idx]['pred']['boxes'],
                                         all_results[idx]['pred']['labels'],
                                         all_results[idx]['pred']['scores']):
                color = 'red'
                for tb, tl in zip(all_results[idx]['target']['boxes'], all_results[idx]['target']['labels']):
                    if int(label) == int(tl) and Tester.compute_iou(box, tb) >= self.iou_threshold:
                        color = 'green'
                        break
                x0, y0, x1, y1 = box
                draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
                label_text = f"{self.names[int(label)-1]}:{score:.2f}" if label > 0 else "bg"
                draw.text((x0, y0-10), label_text, fill=color)

            tensor_img = Ftv.to_tensor(img_pil) #convert Pil to tensor
            model_tag = self.args.model_name.split('.')[0]
            self.writer.add_image(f'{model_tag}/Sample_{idx}', tensor_img, step)

    def test(self, device):
        self.model.eval()
        all_results = []
        total_images = 0
        total_model_time = 0  #time for the forward of the model
        
        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Testing"):
                # Prepare images: move to device
                images_tensor = [img.to(device) for img in images]
                total_images += len(images_tensor)
                
                # measurement of test time
                start_model_time = time.time()
                predictions = self.model(images_tensor)
                end_model_time = time.time()
                total_model_time += (end_model_time - start_model_time)

                for i in range(len(images_tensor)):  # for each image of the batch
                    image_np = images_tensor[i].cpu().detach().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
                    image_np = (image_np * 255).astype("uint8")  # Normalize to [0, 255]
                    image_pil = Image.fromarray(image_np) 

                    # Filter predictions based on confidence threshold
                    mask = predictions[i]['scores'] > self.confidence_threshold
                    pred_boxes = predictions[i]['boxes'][mask].cpu().numpy()
                    pred_scores = predictions[i]['scores'][mask].cpu().numpy()
                    pred_labels = predictions[i]['labels'][mask].cpu().numpy()

                    # Target
                    target_boxes = targets[i]['boxes'].cpu().numpy()
                    target_labels = targets[i]['labels'].cpu().numpy()
                    target_image = image_pil

                    result = {
                        'pred': {
                            'image': image_pil,
                            'boxes': pred_boxes,
                            'labels': pred_labels,
                            'scores': pred_scores
                        },
                        'target': {
                            'image': target_image,
                            'boxes': target_boxes,
                            'labels': target_labels
                        }
                    }
                    all_results.append(result)
        
        # Calculate FPS
        avg_fps = total_images / total_model_time if total_model_time > 0 else 0
        
        # Calculate mAP
        detection_metrics = self.evaluate_results_mAP(all_results)
        mAP = detection_metrics['map']
        mAP50 = detection_metrics['map_50']
        mAP75 = detection_metrics['map_75']
        
        precision, recall, f1 = self.compute_metrics(all_results)
        
        # Salva e logga i valori nel writer
        current_step = len(self.map_values) - 1
        model_tag = self.args.model_name.split('.')[0]
        self.writer.add_scalar(f'{model_tag}/Evaluation/mAP', mAP, current_step)
        self.writer.add_scalar(f'{model_tag}/Evaluation/mAP50', mAP50, current_step)
        self.writer.add_scalar(f'{model_tag}/Evaluation/mAP75', mAP75, current_step)
        self.writer.add_scalar(f'{model_tag}/Evaluation/Avg_FPS', avg_fps, current_step)
        self.writer.add_scalar(f'{model_tag}/Evaluation/Precision', precision, current_step)
        self.writer.add_scalar(f'{model_tag}/Evaluation/Recall', recall, current_step)
        self.writer.add_scalar(f'{model_tag}/Evaluation/F1_Score', f1, current_step)
        

        # Stampa i valori
        print(f"mAP: {mAP:.3f}, mAP@50: {mAP50:.3f}, mAP@75: {mAP75:.3f}, Avg FPS: {avg_fps:.2f}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        # Logging delle immagini con bounding box e visualizzazione interattiva
        self.log_images_with_boxes(all_results, step=current_step, max_images=5)
        self.interactive_visualization(all_results)

        self.writer.close()
