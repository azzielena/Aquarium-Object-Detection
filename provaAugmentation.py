import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from custom_utils import CustomDataset, transformsTrain, transformsTest
from config import images_dir_training, labels_dir_training, images_dir_test, labels_dir_test
from torch.utils.data import DataLoader

# Imposta i percorsi alle cartelle delle immagini e delle etichette
images_dirTR = images_dir_training  
labels_dirTR = labels_dir_training  
images_dirEV = images_dir_test  
labels_dirEV = labels_dir_test  

# Parametri per il ridimensionamento
width, height = 768, 768

# Crea un dataset di training e uno di test (in questo esempio uso i test transforms)
dataset_train = CustomDataset(images_dirTR, labels_dirTR, transforms=transformsTrain)
#dataset_test  = CustomDataset(images_dirEV, labels_dirEV, transforms=transformsTest)

# Crea un DataLoader per iterare sul dataset
dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8, collate_fn=CustomDataset.collate_fn)

def visualize_sample(image_tensor, target):
    """
    Visualizza l'immagine, le bounding box e le etichette contenute nel target.
    """
    # Converte il tensore in numpy e trasforma da [C, H, W] a [H, W, C]
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    
    boxes = target["boxes"]
    labels = target["labels"]
    
    # Disegna ogni bounding box e scrive la label associata
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width_box = xmax - xmin
        height_box = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width_box, height_box, 
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        # Aggiunge la label come testo in alto a sinistra della box
        ax.text(
            xmin, ymin, f'Label: {label.item()}',
            bbox=dict(facecolor='yellow', alpha=0.5),
            fontsize=10, color='black'
        )
    
    plt.axis('off')
    plt.show()

# Visualizza alcuni campioni dal dataloader
for i, (images, targets) in enumerate(dataloader):
    # images e targets sono tuple perché usiamo collate_fn che raggruppa per batch
    image_tensor, target = images[0], targets[0]
    visualize_sample(image_tensor, target)
    if i >= 10:  # Visualizza 5 immagini
        break