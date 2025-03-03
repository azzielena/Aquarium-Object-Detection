import matplotlib.pyplot as plt
import matplotlib.patches as patches
from custom_utils import CustomDataset, transformsTrain
from config import images_dir_training, labels_dir_training
from torch.utils.data import DataLoader


def visualize_sample(image_tensor, target):
    """
    Displays the image, bounding boxes, and labels contained in the target.
    """
    # Converts tensor to numpy and transforms from [C, H, W] to [H, W, C]
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    
    boxes = target["boxes"]
    labels = target["labels"]
    
    # Draws each bounding box and writes the associated label
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width_box = xmax - xmin
        height_box = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width_box, height_box, 
            linewidth=2, edgecolor='white', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin, f'Label: {label.item()}',
            bbox=dict(facecolor='yellow', alpha=0.5),
            fontsize=10, color='black'
        )
    
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    dataset_train = CustomDataset(images_dir_training, labels_dir_training , transforms=transformsTrain)
    dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8, collate_fn=dataset_train.collate_fn)

    # View some samples from the dataloader
    for i, (images, targets) in enumerate(dataloader):
        image_tensor, target = images[0], targets[0]
        visualize_sample(image_tensor, target)
        if i >= 10:  # 10 images
            break
