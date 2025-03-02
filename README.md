# Aquarium-Object-Detection with Faster R-CNN 🚀
This project focuses on underwater object detection using deep learning models, in particular with the Faster R-CNN architecture. The main goal is to develop and optimize a model capable of correctly detecting and classifying species present in underwater images, exploiting real datasets from Kaggle.

## Repository Contents 📂

- **addestramentoTB.py**: Script to train the model.
- **valutazioneTB.py**: Script to evaluate the trained model on the testing dataset.
- **modelVisualizationTest.py**: Script for interactive visualization of results (useful for checking the model's predictions).
- **provaAugmentation.py**: Script to test image transformations and augmentations.
- **solver.py**: Contains the training, validation, and model saving logic.
- **custom_utils.py**: Custom functions and classes, including the dataset and image transformations.
- **config.py**: Configuration file with parameters for training, validation, and dataset paths.

## Prerequisites 🔧

Make sure you have installed:

- Python 3.7 or later
- PyTorch
- torchvision

You can install the necessary packages by running:

```bash
pip install torch torchvision
'''

If you need additional packages (e.g., tensorboard, pyyaml, tqdm, pillow), install them with:
```bash
pip install tensorboard pyyaml tqdm pillow

## Dataset Preparation 📥

Download the dataset from Kaggle using the following link:  
[Aquarium Data COTS](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots/data) 😃

Extract the content and place the dataset in the same folder as the project. Ensure that the folder structure matches what is specified in `config.py`, especially for the directories:
- `images_dir_training` and `labels_dir_training` for training,
- `images_dir_validation` for validation,
- `images_dir_test` for testing.

## How to Use 📝

### 1. Training the Model 💪

To train the model, run the `addestramentoTB.py` script:

```bash
python addestramentoTB.py

This script:
- Loads the training and validation datasets.
- Configures the model according to the parameters defined in config.py.
- Starts the training process, periodically saving the model when an improvement in loss is detected. 👍

### 2. Evaluating the Model 📊
After training the model, run the `valutazioneTB.py` script to evaluate it on the testing dataset:
```bash
python valutazioneTB.py

The script will load the saved model, perform predictions on the testing dataset, and print metrics such as mAP, Precision, Recall, and F1-Score. Additionally, TensorBoard logs will be saved. 🔍

### 3. Interactive Visualization 👀
To start an interactive visualization of the predictions, use the `modelVisualizationTest.py` script:
```bash
python modelVisualizationTest.py

This script allows you to visualize test images with their corresponding bounding boxes and labels, making it easier to analyze the results. ✨

### 4. Support Files 🛠️
The other files in the repository serve the following purposes:

- `custom_utils.py`: Defines the custom dataset and the image transformations (augmentation) for training and testing.
- `solver.py`: Implements the functions for training, validating, and saving the model.
- `tester.py`: Implements the functions for testing the model.
- `provaAugmentation.py`: Provides an example to test and visualize the image transformations.

##Final Notes 💡
Ensure that the Kaggle dataset is placed correctly and that the directories specified in config.py are updated if necessary.
It is recommended to use TensorBoard to monitor the training progress and evaluate performance metrics:
```bash
tensorboard --logdir=runs




