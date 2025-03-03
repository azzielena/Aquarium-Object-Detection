import argparse
import os
from torch.utils.data import DataLoader
import yaml
from config import OUT_SAVEMODEL, BATCH_SIZE, PATIENCE, LEARNING_RATE, NUM_EPOCHS, NUM_WORKERS, DEVICE, MODEL_NAME, MODEL_TYPE
from solver import Solver
from custom_utils import CustomDataset, transformsTest, transformsTrain
from torch.utils.tensorboard import SummaryWriter  # Import per TensorBoard
from config import images_dir_training, labels_dir_training, images_dir_validation, labels_dir_validation, NUM_EPOCHS, yaml_path




def get_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    
    # Dati per il modello
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Nome del modello da salvare/caricare')
    parser.add_argument('--model_type', type=str, default=MODEL_TYPE, help='Nome del modello da salvare/caricare')
    # Parametri di training
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Numero di epoche')
    parser.add_argument('--pat', type=int, default=PATIENCE, help='patience early stopping')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Numero di elementi per batch')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='Numero di workers nel data loader')
    parser.add_argument('--print_every', type=int, default=1, help='Stampa le loss ogni N iterazioni')
    # Ottimizzazione
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'], help='Ottimizzatore usato per il training')
    # Path e checkpoint
    parser.add_argument('--checkpoint_path', type=str, default=OUT_SAVEMODEL, help='Percorso per salvare il modello addestrato')
    parser.add_argument('--config_dir', type=str, default='./config', help='Directory dei file di configurazione')
    
    return parser.parse_args()


def main(args):
    # Extract the model name
    model_tag = os.path.splitext(args.model_name)[0]
    #Create a directory for saving the model data
    train_log_dir = os.path.join("runs", model_tag, "train")
    writer_train = SummaryWriter(log_dir=train_log_dir)

    # Create the dataset and DataLoader for training and validation
    dataset = CustomDataset(images_dir_training, labels_dir_training, transforms=transformsTrain)
    dataset_valid = CustomDataset(images_dir_validation, labels_dir_validation, transforms=transformsTest)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=dataset.collate_fn)
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=dataset_valid.collate_fn)

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) 
        # "nc" is the number of classes excluding the background; we add 1 to include the background
        num_classes = data.get("nc", 0) + 1 
    # Create the model 
    model = Solver.get_model(num_classes=num_classes, model_type=args.model_type)
    model.to(DEVICE)


    # Create the solver
    solver = Solver(train_loader=train_loader,
            valid_loader=valid_loader,
            device=DEVICE,
            net = model,
            writer=writer_train,
            args=args)

    # TRAIN model
    solver.train()


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
    print('End of training!')
