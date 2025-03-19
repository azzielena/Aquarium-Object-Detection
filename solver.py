import os
import time
import torchvision
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights

#Solver for training and validating the model.
class Solver(object):

    def __init__(self, train_loader, valid_loader, device, net, writer, args):
        self.args = args
        self.model_name = '{}.pth'.format(self.args.model_name)
        self.net = net
        
        # Choose optimizer
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        
        #for ADAM
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.writer = writer

    #save the model every time you get an improvement in loss during training
    def save_model(self, epoch, model, optimizer):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        },check_path)
        print("Model saved at", check_path)  

    @staticmethod
    def get_model(num_classes, model_type):
        """Static method to obtain the model based on the chosen type."""
        if model_type == "fasterrcnn_resnet50_fpn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=3 
            )
        elif model_type == "fasterrcnn_mobilenet_v3_large_fpn":
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, trainable_backbone_layers=3)
        else:
            raise ValueError(f"Model type '{model_type}' not supported.")
            
        in_features = model.roi_heads.box_predictor.cls_score.in_features # number of input features for the classifier
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # replaces pre-trained head with a new one
        return model
    

    def train_epoch(self, epoch):
        """Method to train the model for a single epoch."""
        train_loss_list = [] 
        prog_bar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch+1} Train") 

        for i, data in enumerate(prog_bar): # for each batch
            self.optimizer.zero_grad() 
            images, targets = data # load images and targets

            images = [image.to(self.device) for image in images] 
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets] 
            
            # The model returns a dictionary of losses
            loss_dict = self.net(images, targets) 
            
            # Define the weights you want to assign to each component
            w_classifier = 1.5
            w_box_reg = 0.75
            w_objectness = 1.0
            w_rpn_box_reg = 0.75 

            # Calculate total loss
            loss_total = (w_classifier * loss_dict['loss_classifier'] +
                        w_box_reg * loss_dict['loss_box_reg'] +
                        w_objectness * loss_dict['loss_objectness'] +
                        w_rpn_box_reg * loss_dict['loss_rpn_box_reg'])
            
            loss_value = loss_total.item()  # tensor to float
            train_loss_list.append(loss_value)
            
            loss_total.backward()
            self.optimizer.step()
            prog_bar.set_description(f"Loss: {loss_value:.4f}")

        avg_loss = sum(train_loss_list) / len(self.train_loader) 
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Method to validate the model for a single epoch"""
        print('Validating')
        valid_loss_list = []
        prog_bar = tqdm(self.valid_loader, total=len(self.valid_loader))
        for i, data in enumerate(prog_bar):
            images, targets = data

            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            with torch.no_grad(): # not calculate gradients
                loss_dict = self.net(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            valid_loss_list.append(loss_value)
            prog_bar.set_description(f"Val Loss: {loss_value:.4f}")
        avg_loss = sum(valid_loss_list) / len(self.valid_loader)
        self.writer.add_scalar('Loss/valid', avg_loss, epoch)
        return avg_loss
    
    def train(self):
        """Main method for training the model"""
        best_val_loss = float('inf')
        patience = self.args.pat  # number of epochs to wait before stopping training
        trigger_times = 0  # counter for early stopping
        self.model.train()
        
        for epoch in range(self.epochs): 
            print(f"\nEPOCH {epoch+1} of {self.epochs}")
            start = time.time()
            train_loss_avg = self.train_epoch(epoch)
            val_loss_avg = self.validate_epoch(epoch)
            end = time.time()
            epoch_time = (end - start) / 60

            print(f"Epoch #{epoch+1} train loss: {train_loss_avg:.3f}")
            print(f"Epoch #{epoch+1} validation loss: {val_loss_avg:.3f}")
            print(f"Took {epoch_time:.3f} minutes for epoch {epoch+1}")
            
            self.writer.add_scalar('Time/epoch_time', epoch_time, epoch)

            # Log of the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', current_lr, epoch)

            if self.args.opt == "Adam":
                self.scheduler.step()

            # Early stopping: check if validation has improved
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                trigger_times = 0  # reset the counter
                # save the current best model
                self.save_model(epoch, self.net, self.optimizer)
                self.writer.add_text('Model/Saved', f"Model saved at epoch {epoch+1} with validation loss: {val_loss_avg:.3f}", epoch)
            else:
                trigger_times += 1
                print(f"Early stopping trigger count: {trigger_times}/{patience}")
                if trigger_times >= patience:
                    print("Early stopping: interruzione dell'allenamento.")
                    break

        self.writer.close()


