import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import tifffile
from utils import *

class CNN(nn.Module):
    def __init__(self, layers):
        super(CNN, self).__init__()
        self.model = nn.Sequential(*layers)
        print(self.model)

    def forward(self, x):
        return self.model(x)
    

    def calculate_weights(self, class_counts):
        total_pixels = class_counts.sum()
        class_frequencies = class_counts.float() / total_pixels
        median_frequency = torch.median(class_frequencies)
        weights = median_frequency / class_frequencies
        return weights

    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        device = get_device()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(device)
        train_story = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_recall": [],
            "val_prescision": [],
            "val_f1": []
        }
        for epoch in range(num_epochs):
            train_story["train_loss"].append(
                self._train_epoch(epoch, num_epochs, train_loader.loader, optimizer, device)
            )
            scores = self._validate_epoch(epoch, num_epochs, val_loader.loader, device)
            train_story["val_loss"].append(scores[0])
            train_story["val_accuracy"].append(scores[1])
            train_story["val_recall"].append(scores[2])
            train_story["val_prescision"].append(scores[3])
            train_story["val_f1"].append(scores[4])
        return train_story
    
    def _train_epoch(self, epoch, num_epochs, train_loader, optimizer, device):
        self.model.train()
        running_loss = 0.0
        for images, masks, class_counts in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}', leave=False):
            images, masks = images.to(device), masks.to(device)
            class_counts = class_counts.to(device)
            
            # Calculate weights for this batch
            batch_weights = self.calculate_weights(class_counts.sum(dim=0))
            # Create criterion with dynamic weights
            criterion = nn.CrossEntropyLoss(weight=batch_weights)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = round(running_loss/len(train_loader), 2)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {epoch_loss}', end="")
        return epoch_loss

    def _validate_epoch(self, epoch, num_epochs, val_loader, device):
        self.model.eval()
        running_loss = 0.0
        num_classes = 5
        L = len(val_loader)
        accuracy_scores = np.zeros(num_classes)
        recall_scores = np.zeros(num_classes)
        prescision_scores = np.zeros(num_classes)
        f1_scores = np.zeros(num_classes)
        
        with torch.no_grad():
            for images, masks, class_counts in tqdm(val_loader, desc=f'Validate {epoch+1}/{num_epochs}', leave=False):
                images, masks = images.to(device), masks.to(device)
                class_counts = class_counts.to(device)
                
                # Calculate weights for this batch
                batch_weights = self.calculate_weights(class_counts.sum(dim=0))
                
                # Create criterion with dynamic weights
                criterion = nn.CrossEntropyLoss(weight=batch_weights)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                predicted_np = predicted.cpu().numpy()
                masks_np = masks.cpu().numpy()
                accuracy_scores += metric_accuracy(masks_np, predicted_np, num_classes)
                recall_scores += metric_recall(masks_np, predicted_np, num_classes)
                prescision_scores += metric_precision(masks_np, predicted_np, num_classes)
                f1_scores += metric_f1(masks_np, predicted_np, num_classes)

        val_loss = round(running_loss / L, 2)
        accuracy = round(np.mean(accuracy_scores/L), 2)
        recall = round(np.mean(recall_scores/L), 2)
        prescision = round(np.mean(prescision_scores/L), 2)
        f1 = round(np.mean(f1_scores/L), 2)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val loss {val_loss}, Accuracy {accuracy}, Recall {recall}, Prescision {prescision}, F1 {f1}', end="")
        return val_loss, accuracy_scores, recall_scores, prescision_scores, f1_scores
        
            
    def _preprocess_image(self, image_path, target_size=512):
        image = tifffile.imread(image_path)
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        height, width = image.shape[1:3]
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
        return padded_image.astype(np.float32)

    
    def predict(self, image_path):
        device = get_device()
        image = self._preprocess_image(image_path)
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        return predicted_mask
    
    def pickle(self, path):
        torch.save(self.model.state_dict(), path)
        print("model saved")
        
    def unpickle(self, path):
        state_dict = torch.load(path)
        model_arch = self.model.state_dict().keys()
        pickled_arch = state_dict.keys()

        if model_arch != pickled_arch:
            raise ValueError("The architecture of the model does not match the architecture of the pickled model.")

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("model loaded")

def CNN_load(pickle_path):
    state_dict = torch.load(pickle_path)
    input_channels = state_dict['0.weight'].size(1)
    layers = [
        len(torch.load("model.torch")[f'{i}.weight'])
        for i in range(0, len(torch.load("model.torch")), 2)
    ]
    
    num_classes = layers.pop()

    model = CNN(input_channels, layers, num_classes)
    model.unpickle(pickle_path)
    print(model)
    return model