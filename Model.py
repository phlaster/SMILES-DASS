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
    

    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        device = get_device()
        class_weights = train_loader.class_weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(device)
        train_story = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "mean_accuracy": []
        }
        for epoch in range(num_epochs):
            train_story["train_loss"].append(
                self._train_epoch(epoch, num_epochs, train_loader.loader, optimizer, criterion, device)
            )
            val_loss, accuracy_scores = self._validate_epoch(epoch, num_epochs, val_loader.loader, criterion, device)
            train_story["val_loss"].append(val_loss)
            train_story["val_accuracy"].append(accuracy_scores)
            train_story["mean_accuracy"].append(np.mean(accuracy_scores))
        return train_story
    
    def _train_epoch(self, epoch, num_epochs, train_loader, optimizer, criterion, device):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Train {epoch+1}/{num_epochs}'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = round(running_loss/len(train_loader), 2)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train loss: {epoch_loss}', end="")
        return epoch_loss

    def _validate_epoch(self, epoch, num_epochs, val_loader, criterion, device):
        self.model.eval()
        running_loss = 0.0
        num_classes = 5
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Validate {epoch+1}/{num_epochs}'):
                images, masks = images.to(device), masks.to(device)
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                for class_id in range(num_classes):
                    class_mask = (masks == class_id)
                    class_correct[class_id] += (predicted[class_mask] == class_id).sum().item()
                    class_total[class_id] += class_mask.sum().item()

        val_loss = round(running_loss / len(val_loader), 2)
        accuracy_scores = np.divide(class_correct, class_total, out=np.ones_like(class_correct, dtype=float), where=class_total != 0)
        acc = round(np.mean(accuracy_scores), 2)
        print(f'Epoch [{epoch+1}/{num_epochs}], Val loss: {val_loss}, Accuracy: {acc}', end="")
        return val_loss, accuracy_scores
        
            
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