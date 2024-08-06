import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import tifffile

class CNN(nn.Module):
    def __init__(self, input_channels, hidden_layers, num_classes = 5):
        super(CNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, hidden_layers[0], kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_layers[-1], num_classes, kernel_size=1, padding=0))
        self.model = nn.Sequential(*layers)

    def _get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        return device
    
    def forward(self, x):
        return self.model(x)
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        device = self._get_device()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.to(device)

        for epoch in range(num_epochs):
            self._train_epoch(epoch, num_epochs, train_loader, optimizer, criterion, device)
            self._validate_epoch(epoch, num_epochs, val_loader, criterion, device)

    def _train_epoch(self, epoch, num_epochs, train_loader, optimizer, criterion, device):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            images, masks = images.to(device), masks.to(device).long()

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    def _validate_epoch(self, epoch, num_epochs, val_loader, criterion, device):
        self.model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        total_samples = 0
        all_masks = []
        all_predictions = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
                images, masks = images.to(device), masks.to(device).long()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += masks.nelement()

                all_masks.append(masks.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())

        all_masks = np.concatenate(all_masks)
        all_predictions = np.concatenate(all_predictions)
        class_scores = self.match_scores(all_masks, all_predictions)
        print(f'Validation Loss: {val_loss/len(val_loader)}')
        acc = 0.0
        for class_name, accuracy in class_scores.items():
            acc += accuracy
            print(f'\tClass: {class_name}, Accuracy score: {accuracy:.2f}%')
        print(f"\tMean accuracy: {acc/len(class_scores):.2f}%")

    def match_scores(self, mask_raw, prediction):
        class_scores = {}
        class_names = ["open water", "settlements", "bare soil", "forest", "grassland"]

        for i in range(5):
            cls_mask = mask_raw == i
            cls_prediction = prediction == i

            correct_pixels = np.sum(cls_mask & cls_prediction)
            total_pixels = np.sum(cls_mask)

            if total_pixels > 0:
                correct_percentage = (correct_pixels / total_pixels) * 100
            else:
                correct_percentage = 0
            class_scores[class_names[i]] = correct_percentage
        return class_scores
            
    def _preprocess_image(self, image_path, target_size=512):
        image = tifffile.imread(image_path)
        image = np.moveaxis(image, [0, 1, 2], [1, 2, 0])
        height, width = image.shape[1:3]
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode='reflect')
        return padded_image.astype(np.float32)

    
    def predict(self, image_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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