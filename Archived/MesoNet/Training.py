import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
import json

from importlib import import_module
import pandas as pd
import argparse

# Define MesoNet model

def init(end_epoch):
    all_models = json.load(open("config.json"))
    ls=[i for i in all_models if all_models[i]["last"]["epoch"]<end_epoch]

    if(len(ls) != 0):
        model_name1 = ls[0]
        model_name = model_name1[:-2]
        Curr_model = getattr(import_module("Modeling"), model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Detect if GPU is available and use it
        model = Curr_model(num_classes=1).to(device)  # Move model to GPU if available
        best_model_path = "../Models/"+model_name1+".pth"
    
        if(all_models[model_name1]["last"]["epoch"]>0):
            model.load_state_dict(torch.load(best_model_path))
            model.to(device)
        else:
            pd.DataFrame(columns=["epoch","train_accuracy","train_loss","val_accuracy","val_loss"]).to_csv("../History/"+model_name1+".csv",index=False)

        model_dict = all_models[model_name1]
        return model, device, model_name1, model_name, best_model_path, model_dict
    else:
        print(f"All model trained for {end_epoch} epochs")
        return False


# Training function
def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    accuracy_metric = BinaryAccuracy().to(device)
    train_loader_tqdm = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += accuracy_metric(torch.sigmoid(outputs), labels.int()).item()
        train_loader_tqdm.set_postfix(loss=running_loss/len(train_loader_tqdm), accuracy=running_accuracy/len(train_loader_tqdm))
    
    train_loss = running_loss / len(dataloader)
    train_accuracy = running_accuracy / len(dataloader)
    return train_loss, train_accuracy

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    accuracy_metric = BinaryAccuracy().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions = torch.sigmoid(outputs)
            correct += accuracy_metric(predictions, labels.int()).item()
            total += labels.size(0)
    accuracy = correct / len(dataloader)
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy

def dataset(input_shape):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize(input_shape),  # Resize images 
        transforms.ToTensor(),        # Convert images to tensor
    ])

    # Create dataset and dataloader
    train_dataset = datasets.ImageFolder(root='../DataSet/real_vs_fake/real-vs-fake/train/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = datasets.ImageFolder(root='../DataSet/real_vs_fake/real-vs-fake/valid/', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)    
    return train_loader,val_loader

def update_config(ls, mode, model_name1):
    dct = dict(zip(["epoch","train_accuracy","train_loss","val_accuracy","val_loss"],ls))
    if(mode=="last"):
        pd.DataFrame([dct]).to_csv("../History/"+model_name1+".csv", mode='a', header=False, index=False)
    all_models = json.load(open("config.json"))
    all_models[model_name1][mode]= dct
    
    with open("config.json", "w") as out_file:
        out_file.write(json.dumps(all_models,indent=4))

        
def main(end_epoch):
    start = init(end_epoch)
    if(start):
        model, device, model_name1, model_name, best_model_path, model_dict = start
        open("trriger.json", "w").write(json.dumps({"stop":False},indent=4))
        print(f"{model_name1 :=^20}")
#         Initialize model, criterion, and optimizer
        criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_loader,val_loader = dataset(model.input_shape[1:])
        
        best_val_loss = model_dict["best"]["val_loss"]# Initialize best validation loss to infinity
        start_epoch = model_dict["last"]["epoch"]
        for epoch in range(start_epoch, end_epoch):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device,start_epoch, end_epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"{model_name1}: Epoch {epoch + 1}/{end_epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")
            update_config(ls=[epoch+1,train_acc,train_loss,val_acc,val_loss], mode="last", model_name1=model_name1)
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                update_config(ls=[epoch+1,train_acc,train_loss,val_acc,val_loss], mode="best", model_name1=model_name1)
                print(f"{model_name1}: Model saved!")
            
            if(json.load(open("trriger.json"))["stop"]):
                quit()

        print(f"{model_name1} Reached {end_epoch} epochs")
        main(end_epoch)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MesoNet with validation")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    main(parser.parse_args().epochs)
