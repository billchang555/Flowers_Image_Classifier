# import packages
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse

class Loading:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'

    def load_data(self):
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
        # Load datasets
        image_dataset = [datasets.ImageFolder(self.train_dir, transform=train_transforms),
                         datasets.ImageFolder(self.test_dir, transform=test_transforms),
                         datasets.ImageFolder(self.valid_dir, transform=validation_transforms)]

        # define the dataloaders
        dataloader = [torch.utils.data.DataLoader(img_dataset[0], batch_size=64, shuffle=True),
                      torch.utils.data.DataLoader(img_dataset[1], batch_size=64),
                      torch.utils.data.DataLoader(img_dataset[2], batch_size=64)]

        return image_dataset, dataloader

class Network:
    def __init__(self, image_dataset, dataloader, arch, lr, hidden_units, epochs, gpu, save_dir):
        self.image_dataset = image_dataset
        self.dataloader = dataloader
        self.arch = 'vgg16'
        self.lr = 0.001
        self.hidden_units = 1024
        self.epochs = 3
        self.gpu = 'cuda' if gpu == True and torch.cuda.is_available() else 'cpu'
        self.save_dir = ''
        self.device = torch.device(self.gpu)
        self.model, self.optimizer, self.criterion = self._setup_nn()

    def _setup_nn(self):
        model = getattr(models, self.arch)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        input_size = model.classifier[0].in_features
        output_size = 102
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, self.hidden_units)),
                                       ('relu1', nn.ReLU()),
                                       ('drop_out1', nn.Dropout(p=0.5)),
                                       ('fc2', nn.Linear(self.hidden_units, output_size)),
                                       ('output', nn.LogSoftmax(dim=1))]))
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.lr)

        return model, optimizer, criterion

    def train(self):
        self.model.to(self.device)
        step = 0
        running_loss = 0
        print_every = 5
        for epoch in range(self.epochs):
            for inputs, labels in self.dataloader[0]:
                step += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if step % print_every == 0:
                    accuracy = 0
                    valid_loss = 0

                    # turn on val mode
                    self.model.eval()

                    with torch.no_grad():
                        for inputs, labels in self.dataloader[2]:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            loss = self.criterion(logps, labels)
                            valid_loss += loss.item()

                            # calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                         f"Valid loss: {valid_loss/len(self.dataloader[2]):.3f}.. "
                    f"Valid accuracy: {accuracy/len(self.dataloader[2]):.3f}")

            running_loss = 0

            # turn off val mode
            self.model.train()

    def test(self):
        self.model.to(self.device)

        self.model.eval()

        accuracy = 0
        with torch.no_grad():
            for image, label in dataloader[1]:
                image, label = image.to(self.device), label.to(self.device)

                logps = self.model(image)

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == label.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Test accuracy:{:.3f}%".format((accuracy/len(dataloader[1]))*100))

    def save(self):
        self.model.class_to_idx = self.dataset[0].class_to_idx
        self.model.to(self.device)
        filepath = self.save_dir + 'checkpoint.pth'
        torch.save({'pretrained':'vgg16',
            'epochs': self.epochs,
            'classifier': self.classifier,
            'learning_rate': self.lr,
            'state_dict':self.model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'class_to_idx':self.model.class_to_idx},
            filepath)
        print("checkpoint path: {}".format(filepath))

if __name__ == '__main__':
    arg = argparse.ArgumentParser(description='Train.py')
    arg.add_argument('--data_dir', dest="data_dir", action="store", default="./flowers/")
    arg.add_argument('--save_dir', dest="save_dir", action="store")
    arg.add_argument('--arch',dest="arch", action="store", default="vgg16", type = str)
    arg.add_argument('--lr', dest="lr", action="store", default=0.001)
    arg.add_argument('--hidden_units', dest="hidden_units", action="store", default=1024, type=int)
    arg.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
    arg.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = arg.parse_args()

    # ETL image data
    Data = Loading(args.data_dir)
    image_dataset, dataloader = Data.load_data()

    # Train and save model
    model = NeuralNetwork(image_dataset, dataloader, args.arch, args.lr, args.hidden_units, args.epochs, args.gpu, args.save_dir)
    model.train()
    model.test()
    model.save()
