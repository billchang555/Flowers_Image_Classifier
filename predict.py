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

class Checkpoint:
    def __init__(self, imagepath, filepath):
        self.imagepath = imagepath
        self.filepath = filepath

    def load_model(self):
        checkpoint = torch.load(self.filepath)
        model = models.__dict__[checkpoint['pretrained']](pretrained=True)
        model.epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.class_to_idx = checkpoint['class_to_idx']
        model.optimizer = checkpoint['optim']
        model.classifier = checkpoint['classifier']

        return model

    def process_image(self):
        img = Image.open(self.imagepath)
        img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
        img = img_transforms(img)
        img = img.numpy().transpose((2, 0, 1))

        return img

class Prediction:
    def __init__(self, model, img, topk, cat_to_name, gpu):
        self.model = model
        self.img = img
        self.topk = 5
        self.cat_to_name = 'cat_to_name.json'
        with open(self.cat_to_name, 'r') as f:
            self.cat_to_name = json.load(f)
        self.gpu = 'cuda' if gpu == True and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.gpu)

    def predict(self):
        self.model.to(self.device)
        self.model.eval()

        # Convert image from numpy to torch
        img = np.expand_dims(process_image(self.imagepath), axis=0)
        torch_image = torch.from_numpy(img).type(torch.FloatTensor)
        logps = self.model.forward(torch_image)
        ps = torch.exp(logps)

        # Find the top 5 ps
        top_ps, top_labels = ps.topk(self.top_k)

        # Detatch all of the details
        top_ps = np.array(top_ps.detach())[0]
        top_labels = np.array(top_labels.detach())[0]

        # Convert to classes
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        top_labels = [idx_to_class[label] for label in top_labels]

        return top_ps, top_labels

if __name__ == '__main__':
    # Args for python script
    arg = argparse.ArgumentParser(description='predict-file')
    arg.add_argument('imagepath', action="store", type=str)
    arg.add_argument('checkpoint', action="store", type=str)
    arg.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    arg.add_argument('--category_names',  dest="category_names", action="store", default='cat_to_name.json', type=str)
    arg.add_argument('--gpu',default="gpu", action="store", dest="gpu")
    args = parser.parse_args()

    # Load and process image for prediction
    loadCheckpoint = Checkpoint(args.imagepath, args.checkpoint)
    model = loadCheckpoint.load_model()
    img = loadCheckpoint.process_image()

    # Prediction/Inference
    prediction = Prediction(model, img, args.top_k, args.category_names, args.gpu)
    ps, classes = prediction.predict()
    print("Flowers: {}".format(classes)) # top classes
    print("probability: {}".format(ps)) # top probabilities
