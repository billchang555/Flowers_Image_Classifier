# Flowers_Image_Classifier_Project

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

## Command Line Application
Train a new network on a data set with train.py

Basic Usage : python train.py data_directory
Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
Output: A trained network ready with checkpoint saved for doing parsing of flower images and identifying the species.
Predict flower name from an image with predict.py along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top K most likely classes: python predict.py input checkpoint ---top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_To_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

## Data
The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.
The data need to comprised of 3 folders:

- test
- train
- validate
Generally the proportions should be 70% training 10% validate and 20% test.
