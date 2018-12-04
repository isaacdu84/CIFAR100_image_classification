# Machine learning II final project - Group 2
This repository is for the final project of Machine learning II at the George Washington University. Please read the sections below for more detailed instructions on running the scripts

## Multilayer perceptron
MLP in PyTorch.py contains the main training model and accuracy result.<br>
MLP-class-accuracy plot.py contains visualization for top10 accuracy classes in 100 classes and 20 subclasses.<br>
MLP-confusion matrix.py contains confusion matrix for 100 classes and 20 subclasses, also contains confusion matrix plot for 20 subclasses.<br>


## Transfer learning
"Transfer_learning_training.py" includes python scripts that download the Inception model from Google and ingest it into Tensorflow. It also loads the CIFAR100 dataset from Keras API, sends the train and test images into the model and save (using pickle) the output from this model to the local directory for subsequent modeling.<br>
Be adviced that executing this script will take about 1.5 hour on GPU. Fortunately, this only needs to be done once.

"Transfer_learning.py" includes python scripts that load the output from the Inception model and send them through MLP layers. This script also includes lines for visualization of the peformance and confusion matrix.
