# Machine learning II final project - Group 2
This repository is for the final project of Machine learning II at the George Washington University. Please read the sections below for more detailed instructions on running the scripts

## Transfer learning
"Transfer_learning_training.py" includes python scripts that download the Inception model from Google and ingest it into Tensorflow. It also loads the CIFAR100 dataset from Keras API, sends the train and test images into the model and save (using pickle) the output from this model to the local directory for subsequent modeling.<br>
Be adviced that executing this script will take about 1.5 hour on GPU. Fortunately, this only needs to be done once.

"Transfer_learning.py" includes python scripts that load the output from the Inception model and send them through MLP layers. This script also includes lines for visualization of the peformance and confusion matrix.
