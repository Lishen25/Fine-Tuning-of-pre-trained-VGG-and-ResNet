# import the necessary packages
import torch
import os

# define the parent data dir followed by the training and test paths
BASE_PATH = "dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "training_set")
TEST_PATH = os.path.join(BASE_PATH, "test_set")

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# In fact, it is not necessary, in train.py, a different method is applied

# specify training hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 128
PRED_BATCH_SIZE = 4
EPOCHS = 15
LR = 0.0001

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define paths to store training plot and trained model
PLOT_PATH = os.path.join("output", "ResNet_model_training.png")
MODEL_PATH = os.path.join("output", "ResNet_model.pth")