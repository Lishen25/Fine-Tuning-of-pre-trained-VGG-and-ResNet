# %%
# USAGE
# python train.py --model vgg
# python train.py --model resnet
# import the necessary packages
from pyimagesearch import config
from pyimagesearch.classifier import classifier
from pyimagesearch.datautils import get_dataloader 
from pyimagesearch.datautils import train_val_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg", choices=["vgg", "resnet"], help="name of the backbone model")
args = vars(ap.parse_args()) # --> returns a dict
# There, users should enter the name of the model

# check if the name of the backbone model is VGG
if args["model"] == "vgg":
	# load VGG-11 model
	from torchvision.models import vgg11, VGG11_Weights
	weights = VGG11_Weights.DEFAULT
	baseModel = vgg11(weights=weights)
	
	# freeze the layers of the VGG-11 model
	for param in baseModel.features.parameters():
		param.requires_grad = False
		# vgg.features.parameters() gets the parameters of all the convolutional layers
		# Thus, here in fact freeze all the convolutional layers' parameters
		
# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	from torchvision.models import resnet18, ResNet18_Weights
	weights = ResNet18_Weights.DEFAULT
	baseModel = resnet18(weights=weights)
	
	# define the last and the current layer of the model
	lastLayer = 8
	currentLayer = 1
	
	# loop over the child layers of the model
	for child in baseModel.children(): # -->.children() gets BLOCKS in the model(NOTE: NOT LAYERS)
		# check if we haven't reached the last layer
		if currentLayer < lastLayer:
			# loop over the child layer's parameters and freeze them
			for param in child.parameters():
				param.requires_grad = False
		# otherwise, we have reached the last layers so break the loop
		else:
			break
		# increment the current layer
		currentLayer += 1   
		
# # define the transform pipelines
# trainTransform = transforms.Compose([transforms.RandomResizedCrop(config.IMAGE_SIZE),
# 									 transforms.RandomHorizontalFlip(), transforms.RandomRotation(90),
# 									 transforms.ToTensor(), transforms.Normalize(mean=config.MEAN, std=config.STD)
# 									 ])

# A different way of transformation
trainTransform = transforms.Compose([transforms.RandomResizedCrop(config.IMAGE_SIZE),
									 transforms.RandomHorizontalFlip(), 
									 transforms.RandomRotation(90), transforms.ToTensor(), 
									 transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
									 ])
# However, there is still a problem: when loading the trained local model
# In the inference.py, the transforms.Normalize() with weights.transforms().mean etc. should also be called
# However, in the file there is no weights parameter unless we load it, but we do not know which model we load
# unless we use if-else to judge it.

# create training dataset using ImageFolder
trainDataset = ImageFolder(config.TRAIN_PATH, trainTransform)

# create training and validation data split
(trainDataset, valDataset) = train_val_split(dataset=trainDataset)
# In fact, the operation here is not rigorous, because validation set 
# should not be transformed as the same as training set

# create training and validation data loaders
trainLoader = get_dataloader(trainDataset, config.BATCH_SIZE)
valLoader = get_dataloader(valDataset, config.BATCH_SIZE)

# build the custom model
model = classifier(baseModel=baseModel.to(config.DEVICE), numClasses=2, model=args["model"])
model = model.to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = nn.CrossEntropyLoss()
lossFunc.to(config.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

# initialize the softmax activation layer
softmax = nn.Softmax(dim=1)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataset) // config.BATCH_SIZE
valSteps = len(valDataset) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {
	"trainLoss": [],
	"trainAcc": [],
	"valLoss": [],
	"valAcc": []
}

# Initial parameter for saving the model
best_loss = float('inf')

# loop over epochs
print("[INFO] training the network...")
for epoch in range(config.EPOCHS):
	# set the model in training mode
	model.train()
	
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	
    # loop over the training set
	for (image, target) in tqdm(trainLoader):
		# send the input to the device
		(image, target) = (image.to(config.DEVICE), target.to(config.DEVICE))
		
		# perform a forward pass and calculate the training loss
		logits = model(image)
		loss = lossFunc(logits, target)
		
		# zero out the gradients, perform the backpropagation step, and update the weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# add the loss to the total training loss so far, pass the
		# output logits through the softmax layer to get output
		# predictions, and calculate the number of correct predictions
		totalTrainLoss += loss.item()
		pred = softmax(logits)
		trainCorrect += (pred.argmax(dim=-1) == target).sum().item()
		
    # switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		
		# loop over the validation set
		for (image, target) in tqdm(valLoader):
			# send the input to the device
			(image, target) = (image.to(config.DEVICE), target.to(config.DEVICE))
			
			# make the predictions and calculate the validation loss
			logits = model(image)
			valLoss = lossFunc(logits, target)
			totalValLoss += valLoss.item()
			
			# pass the output logits through the softmax layer to get
			# output predictions, and calculate the number of correct predictions
			pred = softmax(logits)
			valCorrect += (pred.argmax(dim=-1) == target).sum().item()

    # calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	
    # Saving the model with the lowest loss
	if best_loss > avgValLoss:
		torch.save(model.state_dict(), config.MODEL_PATH)
		best_loss = avgValLoss
	
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataset)
	valCorrect = valCorrect / len(valDataset)
	
	# update our training history
	H["trainLoss"].append(avgTrainLoss)
	H["valLoss"].append(avgValLoss)
	H["trainAcc"].append(trainCorrect)
	H["valAcc"].append(valCorrect)
	
	# print the model training and validation information
	print(f"[INFO] EPOCH: {epoch + 1}/{config.EPOCHS}")
	print(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
	print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}")

# %%
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["trainLoss"], label="train_loss")
plt.plot(H["valLoss"], label="val_loss")
plt.plot(H["trainAcc"], label="train_acc")
plt.plot(H["valAcc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)