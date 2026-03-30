# USAGE
# python inference.py --model vgg
# python inference.py --model resnet
# import the necessary packages
from pyimagesearch import config
from pyimagesearch.classifier import classifier
from pyimagesearch.datautils import get_dataloader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.nn import Softmax
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
from tqdm import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg", choices=["vgg", "resnet"], help="name of the backbone model")
args = vars(ap.parse_args())

# check if the name of the backbone model is VGG
if args["model"] == "vgg":
	# load VGG-11 model
	from torchvision.models import vgg11, VGG11_Weights
	weights = VGG11_Weights.DEFAULT
	baseModel = vgg11(weights=weights)
	
# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	from torchvision.models import resnet18, ResNet18_Weights
	weights = ResNet18_Weights.DEFAULT
	baseModel = resnet18(weights=weights)
	
# # initialize test transform pipeline
# testTransform = transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
# 									transforms.ToTensor(), transforms.Normalize(mean=config.MEAN, std=config.STD)
# 									])

testTransform = transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), transforms.ToTensor(), 
									transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
									])

# calculate the inverse mean and standard deviation
# invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
# invStd = [1/s for s in config.STD]

invMean = [-m/s for (m, s) in zip(weights.transforms().mean, weights.transforms().std)]
invStd = [1/s for s in weights.transforms().std]

# define our denormalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# create the test dataset
testDataset = ImageFolder(config.TEST_PATH, testTransform)

# initialize the test data loader
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)
	
# build the custom model
model = classifier(baseModel=baseModel.to(config.DEVICE), numClasses=2, model=args["model"])
model = model.to(config.DEVICE)

# load the model state and initialize the loss function
model.load_state_dict(torch.load(config.MODEL_PATH))
lossFunc = nn.CrossEntropyLoss()
lossFunc.to(config.DEVICE)

# initialize test data loss
testCorrect = 0
totalTestLoss = 0
soft = Softmax()

# switch off autograd
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# loop over the validation set
	for (image, target) in tqdm(testLoader):
		# send the input to the device
		(image, target) = (image.to(config.DEVICE), target.to(config.DEVICE))
		
		# make the predictions and calculate the validation loss
		logit = model(image)
		loss = lossFunc(logit, target)
		totalTestLoss += loss.item()
		
		# output logits through the softmax layer to get output
		# predictions, and calculate the number of correct predictions
		pred = soft(logit)
		testCorrect += (pred.argmax(dim=-1) == target).sum().item()
		
# print test data accuracy		
print("Test Accuracy:", testCorrect/len(testDataset))

# initialize iterable variable
sweeper = iter(testLoader)

# grab a batch of test data
batch = next(sweeper)
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10))

# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)
	
	# make the predictions
	preds = model(images)
	
	# loop over all the batch
	for i in range(0, config.PRED_BATCH_SIZE):
		# initialize a subplot
		ax = plt.subplot(config.PRED_BATCH_SIZE, 1, i + 1)
		
		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[i]
		image = deNormalize(image).cpu().numpy()
		image = (image * 255).astype("uint8")
		image = image.transpose((1, 2, 0))
		
		# grab the ground truth label
		idx = labels[i].cpu().numpy()
		gtLabel = testDataset.classes[idx]
		
		# grab the predicted label
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDataset.classes[pred]
		
		# add the results and image to the plot
		info = "Ground Truth: {}, Predicted: {}".format(gtLabel, predLabel)
		plt.imshow(image)
		plt.title(info)
		plt.axis("off")
	
	# show the plot
	plt.tight_layout()
	plt.show()