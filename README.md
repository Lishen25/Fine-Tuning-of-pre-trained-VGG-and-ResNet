# The goal of this project
This project provides a simple implement of using pre-trained convolution networks:
VGG and ResNet. By going through this project, a beginner may have a deeper understanding
of how to fine-tune pre-trained models

# The origin of this project
Most of the code in this project comes from this online tutorial: https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/

The data used in this project comes from this kaggle website: https://www.kaggle.com/datasets/chetankv/dogs-cats-images

# The structure of this project
```
.
├── README.md
├── dataset
│   ├── test_set
│   └── training_set
├── inference.py
├── output
│   ├── ResNet_model.pth
│   ├── ResNet_model_training.png
│   ├── VGG_model.pth
│   └── VGG_model_training.png
├── pyimagesearch
│   ├── __pycache__
│   ├── classifier.py
│   ├── config.py
│   └── datautils.py
└── train.py
```

# The function of each file/folder
pyimagesearch: \
classifier.py: defines the structure of our model based on pre-trained models\
config.py: stores some parameters of this project\
datautils: defines some functions that will be used in other files

train.py: the training process

inference.py: this file is used to show the result/performance of fine-tuned model

output: stores the result of models