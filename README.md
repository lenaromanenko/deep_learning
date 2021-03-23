## Image Recognition with Deep Learning Models

This project includes following sub-projects:
1. Implementing a Feed-Forward-Network from scratch
2. Writing an Image Classifier with pretrained networks: MobileNetV2, ResNet50, VGG16

## 1. Implementing a Feed-Forward-Network from scratch
This introductionary sub-project guides you through the steps of writing your own feed-forward-network for a fundamental understanding of the core principles of the Deep Learning models: https://github.com/lenaromanenko/deep_learning/blob/main/building_neural_network_from_scratch/Feed-Forward-Network.ipynb


## 2. Image Classifier with pretrained networks: MobileNetV2, ResNet50, VGG16
The goal of this project is to compare the different pre-trained networks and to build an image classifier using the best model. The program [image_classifier.py](https://github.com/lenaromanenko/deep_learning/blob/main/pretrained_network/image_classifier.py) accepts different pre-trained networks to find the anemone fishes (Nemo) in the image of an aquarium. Tested pre-trained networks include:

* MobileNetV2
* ResNet50
* VGG16

### Comparison of the three networks side-by-side:
In the direct comparison VGG16 provides the best results. 

<img src="https://github.com/lenaromanenko/deep_learning/blob/main/images/readme_file_images/1.png" width="700" height="350">

## Improving the results of VGG16:
The predictions by VGG16 can be further improved by tweaking the image and the NumPy arrays before analyzing the picture.

### 1. Resizing the picture

Our model learns on the 224 by 224 Numpy arrays which are equal to 224x224 pixels frames. Since the image is too large, many fishes won’t fit into our 224x224 pixels frames. To solve this problem we can set the image width to a smaller size.

<img src="https://github.com/lenaromanenko/deep_learning/blob/main/images/readme_file_images/2.png" width="700" height="350">

### 2. Changing steps for frame iterations
To analyze more objects on the picture we could iterate through the picture in smaller steps. This makes our predictions better but it also increases the time needed for predictions.

<img src="https://github.com/lenaromanenko/deep_learning/blob/main/images/readme_file_images/4.png" width="700" height="350">

### 3. Adding a border to the picture
The fishes close to the borders of our picture would fit into less frames as compared to the fishes which are closer to the center. To solve this problem we can add a frame around our picture.

<img src="https://github.com/lenaromanenko/deep_learning/blob/main/images/readme_file_images/3.png" width="700" height="350">

The effect of adding a border to the picture can be seen below:

<img src="https://github.com/lenaromanenko/deep_learning/blob/main/images/readme_file_images/5.png" width="700" height="350">

