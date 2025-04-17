# FOOD-IMAGE-CLASSIFICATION-WITH-SWIN-TRANSFORMER-AND-SVM-CLASSIFIER-
The dataset Food-101 is analyzed by evaluating and comparing the performance of five state-of-the-art models, they are MobileNetV3, ResNet50, EfficientNetV2, CoAtNet, and Swin Transformer. Subsequently, the research proceeds with the implementation of Support Vector Machine (SVM).

## Food101 Dataset
The Food-101 dataset is widely recognized as a prominent benchmark dataset for image classification tasks specifically focused on food recognition. It comprises a diverse collection of 101 distinct food categories, each containing 1,000 images. These images were sourced from various online recipe websites and contributed by both professional and amateur photographers, resulting in a rich and challenging dataset.

![image](https://github.com/user-attachments/assets/a48b5a4b-984d-4e19-82eb-f5b36f3b5e34)

# Five state-of-the-art models
## 1. CoatNet
The CoatNet architecture is designed to improve the generalization performance of the network on image recognition tasks by encouraging the network to learn features that are both informative and diverse. The architecture is based on the ResNet architecture and includes a new co-activation regularization technique.

## 2. ResNet50
The ResNet50 architecture is a deep convolutional neural network (CNN) designed for image classification tasks. It is a 50-layer network, consisting of multiple building blocks, each of which contains several convolutional and batch normalization layers, followed by a rectified linear unit (ReLU) activation function.

## 3. MobileNetV3
MobileNetV3 incorporates various features, such as deep separable convolution, inverted residual structure, lightweight attention model, and hard_sigmoid/hard_swish activation functions, from MobileNetV1, MobileNetV2, and MnasNet. The study combines NAS and NetAdapt techniques to search for the network structure. NAS searches each module in the network with a set amount of computation and parameters, while NetAdapt fine-tunes the network layers once each module is determined.

## 4. EfficientNetV2
EfficientNetV2, which incorporates both MBConv and Fused-MBConv blocks, was developed to overcome certain limitations of the original EfficientNet architecture. Its main goal is to create smaller, more efficient models that still deliver high accuracy across a range of computer vision tasks.

## 5. Swin Transformer
The Swin Transformer introduces a hierarchical design and a shifted window approach, which allows it to model at various scales, accommodating the variations in the scale of visual entities. It constructs hierarchical feature maps by starting with smallsized patches and gradually merging neighboring patches in deeper Transformer layers. This hierarchical representation enables the Swin Transformer to leverage advanced techniques for dense prediction tasks like object detection and semantic segmentation.

![image](https://github.com/user-attachments/assets/cc7fb90e-dbf5-4d04-b444-82055a1db82b)

## Support Vector Machine (SVM)
SVM operates by creating an optimal hyperplane that separates different classes in the feature space. The objective is to find the hyperplane that maximizes the margin between classes, allowing for better generalization and robustness to new data points.

## Pre-Processing
The dataset provides a predefined train/test split, where 75% of the images are designated for training purposes, while the remaining 25% are reserved for evaluation. 
Two common transformations applied to the images are resizing and data augmentation. 

## Swin Transformer feature extraction combined with SVM classifier 
![image](https://github.com/user-attachments/assets/cbd23544-fa90-40d5-8ba3-de8022d0c421)

This study propose utilizing the Swin Transformer model for feature extraction from the training dataset. Following this extraction, the obtained features are inputted into an SVM classifier for training.

![image](https://github.com/user-attachments/assets/866d8e26-43e8-40f2-93c0-63f96ba0bc8c)

In the implementation of the Swin Transformer model, the initial learning rate value was set to 1e-3. The variations in loss values corresponding to changes in the learning rate – encompassing the minimum, steep, valley, and slide points. The valley point, characterized by a learning rate value around 0.00057, was selected as the optimal learning rate for the fine-tuning process. This choice aids the model in converging more effectively and enhancing performance in food recognition tasks.  

![image](https://github.com/user-attachments/assets/b4d57897-45ad-493f-8fc3-8f38701f113c)

The training outcomes for the Swin Transformer model revealed a training loss of 0.444791 and a validation loss of 0.427032. The error rate was recorded at 0.114257, indicating the model's proficiency in minimizing misclassifications. Notably, the top-1 accuracy achieved an impressive 88.57%, showcasing the model's capability to accurately predict the primary class label.  Additionally, the top-5 accuracy which signifies the model's proficiency in identifying the correct class among the top five predictions, achieved an outstanding value of 97.54%.

## SVM Clasifier
This study further leverages the Swin Transformer to extract features from the Food-101 dataset. The resultant features are subsequently trained by an SVM algorithm. The search encompassed both RBF and Linear kernels, with a range of C values. The highest accuracy was achieved when the SVM algorithm employed the RBF kernel with a C value of 1, resulting in a testing accuracy of 91.05%. 




