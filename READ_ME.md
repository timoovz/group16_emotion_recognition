---
noteId: "b53482e0ab5611f09d98196b7cfc3e97"
tags: []

---

# Detecting emotions with ML algorithms

The goal of this code is to recognize facial expressions from laptop camera feed. It identifies the face, makes face pictures, extracts features from the images and uses a machine learning algorithm to classify the expression on the image.

[IMPORTANT]: when

### Data preparation
The images are 48x48 matrices with gray-scale values (0 black - 255 white).
The labels for emotions are 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'.

The dataset comes from ...

it is imbalanced, this is fixed by ...

Data is augmented with:
- color jitter
- horizontal flips
- rotating faces (?)
- shifting faces (?)

Determine invariants:
- translation
- rotation
- lighting


### Feature engineering

features extracted:
- HOG features
- landmarks
- gradients
- facial landmarks
- histogram of oriented gradients
- eigenfaces/eigen spaces
- Haar-like features
- pixel
- local binary patterns
- pixel intensity
- Fourier transform (when is this good?)
(for each feature, chosen because ... ; it models ...)

These features are the inputs for the machine learning model. 

### Machine learning implementation
Models used:
- SVM (support vector machine)
    - features used: ...
- MLP
- Random forest
- Fuzzy system 
    - suited for relativly low dimensions; so either use low dimensional features or implement dimensionality reduction


The performance of <model> is 
- accuracy
- balanced accuracy
- confusion matrix


### Images from laptop camera feed

getting live feed is done with (OpenCV or Dlib)

### Integration