# EmpathyBot
The goal of this project is to model empathy and specifically empathic response coming from a robot. The project is implemented as a semestral project for the cognitive systems course at FEE CTU.
The robot used in the experiment is Pepper robot produced by SoftBank Robotics. It is provided by Cognito lab at Czech Institute of Informatics, Robotics and Cybernetics.
The project is divided into two main parts. The first part is to detect a face of a person using the frontal robot cameras and recognize the current emotional state. The pipeline is following. Whenever the face is detected, it is used as an input to a Deep Convolutional NN, that is pre-trained by ourselves. The NN will label the input by one of the defined emotions:
1) Anger, 
2) Disgust,
3) Fear, 
4) Happiness, 
5) Sadness,
6) Surprise, 
7) Neutral.

The second part is to produce a reaction corresponding to the recognized emotion. The reaction is generated using the robot's API, naoqi.

## Already done:
- **The architecture**: for the NN and forward function;
- **The training script**: located in the root folder;
- **Face detection**: is implemented within the script **face_detection.py**. It uses OpenCV face detection by haarcascade, which is Object Detection Algorithm used to identify faces in an image or a real time video.
- **Useful helper functions**: under the helpers folder, which allow us to easily download and parse the data.

## Goals for Development
- **Find and implement suitable emotion recognition model**: use pytorch (do research on the model to use)
- **Investigate naoqi**: find out the parameters of the photos from the Pepper
- **Think of the way empathy is translated to gesture**: what is a best answer to the specific emotion. Introduce a set of gestures.
- **Code the movements**: transfer gestures to the code driven the robot's movements 

## Useful links
* A similar implementation of one of my friends (it is simple, but we can use it for the start): https://github.com/PeterKillerio/Neural_Networks/tree/master/Tensorflow/Emotion_recognition
* Here's a video of how it works (the result of the project above):
https://www.youtube.com/watch?v=PdgOubpjWac&ab_channel=PeterBas%C3%A1r
* Dataset that contains gray photos of facial expressions, which is used for training:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
* Links to the robot:
  https://www.softbankrobotics.com/emea/en/pepper,
https://en.wikipedia.org/wiki/Pepper_(robot)


# Authors
**Lev Kisselyov**: https://github.com/RancidSpring

**Daria Fedorova**: https://github.com/amdorra57
