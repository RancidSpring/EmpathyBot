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
7) Neutral emotional state.

The second part is to produce a reaction corresponding to the recognized emotion. The reaction is generated using the robot's API, naoqi.

## Running the project  

## Useful links
* Dataset that contains gray photos of facial expressions, which is used for training:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
* Links to the robot:
  https://www.softbankrobotics.com/emea/en/pepper,
https://en.wikipedia.org/wiki/Pepper_(robot)


# Authors
**Lev Kisselyov**: https://github.com/RancidSpring

**Daria Fedorova**: https://github.com/amdorra57
