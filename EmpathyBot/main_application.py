import matplotlib.pyplot as plt
import cv2
import torch
from model import DeepEmotionRecognitionModel
import torch.nn.functional as F
from torchvision import transforms
from pepper.robot import Pepper
import random, os, time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def detect_face(img):
    """
    This function takes an image as an input and searches for all faces captured in it.
    :param img: the image read using the cv2.imread()
    :return: faces_array, which is a list of all detected faces.
             Each element of the array is a pair of a face area in gray and colored format
    """
    # convert the test image to gray image as OpenCV face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('xml_face_detection/haarcascade_frontalface_default.xml')

    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    faces_array = []
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        faces_array.append((roi_gray, roi_color))

    if len(faces_array) == 0:
        print("No faces detected")
        return None

    return img, faces_array


def load_img(img):
    """
    Correctly adjusts an image and prepares it to be the input to NN by applying pre-defined transformation, packs it
    with Variable.
    :param img: input image that is to be adjusted
    :return: adjusted image
    """
    img = transformation(img).float()
    img = torch.autograd.Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    return img.to(device)


def apply_emotion_recognition(path_to_picture="pictures/338.png", show_photo=False):
    """
    This function takes a photo from the robot as an input and classifies it using pre-trained NN model.
    First step is face detection, in case the face wasn't detected on the picture, None is returned.
    If there is a face, the picture goes to the next stage: NN classification.
    Before inputting the picture to the network, it underpasses some adjustment filters.
    :param path_to_picture: location of the picture taken
    :param show_photo: flag indicating whether to show the result photo or not
    :return: None or NN output
    """
    picture = cv2.imread(path_to_picture)
    classes = ('Angry', 'Disgusted', 'Scared', 'Happy', 'Sad', 'Surprised', 'Neutral')
    detection_result = detect_face(picture)
    if detection_result:
        picture_with_frame, all_faces = detection_result
        print("The number of detected faces is", len(all_faces))
        first_detected_face = cv2.cvtColor(all_faces[0][1], cv2.COLOR_BGR2RGB)
        roi = cv2.cvtColor(first_detected_face, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        nn_input = load_img(roi)
        net = DeepEmotionRecognitionModel()
        net.load_state_dict(torch.load("model_snapshots/deep_emotion-200-64-0.005-0.5488265527277231.pt"))
        net.to(device)
        out = net(nn_input)
        pred = F.softmax(out)
        classs = torch.argmax(pred, 1)
        prediction = classes[classs.item()]
        print(prediction)
        if show_photo:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2
            output_img = cv2.putText(first_detected_face, prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
            plt.imshow(output_img)
            plt.show()
        return prediction
    return None


def gesture_reaction(robot, detected_emotion, mood_dict):
    """
    This function maps the input emotion to a corresponding emotion class, choosing random reaction from the list
    :param robot: robot object
    :param detected_emotion: NN output
    :param mood_dict: dictionary of emotions
    :return: None
    """
    animation = np.random.choice(mood_dict[detected_emotion])
    print("Chosen animation is: ", animation)
    try:
        animation_finished = robot.animation_service.run("animations/[posture]/" + animation, _async=True)
        animation_finished.value()
        return True
    except Exception as error:
        print(error)
        return False


def create_mood_dictionary():
    """
    This function creates a dictionary of gesture reactions on particular emotion input
    :return:
    """
    mood_dict = dict()
    mood_dict["Happy"] = ['Emotions/Positive/Laugh_1',
                          'Emotions/Positive/Laugh_2',
                          'Emotions/Positive/Laugh_3',
                          'Emotions/Positive/Happy_4',
                          'Gestures/Enthusiastic_4',
                          'Gestures/Enthusiastic_5']

    mood_dict["Sad"] = ['Emotions/Negative/Sad_1',
                        'Emotions/Negative/Sad_2',
                        'Emotions/Negative/Disappointed_1',
                        'Gestures/Desperate_1',
                        'Gestures/Desperate_2']

    mood_dict["Neutral"] = ["Emotions/Neutral/AskForAttention_1",
                            "Emotions/Neutral/Suspicious_1",
                            'Emotions/Negative/Bored_1',
                            'Emotions/Positive/Peaceful_1']

    mood_dict["Angry"] = ['Emotions/Negative/Angry_1',
                          'Emotions/Negative/Angry_2',
                          'Emotions/Negative/Angry_3']

    mood_dict["Surprised"] = ["Emotions/Positive/Hysterical_1",
                              'Gestures/Excited_1',
                            ]

    mood_dict["Disgusted"] = [
                              'Gestures/CalmDown_5',
                              'Gestures/CalmDown_6',
                              'Gestures/Desperate_4',
                              'Gestures/Desperate_5',
                              ]

    mood_dict["Scared"] = ["Emotions/Neutral/Innocent_1",
                           'Gestures/CalmDown_5',
                           'Gestures/CalmDown_6'
                           ]
    return mood_dict


def say_random_sentence():
    """
    These sentences are used to trigger some emotions in a person in front of the robot
    :return:
    """
    sentences = [
                 "You look particularly marvelous today",
                 "Have you ever thought of getting on a diet?",
                 "Two guys stole a calendar. They got six months each",
                 "Knock knock. Who is there? It's me, Pepper, and I want to disassemble your emotional state",
                 "Somebody once told me, the world is gonna roll me.",
                 "Empathy is the ability to emotionally understand what other people feel, but I turned off my understanding modules",
                 "I just want to go home.",
                 "Your emotions don't matter to me, but okay, I'll do this",
                 "By looking at you, I can tell that you are a great person",
                 "I'm running out of phrases so please smile, get angry or whatever"
                 ]
    return np.random.choice(sentences)


if __name__ == "__main__":
    ip_address = "10.37.1.249"
    port = 9559
    print("Program Started")

    # Initialize the robot
    robot = Pepper(ip_address, port)
    robot.set_english_language()
    robot.set_awareness(on=False)
    mood_dictionary = create_mood_dictionary()
    say_something = True
    phrase = None

    for i in range(10):
        if say_something:
            phrase = say_random_sentence()
        picture_path = robot.look_for_a_person_and_take_picture(say_something, phrase)
        robot.say("Let me think")
        # Pass the picture taken through the trained model
        # picture_path = "../photos/surprise_face.png"
        NN_emotion_output = apply_emotion_recognition(picture_path)
        if NN_emotion_output:
            robot.say("I think that your emotion is " + NN_emotion_output)

            # Use the prepared reactions dictionary for choosing the right gesture
            gesture_reaction(robot, NN_emotion_output, mood_dictionary)

        print("Program is Finished")
