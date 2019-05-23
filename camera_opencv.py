from base_camera import BaseCamera
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Load dependecies
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model._make_predict_function()
model.load_weights('facial_expression_model_weights.h5')

emotion_frequency = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        cap = cv2.VideoCapture(Camera.video_source)

        if not cap.isOpened():
            raise RuntimeError('Could not start camera.')


        while True:
            # read current frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # draw rectangle on each face
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

                detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)

                img_pixels /= 255

                predictions = model.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                emotion = emotions[max_index]

                emotion_frequency[emotion] += 1

                cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()
