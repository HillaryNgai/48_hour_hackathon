#!/usr/bin/env python
import os
from importlib import import_module
from flask import Flask, render_template, Response, request
import speech_recognition as sr
import threading


# Audio thread
microphone_message = ""
class audio_thread(threading.Thread):
    def __init__(self, i):
        threading.Thread.__init__(self)
    def run(self):
        global microphone_message
        r = sr.Recognizer()
        r.energy_threshold = 500
        mic = sr.Microphone()
        print('Starting Audio Thread...')
        with mic as source: 
            while True:
                audio = r.listen(source)
                try:
                    microphone_message += ' ' + r.recognize_google(audio)
                    print(microphone_message)
                except:
                    pass

thread1 = audio_thread(1)
thread1.start()

# import camera driver
Camera = import_module('camera_opencv').Camera

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/microphone_feed')
def microphone_feed():
    while True:
        print(microphone_message)
        return Response(microphone_message, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        from camera_opencv import emotion_frequency

        num_of_emotions = sum(emotion_frequency.values())
        emotion_percentage = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0
        }
        for emotion, frequency in emotion_frequency.items():
            emotion_percentage[emotion] = str(round(frequency/num_of_emotions*100, 2)) + '%'

        result = emotion_percentage
        return render_template("result_negative.html", result = result, emotion_frequency = emotion_frequency)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
