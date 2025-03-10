from tensorflow.keras.models import load_model
import librosa
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
import time
import os

classes = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvel", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"]

data_dir = 'audios'
sample_rate = 16000
n_mfcc = 40
max_pad_len = 173  

def preprocess_audio_coral(file_path):
    
    x, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    
    mfccs = np.expand_dims(mfccs, axis=0) 
    mfccs = np.expand_dims(mfccs, axis=-1)
    
    return mfccs

def preprocess_audio_keras(file_path, sample_rate=16000, n_mfcc=40, max_pad_len=173):
    x, sr = librosa.load(file_path, sr=sample_rate)
    
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    
    mfccs = mfccs[np.newaxis, ..., np.newaxis]

    return mfccs

def predict_coral(audio):
    input_data = preprocess_audio_coral(audio)

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(output_data, axis=1)[0]
        
    return predicted_class, end_time - start_time


def predict_keras(audio, model):
    input_data = preprocess_audio_keras(audio)

    start_time = time.time()
    predictions = model.predict(input_data)
    end_time = time.time()

    predicted_class = np.argmax(predictions)

    return predicted_class, end_time - start_time

model_kreas = load_model('keyphrase_detection_model.h5')

model_path = "model_quantized_edgetpu.tflite"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tests = []

for i, class_ in enumerate(classes):
    audios = os.listdir("audios/" + class_)
    cr = 0
    for audio in audios[:5]:
        file_path = "audios/" + class_ + "/" + audio
        keras_p = predict_keras(file_path, model_kreas)
        coral_p = predict_coral(file_path)  
        tests.append((file_path, keras_p[0], classes[keras_p[0]],keras_p[1], coral_p[0], classes[coral_p[0]], coral_p[1]))

for test in tests:
    print(test)