import numpy as np
from pycoral.utils.edgetpu import make_interpreter
import librosa
import os
import sys
import matplotlib.pyplot as plt
import librosa.display

# Ruta al modelo tflite cuantizado
model_path = "model_quantized_edgetpu.tflite"

# Cargar el modelo con el delegado de TPU (para ejecutar en Coral)
interpreter = make_interpreter(model_path)

# Cargar el modelo
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Asegúrate de que el modelo está en modo `int8` y la entrada sea también `int8`
print(f"Input details: {input_details}")
print(f"Output details: {output_details}")

data_dir = 'audios'  # Directorio donde están tus archivos WAV
sample_rate = 16000
n_mfcc = 40
max_pad_len = 173  

def preprocess_audio(file_path):
    
    x, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    
    if len(sys.argv) > 1:
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfccs, x_axis="time", sr=sr, cmap="coolwarm")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Coeficientes MFCC")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Coeficientes")
        plt.tight_layout()
        plt.show()
    
    # Rellenar las secuencias para que todas tengan la misma longitud
    if mfccs.shape[1] < max_pad_len:
        # Añadir ceros (padding) al final para igualar la longitud
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    elif mfccs.shape[1] > max_pad_len:
        # Recortar la secuencia si es más larga
        mfccs = mfccs[:, :max_pad_len]
    
    mfccs = np.expand_dims(mfccs, axis=0)  # Añadir el batch size
    mfccs = np.expand_dims(mfccs, axis=-1)  # Añadir la dimensión de canales (1)
    
    return mfccs

def preprocess_audio_old(file_path):
    # Cargar y preprocesar el audio
    y, sr = librosa.load(file_path, sr=16000)
    
    # Extraer los coeficientes MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Rellenar las secuencias más cortas con ceros para igualar el tamaño
    num_mfcc_features = mfcc.shape[1] 
    if num_mfcc_features < 173:
        mfcc = np.pad(mfcc, ((0, 0), (0, 173 - mfcc.shape[1])), mode='constant')
    elif num_mfcc_features > 173:
        mfcc = mfcc[:, :173]
    
    # Expande las dimensiones: Añadir batch_size (1) y canales (1)
    mfcc = np.expand_dims(mfcc, axis=0)  # Añadir el batch size
    mfcc = np.expand_dims(mfcc, axis=-1)  # Añadir la dimensión de canales (1)
    
    # Verifica las dimensiones del tensor
    return mfcc

def predict(audio):
    input_data = preprocess_audio(audio)

    # Configurar el tensor de entrada
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Realizar la inferencia
    interpreter.invoke()

    # Obtener los resultados
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(output_data, axis=1)
        
    return predicted_class

####test

classes = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvel", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"]

if len(sys.argv) > 1:
    p = predict(sys.argv[1])[0]
    print(sys.argv[1], p, classes[p])
else:
   
    
    num_classes = len(classes)

    for i, class_ in enumerate(classes):

        audios = os.listdir("audios/" + class_)
        cr = 0
        for audio in audios:
            p = predict("audios/" + class_ + "/" + audio)[0]
            if p == i:
                cr += 1 
            print(class_, audio, p)
        
        print(class_, len(audios), cr, cr/len(audios))
    