# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.

from flask import Flask, render_template, Response
from io import BytesIO
import cv2
import numpy as np
import requests

import time

app = Flask(__name__)

# Configuración de la URL de la cámara
# _URL = 'http://192.168.61.97'
_URL = 'http://192.168.10.28'
_PORT = '81'
_STREAM_ROUTE = '/stream'
_SEP = ':'
stream_url = f"{_URL}{_SEP}{_PORT}{_STREAM_ROUTE}"

# Parámetros para procesamiento de video
MAX_FRAMES = 1000
N = 2
THRESH = 60
ASSIGN_VALUE = 255

# Función de captura de video desde la cámara
def video_capture():
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):

        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                N = 537
                height, width = gray.shape
                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))
                
                noise[random_positions[0], random_positions[1]] = 255

                noise_image = cv2.bitwise_or(gray, noise)

                total_image = np.zeros((height, width * 2), dtype=np.uint8)
                total_image[:, :width] = gray
                total_image[:, width:] = noise_image

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue
# Generador de frames para detección de movimiento

## La ecualizacion del histograma es usado para elevar el contraste de una imagen por medio del uso de histrogramas.
## Clahe es una mejora de Ahe, puesto que este opera en pequeñas regiones elevando asi su contraste pero mucho mejor
## Mientras que Gamma Correction es usado para ajustar la luminancia de la imagen 
##
def detectorDeMovimiento(bg_subtractor=None, diff_method=False, gamma=None, color=None, clahe=None, equa=None):
    cap = cv2.VideoCapture(stream_url)
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("Error al capturar el primer frame.")
        return

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Inicializa variables para calcular FPS
    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Selección del método de procesamiento de movimiento
        if bg_subtractor:
            fgmask = bg_subtractor.apply(gray)  # Sustracción de fondo
            combined_img = fgmask
        elif diff_method:
            diff_frame = cv2.absdiff(prev_frame, gray)  # Diferencia de fotogramas
            prev_frame = gray
            combined_img = diff_frame
        elif gamma:
            inv_gamma = 0.5 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
            combined_img = cv2.LUT(gray, table)
        elif color:
            combined_img = frame
        elif clahe:
            img_clahe = cv2.createCLAHE(clipLimit=3)
            img = img_clahe.apply(gray) + 30
            #_, ordinary_img = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
            combined_img = img
        elif equa:
            img_clahe = cv2.equalizeHist(gray)
            combined_img = img_clahe

        else:
            combined_img = gray  # Solo escala de grises

        # Calcula los FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Dibuja el FPS en la imagen
        cv2.putText(combined_img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

        # Codifica la imagen para transmitirla
        _, buffer = cv2.imencode('.jpg', combined_img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Rutas para los diferentes modos de visualización
@app.route('/video_stream1')
def video_stream1():
    return Response(detectorDeMovimiento(bg_subtractor=cv2.createBackgroundSubtractorMOG2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream2')
def video_stream2():
    return Response(detectorDeMovimiento(diff_method=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream3')
def video_stream3():
    return Response(detectorDeMovimiento(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_stream4')
def video_stream4():
    return Response(detectorDeMovimiento(gamma=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_stream5')
def video_stream5():
    return Response(detectorDeMovimiento(color=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
@app.route('/video_stream6')
def video_stream6():
    return Response(detectorDeMovimiento(clahe=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    
@app.route('/video_stream7')
def video_stream7():
    return Response(detectorDeMovimiento(equa=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta de video genérica con detector de movimiento
@app.route('/video_stream')
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(debug=True)
