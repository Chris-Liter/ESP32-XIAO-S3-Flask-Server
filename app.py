# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.

from flask import Flask, render_template, Response, request, jsonify
from io import BytesIO
import cv2
import numpy as np
import requests

import time

import random 
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

current_kernel_size = 1

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
            
          
def reducir_ruido(i, frame):
    kernel = np.ones((i, i), np.uint8)
    imagen = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel=kernel)
    return imagen


@app.route('/set_kernel_size', methods=['POST'])
def set_kernel_size():
    global current_kernel_size
    data = request.get_json()
    current_kernel_size = int(data.get('kernelSize'))  # Actualiza el tamaño del kernel
    return jsonify({"status": "success", "kernelSize": current_kernel_size})

@app.route('/set_noise_level', methods=['POST'])
def set_noise_level():
    global noise_level
    data = request.get_json()
    noise_level = float(data.get('noiseLevel')) / 10000  # Ajusta el nivel de ruido
    return jsonify({"status": "success", "noiseLevel": noise_level})
            
def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1-prob
    for i in range (image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# def sp_noise(image, prob):
#     """
#     Agrega ruido sal y pimienta a la imagen.
#     prob: probabilidad de que un píxel se convierta en ruido.
#     """
#     output = np.copy(image)
#     # Genera ruido aleatorio en la imagen
#     black = np.random.rand(*image.shape[:2]) < prob
#     white = np.random.rand(*image.shape[:2]) < prob
#     output[black] = 0
#     output[white] = 255
#     return output


# Generador de frames para detección de movimiento

## La ecualizacion del histograma es usado para elevar el contraste de una imagen por medio del uso de histrogramas.
## Clahe es una mejora de Ahe, puesto que este opera en pequeñas regiones elevando asi su contraste pero mucho mejor
## Mientras que Gamma Correction es usado para ajustar la luminancia de la imagen 
##
def detectorDeMovimiento(bg_subtractor=None, diff_method=False, gamma=None, color=None, clahe=None, equa=None, ruido=None):
    global current_kernel_size
    cap = cv2.VideoCapture(stream_url)
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("Error al capturar el primer frame.")
        return

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
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
        
        elif ruido:
            noise_level = current_kernel_size / 100.0  # Valor de ruido basado en el control deslizante
            noisy_img = sp_noise(gray, noise_level)  # Aplica ruido a la imagen

            kernel_size = max(1, 3 * (100 - current_kernel_size) // 100)  # Aumenta el kernel a medida que disminuye el ruido
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            combined_img = cv2.morphologyEx(noisy_img, cv2.MORPH_DILATE, kernel)
        
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # noisy_img = sp_noise(frame, 0.05)
            # filtered_img = cv2.medianBlur(noisy_img, 5)  # Cambia el tamaño del kernel si es necesario

            
            # kernel = np.ones((current_kernel_size, current_kernel_size), np.uint8)
            # combined_img = cv2.morphologyEx(filtered_img, cv2.MORPH_DILATE, kernel=kernel)

        
        
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

def parte2( erosion=None, dilatacion=None, tophat=None, blackhat=None, combinado=None):
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    if dilatacion:
            image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Dilatar el frame
            erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_DILATE, elemento, iterations=3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row1 = np.hstack((image_rgb, erosion_resized))
            #################################################################################
            image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

            # Dilatar el frame
            erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_DILATE, elemento, iterations=3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row2 = np.hstack((image_rgb2, erosion_resized2))
            
            image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)

            # Dilatar el frame
            erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_DILATE, elemento, iterations=3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row3 = np.hstack((image_rgb3, erosion_resized3))
            
            combined_img = np.vstack((row1, row2, row3))

            
            
    elif erosion:
            image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Dilatar el frame
            erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_ERODE, elemento, iterations=3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row1 = np.hstack((image_rgb, erosion_resized))
            #################################################################################
            image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

            # Dilatar el frame
            erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_ERODE, elemento, iterations=3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row2 = np.hstack((image_rgb2, erosion_resized2))
            
            image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)

            # Dilatar el frame
            erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_ERODE, elemento, iterations=3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row3 = np.hstack((image_rgb3, erosion_resized3))
            
            combined_img = np.vstack((row1, row2, row3))
        
    elif tophat:
            image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

            # Dilatar el frame
            erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_TOPHAT, elemento)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row1 = np.hstack((image_rgb, erosion_resized))
            ###########################################################################
            image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            elemento2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

            # Dilatar el frame
            erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_TOPHAT, elemento2)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row2 = np.hstack((image_rgb2, erosion_resized2))
            ###########################################################################
            image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
            elemento3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

            # Dilatar el frame
            erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_TOPHAT, elemento3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row3 = np.hstack((image_rgb3, erosion_resized3))
            
            combined_img = np.vstack((row1, row2, row3))

            
        
    elif blackhat:
            image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

            # Dilatar el frame
            erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_BLACKHAT, elemento)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row1 = np.hstack((image_rgb, erosion_resized))
            ###########################################################################
            image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            elemento2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

            # Dilatar el frame
            erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_BLACKHAT, elemento2)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row2 = np.hstack((image_rgb2, erosion_resized2))
            ###########################################################################
            image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
            # Convertir a RGB
            image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
            elemento3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

            # Dilatar el frame
            erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_BLACKHAT, elemento3)

            # Redimensionar erosion para que coincida con image_rgb
            erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

            # Unir las dos imágenes horizontalmente (original y erosionada)
            row3 = np.hstack((image_rgb3, erosion_resized3))
            
            combined_img = np.vstack((row1, row2, row3))
        
    elif combinado:
            image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

            # Calcular Top Hat y Black Hat
            top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, elemento)
            black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb = cv2.add(image, cv2.subtract(top_hat, black_hat))
            row1 = np.hstack((image, top_hat, black_hat, comb))
            
            image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            # Calcular Top Hat y Black Hat
            top_hat2 = cv2.morphologyEx(image2, cv2.MORPH_TOPHAT, elemento)
            black_hat2 = cv2.morphologyEx(image2, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb2 = cv2.add(image2, cv2.subtract(top_hat2, black_hat2))
            row2 = np.hstack((image2, top_hat2, black_hat2, comb2))
            
            image3 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            # Calcular Top Hat y Black Hat
            top_hat3 = cv2.morphologyEx(image3, cv2.MORPH_TOPHAT, elemento)
            black_hat3 = cv2.morphologyEx(image3, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb3 = cv2.add(image3, cv2.subtract(top_hat3, black_hat3))
            row3 = np.hstack((image3, top_hat3, black_hat3, comb3))
            
            combined_img = np.vstack((row1, row2, row3))

            
    
    _, buffer = cv2.imencode('.jpg', combined_img)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    

# Rutas para los diferentes modos de visualización
@app.route('/video_stream1')
def video_stream1():
    return Response(detectorDeMovimiento(bg_subtractor=cv2.createBackgroundSubtractorMOG2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#########################################################################
@app.route('/video_stream2')
def video_stream2():
    return Response(detectorDeMovimiento(diff_method=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#########################################################################
@app.route('/video_stream3')
def video_stream3():
    return Response(detectorDeMovimiento(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#########################################################################
@app.route('/video_stream4')
def video_stream4():
    return Response(detectorDeMovimiento(gamma=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#########################################################################
@app.route('/video_stream5')
def video_stream5():
    return Response(detectorDeMovimiento(color=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
#########################################################################
@app.route('/video_stream6')
def video_stream6():
    return Response(detectorDeMovimiento(clahe=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#########################################################################
    
@app.route('/video_stream7')
def video_stream7():
    return Response(detectorDeMovimiento(equa=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_stream8')
def video_stream8():
    return Response(detectorDeMovimiento(ruido=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream9')
def video_stream9():
    return Response(parte2(erosion=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream10')
def video_stream10():
    return Response(parte2(dilatacion=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream11')
def video_stream11():
    return Response(parte2(tophat=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream12')
def video_stream12():
    return Response(parte2(blackhat=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream13')
def video_stream13():
    return Response(parte2(combinado=True),
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
