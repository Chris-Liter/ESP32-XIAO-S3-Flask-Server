# # Author: vlarobbyk
# # Version: 1.0
# # Date: 2024-10-20
# # Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.

# from flask import Flask, render_template, Response, request, jsonify
# from io import BytesIO
# import cv2
# import numpy as np
# import requests

# import time

# import random 
# app = Flask(__name__)

# # Configuración de la URL de la cámara
# # _URL = 'http://192.168.61.97'
# _URL = 'http://192.168.10.28'
# _PORT = '81'
# _STREAM_ROUTE = '/stream'
# _SEP = ':'
# stream_url = f"{_URL}{_SEP}{_PORT}{_STREAM_ROUTE}"

# # Parámetros para procesamiento de video
# MAX_FRAMES = 1000
# N = 2
# THRESH = 60
# ASSIGN_VALUE = 255

# current_kernel_size = 1

# # Función de captura de video desde la cámara
# def video_capture():
#     res = requests.get(stream_url, stream=True)
#     for chunk in res.iter_content(chunk_size=100000):

#         if len(chunk) > 100:
#             try:
#                 img_data = BytesIO(chunk)
#                 cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
#                 gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#                 N = 537
#                 height, width = gray.shape
#                 noise = np.full((height, width), 0, dtype=np.uint8)
#                 random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))
                
#                 noise[random_positions[0], random_positions[1]] = 255

#                 noise_image = cv2.bitwise_or(gray, noise)

#                 total_image = np.zeros((height, width * 2), dtype=np.uint8)
#                 total_image[:, :width] = gray
#                 total_image[:, width:] = noise_image

#                 (flag, encodedImage) = cv2.imencode(".jpg", total_image)
#                 if not flag:
#                     continue

#                 yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
#                 bytearray(encodedImage) + b'\r\n')

#             except Exception as e:
#                 print(e)
#                 continue
            
          
# def reducir_ruido(i, frame):
#     kernel = np.ones((i, i), np.uint8)
#     imagen = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel=kernel)
#     return imagen


# @app.route('/set_kernel_size', methods=['POST'])
# def set_kernel_size():
#     global current_kernel_size
#     data = request.get_json()
#     current_kernel_size = int(data.get('kernelSize'))  # Actualiza el tamaño del kernel
#     return jsonify({"status": "success", "kernelSize": current_kernel_size})

# @app.route('/set_noise_level', methods=['POST'])
# def set_noise_level():
#     global noise_level
#     data = request.get_json()
#     noise_level = float(data.get('noiseLevel')) / 10000  # Ajusta el nivel de ruido
#     return jsonify({"status": "success", "noiseLevel": noise_level})
            
# def sp_noise(image, prob):
#     output = np.zeros(image.shape, np.uint8)
#     thres = 1-prob
#     for i in range (image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output

# # def sp_noise(image, prob):
# #     """
# #     Agrega ruido sal y pimienta a la imagen.
# #     prob: probabilidad de que un píxel se convierta en ruido.
# #     """
# #     output = np.copy(image)
# #     # Genera ruido aleatorio en la imagen
# #     black = np.random.rand(*image.shape[:2]) < prob
# #     white = np.random.rand(*image.shape[:2]) < prob
# #     output[black] = 0
# #     output[white] = 255
# #     return output


# # Generador de frames para detección de movimiento

# ## La ecualizacion del histograma es usado para elevar el contraste de una imagen por medio del uso de histrogramas.
# ## Clahe es una mejora de Ahe, puesto que este opera en pequeñas regiones elevando asi su contraste pero mucho mejor
# ## Mientras que Gamma Correction es usado para ajustar la luminancia de la imagen 
# ##
# def detectorDeMovimiento(bg_subtractor=None, diff_method=False, gamma=None, color=None, clahe=None, equa=None, ruido=None):
#     global current_kernel_size
#     cap = cv2.VideoCapture(stream_url)
#     ret, prev_frame = cap.read()
#     if not ret or prev_frame is None:
#         print("Error al capturar el primer frame.")
#         return

#     prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
#     fps = 0
#     prev_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Selección del método de procesamiento de movimiento
#         if bg_subtractor:
#             fgmask = bg_subtractor.apply(gray)  # Sustracción de fondo
#             combined_img = fgmask
#         elif diff_method:
#             diff_frame = cv2.absdiff(prev_frame, gray)  # Diferencia de fotogramas
#             prev_frame = gray
#             combined_img = diff_frame
#         elif gamma:
#             inv_gamma = 0.5 / gamma
#             table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
#             combined_img = cv2.LUT(gray, table)
#         elif color:
#             combined_img = frame
#         elif clahe:
#             img_clahe = cv2.createCLAHE(clipLimit=3)
#             img = img_clahe.apply(gray) + 30
#             #_, ordinary_img = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
#             combined_img = img
#         elif equa:
#             img_clahe = cv2.equalizeHist(gray)
#             combined_img = img_clahe
        
#         elif ruido:
#             noise_level = current_kernel_size / 100.0  # Valor de ruido basado en el control deslizante
#             noisy_img = sp_noise(gray, noise_level)  # Aplica ruido a la imagen

#             kernel_size = max(1, 3 * (100 - current_kernel_size) // 100)  # Aumenta el kernel a medida que disminuye el ruido
#             kernel = np.ones((kernel_size, kernel_size), np.uint8)
#             combined_img = cv2.morphologyEx(noisy_img, cv2.MORPH_DILATE, kernel)
        
#             # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # noisy_img = sp_noise(frame, 0.05)
#             # filtered_img = cv2.medianBlur(noisy_img, 5)  # Cambia el tamaño del kernel si es necesario

            
#             # kernel = np.ones((current_kernel_size, current_kernel_size), np.uint8)
#             # combined_img = cv2.morphologyEx(filtered_img, cv2.MORPH_DILATE, kernel=kernel)

        
        
#         else:
#             combined_img = gray  # Solo escala de grises

#         # Calcula los FPS
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time

#         # Dibuja el FPS en la imagen
#         cv2.putText(combined_img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

#         # Codifica la imagen para transmitirla
#         _, buffer = cv2.imencode('.jpg', combined_img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# def parte2( erosion=None, dilatacion=None, tophat=None, blackhat=None, combinado=None):
#     elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     if dilatacion:
#             image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#             # Dilatar el frame
#             erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_DILATE, elemento, iterations=3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row1 = np.hstack((image_rgb, erosion_resized))
#             #################################################################################
#             image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

#             # Dilatar el frame
#             erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_DILATE, elemento, iterations=3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row2 = np.hstack((image_rgb2, erosion_resized2))
            
#             image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)

#             # Dilatar el frame
#             erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_DILATE, elemento, iterations=3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row3 = np.hstack((image_rgb3, erosion_resized3))
            
#             combined_img = np.vstack((row1, row2, row3))

            
            
#     elif erosion:
#             image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#             # Dilatar el frame
#             erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_ERODE, elemento, iterations=3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row1 = np.hstack((image_rgb, erosion_resized))
#             #################################################################################
#             image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

#             # Dilatar el frame
#             erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_ERODE, elemento, iterations=3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row2 = np.hstack((image_rgb2, erosion_resized2))
            
#             image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)

#             # Dilatar el frame
#             erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_ERODE, elemento, iterations=3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row3 = np.hstack((image_rgb3, erosion_resized3))
            
#             combined_img = np.vstack((row1, row2, row3))
        
#     elif tophat:
#             image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#             elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

#             # Dilatar el frame
#             erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_TOPHAT, elemento)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row1 = np.hstack((image_rgb, erosion_resized))
#             ###########################################################################
#             image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
#             elemento2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

#             # Dilatar el frame
#             erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_TOPHAT, elemento2)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row2 = np.hstack((image_rgb2, erosion_resized2))
#             ###########################################################################
#             image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
#             elemento3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

#             # Dilatar el frame
#             erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_TOPHAT, elemento3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row3 = np.hstack((image_rgb3, erosion_resized3))
            
#             combined_img = np.vstack((row1, row2, row3))

            
        
#     elif blackhat:
#             image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#             elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

#             # Dilatar el frame
#             erosion = cv2.morphologyEx(image_rgb, cv2.MORPH_BLACKHAT, elemento)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized = cv2.resize(erosion, (image_rgb.shape[1], image_rgb.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row1 = np.hstack((image_rgb, erosion_resized))
#             ###########################################################################
#             image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
#             elemento2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

#             # Dilatar el frame
#             erosion2 = cv2.morphologyEx(image_rgb2, cv2.MORPH_BLACKHAT, elemento2)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized2 = cv2.resize(erosion2, (image_rgb2.shape[1], image_rgb2.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row2 = np.hstack((image_rgb2, erosion_resized2))
#             ###########################################################################
#             image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)
#             # Convertir a RGB
#             image_rgb3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
#             elemento3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

#             # Dilatar el frame
#             erosion3 = cv2.morphologyEx(image_rgb3, cv2.MORPH_BLACKHAT, elemento3)

#             # Redimensionar erosion para que coincida con image_rgb
#             erosion_resized3 = cv2.resize(erosion3, (image_rgb3.shape[1], image_rgb3.shape[0]))

#             # Unir las dos imágenes horizontalmente (original y erosionada)
#             row3 = np.hstack((image_rgb3, erosion_resized3))
            
#             combined_img = np.vstack((row1, row2, row3))
        
#     elif combinado:
#             image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)

#             # Crear un elemento estructurante
#             elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#             # Calcular Top Hat y Black Hat
#             top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, elemento)
#             black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, elemento)

#             # Combinar la imagen original con (Top Hat - Black Hat)
#             comb = cv2.add(image, cv2.subtract(top_hat, black_hat))
#             row1 = np.hstack((image, top_hat, black_hat, comb))
            
#             image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)

#             # Crear un elemento estructurante
#             # Calcular Top Hat y Black Hat
#             top_hat2 = cv2.morphologyEx(image2, cv2.MORPH_TOPHAT, elemento)
#             black_hat2 = cv2.morphologyEx(image2, cv2.MORPH_BLACKHAT, elemento)

#             # Combinar la imagen original con (Top Hat - Black Hat)
#             comb2 = cv2.add(image2, cv2.subtract(top_hat2, black_hat2))
#             row2 = np.hstack((image2, top_hat2, black_hat2, comb2))
            
#             image3 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)

#             # Crear un elemento estructurante
#             # Calcular Top Hat y Black Hat
#             top_hat3 = cv2.morphologyEx(image3, cv2.MORPH_TOPHAT, elemento)
#             black_hat3 = cv2.morphologyEx(image3, cv2.MORPH_BLACKHAT, elemento)

#             # Combinar la imagen original con (Top Hat - Black Hat)
#             comb3 = cv2.add(image3, cv2.subtract(top_hat3, black_hat3))
#             row3 = np.hstack((image3, top_hat3, black_hat3, comb3))
            
#             combined_img = np.vstack((row1, row2, row3))

            
    
#     _, buffer = cv2.imencode('.jpg', combined_img)
#     frame = buffer.tobytes()
#     yield (b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    

# # Rutas para los diferentes modos de visualización
# @app.route('/video_stream1')
# def video_stream1():
#     return Response(detectorDeMovimiento(bg_subtractor=cv2.createBackgroundSubtractorMOG2()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# #########################################################################
# @app.route('/video_stream2')
# def video_stream2():
#     return Response(detectorDeMovimiento(diff_method=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# #########################################################################
# @app.route('/video_stream3')
# def video_stream3():
#     return Response(detectorDeMovimiento(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
    
# #########################################################################
# @app.route('/video_stream4')
# def video_stream4():
#     return Response(detectorDeMovimiento(gamma=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
    
# #########################################################################
# @app.route('/video_stream5')
# def video_stream5():
#     return Response(detectorDeMovimiento(color=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
# #########################################################################
# @app.route('/video_stream6')
# def video_stream6():
#     return Response(detectorDeMovimiento(clahe=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
# #########################################################################
    
# @app.route('/video_stream7')
# def video_stream7():
#     return Response(detectorDeMovimiento(equa=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/video_stream8')
# def video_stream8():
#     return Response(detectorDeMovimiento(ruido=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_stream9')
# def video_stream9():
#     return Response(parte2(erosion=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_stream10')
# def video_stream10():
#     return Response(parte2(dilatacion=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_stream11')
# def video_stream11():
#     return Response(parte2(tophat=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_stream12')
# def video_stream12():
#     return Response(parte2(blackhat=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_stream13')
# def video_stream13():
#     return Response(parte2(combinado=True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # Página principal
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Ruta de video genérica con detector de movimiento
# @app.route('/video_stream')
# def video_stream():
#     return Response(video_capture(),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, Response, stream_with_context, Request, request
from io import BytesIO

import cv2
import numpy as np
import requests
import time


#res = requests.get('http://192.168.221.62:81/stream', stream=True)
res = requests.get('http://192.168.61.97:81/stream', stream=True)



print(res.status_code)  # Debería ser 200 si la conexión es exitosa.



app = Flask(__name__)
# IP Address
#_URL = 'http://192.168.221.62'
_URL = 'http://192.168.61.97'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])
##stream_url = f"http://{_URL}:{_PORT}{_ST}"


# Variables para cálculo de FPS
prev_frame_time = 0
new_frame_time = 0

# Configuración para MOG2
LEARNING_RATE = -1
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variables globales para los niveles de ruido
salt_level = 0.01
pepper_level = 0.01
mask_size = 5

@app.route('/update-settings', methods=['POST'])
def update_settings():
    global salt_level, pepper_level, mask_size
    data = request.json
    salt_level = float(data['saltLevel'])
    pepper_level = float(data['pepperLevel'])
    mask_size = float(data['maskSize'])
    return {'status': 'success'}, 200

def add_salt_and_pepper_noise(image, salt_amount, pepper_amount):
    noisy_image = image.copy()
    salt_num = int(salt_amount * image.size / 100)
    pepper_num = int(pepper_amount * image.size / 100)
    
    # Añade "sal"
    coords_salt = [np.random.randint(0, i - 1, salt_num) for i in image.shape]
    noisy_image[coords_salt[0], coords_salt[1]] = 255

    # Añade "pimienta"
    coords_pepper = [np.random.randint(0, i - 1, pepper_num) for i in image.shape]
    noisy_image[coords_pepper[0], coords_pepper[1]] = 0

    return noisy_image

def apply_filters(gray_with_noise, mask_size):
    # Asegurarse de que el tamaño de la máscara sea impar
    mask_size = int(mask_size)  # Asegurarse de que sea un entero
    if mask_size <= 0:
        mask_size = 3  # Valor por defecto mínimo
    if mask_size % 2 == 0:
        mask_size += 1  # Hacer impar si es par

    # Aplicar filtro de mediana
    median_filtered = cv2.medianBlur(gray_with_noise, mask_size)
    
    # Aplicar filtro de desenfoque (blur)
    blur_filtered = cv2.blur(gray_with_noise, (mask_size, mask_size))
    
    # Aplicar filtro gaussiano
    gaussian_filtered = cv2.GaussianBlur(gray_with_noise, (mask_size, mask_size), 0)

    return median_filtered, blur_filtered, gaussian_filtered


def video_capture():
    res = requests.get(stream_url, stream=True)
    global prev_frame_time
    global new_frame_time
    global salt_level, pepper_level
    global mask_size

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                
                # Imagen original en color
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                # Calcular FPS
                new_frame_time = time.time()
                fps = int(1 / (new_frame_time - prev_frame_time))
                prev_frame_time = new_frame_time
                
                # Crear imagen en escala de grises
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                #ESCALA DE GRISES
                ecualizada = cv2.equalizeHist(gray)

                #CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = clahe.apply(gray)

                #CONTRAST STRETCHING
                # Obtener el mínimo y máximo de la imagen
                minimo = np.min(gray)
                maximo = np.max(gray)

                # Aplicar el estiramiento de contraste
                imagen_stretch = ((gray - minimo) / (maximo - minimo)) * 255.0
                imagen_stretch = np.uint8(imagen_stretch)

                #DETECCION DE MOVIMIENTO
                # Aplicar el sustractor de fondo MOG2 para detección de movimiento
                motion_mask = fgbg.apply(cv_img, LEARNING_RATE)
                background = fgbg.getBackgroundImage()


                # Aplicar el filtro de ruido de sal y pimienta con los valores de los sliders
                gray_with_noise = add_salt_and_pepper_noise(gray, salt_level, pepper_level)

                # Aplicar filtros de suavizado
                median_filtered, blur_filtered, gaussian_filtered = apply_filters(gray_with_noise, mask_size)

                #APLICAR PRIMER ALGORITMO DE DETECCION DE BORDES SOBEL SIN SUAVIZAR
                sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
                sobel_xAbs =cv2.convertScaleAbs(sobel_x)
                sobel_yAbs =cv2.convertScaleAbs(sobel_y)


                sobelFiltered = cv2.addWeighted(sobel_xAbs, 0.5, sobel_yAbs, 0.5, 0)


                # Añadir ruido y procesar imagen para detección de bordes
                N = 537
                height, width = gray.shape
                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))
                noise[random_positions[0], random_positions[1]] = 255
                noise_image = cv2.bitwise_or(gray, noise)
                edge_image = cv2.Canny(gray, 100, 200)

                total_height = height * 2
                total_width = width * 7  # 4 columnas en total

                # Crear un nuevo marco donde se visualizan las imágenes lado a lado
                total_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                total_image[0:height, 0:width] = cv_img  # Imagen original a color
                total_image[0:height, width:width*2] = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)  # Imagen de bordes
                total_image[0:height, width*2:width*3] = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
                total_image[0:height, width*3:width*4] = cv2.cvtColor(ecualizada, cv2.COLOR_GRAY2BGR)
                total_image[0:height, width*4:width*5] = cv2.cvtColor(imagen_stretch, cv2.COLOR_GRAY2BGR)
                total_image[0:height, width*5:width*6] = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                total_image[0:height, width*6:width*7] = cv2.cvtColor(gray_with_noise, cv2.COLOR_GRAY2BGR)

                total_image[height:total_height, 0:width] = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
                total_image[height:total_height, width:width*2] = cv2.cvtColor(blur_filtered, cv2.COLOR_GRAY2BGR)
                total_image[height:total_height, width*2:width*3] = cv2.cvtColor(gaussian_filtered, cv2.COLOR_GRAY2BGR)
                total_image[height:total_height, width*3:width*4] = cv2.cvtColor(sobelFiltered, cv2.COLOR_GRAY2BGR)
                total_image[height:total_height, width*4:width*5] = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)





                # Mostrar FPS en la imagen original
                cv2.putText(total_image[:, :width], f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(total_image[:, width*2:width*3],f"CLAHE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(total_image[:, width*3:width*4],f"Ecualizada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(total_image[:, width*4:width*5],f"Contrast Stretching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


                # Codificar la imagen total a JPEG
                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue





def parte2( erosion=None, dilatacion=None, tophat=None, blackhat=None, combinado=None, todo = None):
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
            
            image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            # Calcular Top Hat y Black Hat
            top_hat3 = cv2.morphologyEx(image3, cv2.MORPH_TOPHAT, elemento)
            black_hat3 = cv2.morphologyEx(image3, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb3 = cv2.add(image3, cv2.subtract(top_hat3, black_hat3))
            row3 = np.hstack((image3, top_hat3, black_hat3, comb3))
            
            combined_img = np.vstack((row1, row2, row3))
    elif todo:
        
            image = cv2.imread('gg (128).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            
            #Calcular erosion y diatacion
            erosion = cv2.morphologyEx(image, cv2.MORPH_ERODE, elemento, iterations=3)
            dilatacion = cv2.morphologyEx(image, cv2.MORPH_DILATE, elemento, iterations=3)

            # Calcular Top Hat y Black Hat
            top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, elemento)
            black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb = cv2.add(image, cv2.subtract(top_hat, black_hat))
            row1 = np.hstack((image, erosion, dilatacion, top_hat, black_hat, comb))
            
            image2 = cv2.imread('gg (131).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            # Calcular Top Hat y Black Hat
            
            erosion2 = cv2.morphologyEx(image2, cv2.MORPH_ERODE, elemento, iterations=3)
            dilatacion2 = cv2.morphologyEx(image2, cv2.MORPH_DILATE, elemento, iterations=3)


            top_hat2 = cv2.morphologyEx(image2, cv2.MORPH_TOPHAT, elemento)
            black_hat2 = cv2.morphologyEx(image2, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb2 = cv2.add(image2, cv2.subtract(top_hat2, black_hat2))
            row2 = np.hstack((image2,erosion2,dilatacion2, top_hat2, black_hat2, comb2))
            
            image3 = cv2.imread('gg (141).jpg', cv2.IMREAD_GRAYSCALE)

            # Crear un elemento estructurante
            
            erosion3 = cv2.morphologyEx(image3, cv2.MORPH_ERODE, elemento, iterations=3)
            dilatacion3 = cv2.morphologyEx(image3, cv2.MORPH_DILATE, elemento, iterations=3)

            
            # Calcular Top Hat y Black Hat
            top_hat3 = cv2.morphologyEx(image3, cv2.MORPH_TOPHAT, elemento)
            black_hat3 = cv2.morphologyEx(image3, cv2.MORPH_BLACKHAT, elemento)

            # Combinar la imagen original con (Top Hat - Black Hat)
            comb3 = cv2.add(image3, cv2.subtract(top_hat3, black_hat3))
            row3 = np.hstack((image3,erosion3,dilatacion3, top_hat3, black_hat3, comb3))
            
            combined_img = np.vstack((row1, row2, row3))




            
    
    _, buffer = cv2.imencode('.jpg', combined_img)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream1")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


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

@app.route('/video_stream14')
def video_stream14():
    return Response(parte2(todo=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)




#if __name__ == "__main__":
#    app.run(host='0.0.0.0', port=5000, debug=True)