import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path='/home/pi/my_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Clases de ropa
class_names = ['Oscura', 'Blanca', 'Colores']

# Iniciar cámara
cam = cv2.VideoCapture(0)
cv2.namedWindow("Clasificador de Ropa")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

    cv2.imshow("Clasificador de Ropa", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC para salir
        print("Cerrando...")
        break
    elif k % 256 == 32:  # ESPACIO para tomar foto
        img = frame.copy()
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_input = np.reshape(img_normalized, [1, 224, 224, 3]).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_idx = np.argmax(prediction)
        pred_class = class_names[class_idx]
        confidence = np.max(prediction)

        print(f"Predicción: {pred_class} ({confidence*100:.1f}%)")

        # Mostrar la predicción en la imagen
        color = (0, 255, 0) if pred_class == 'Blanca' else (255, 0, 0) if pred_class == 'Oscura' else (0, 165, 255)
        cv2.putText(img, f"{pred_class} ({confidence*100:.1f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Mostrar imagen en matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Ropa: {pred_class}")
        plt.axis('off')
        plt.show()

cam.release()
cv2.destroyAllWindows()
