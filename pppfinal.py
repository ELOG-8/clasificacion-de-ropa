import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import time

# ==== CONFIGURACIÓN GPIO ====
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

servo_pins = {
    'Blanca': 17,
    'Oscura': 27,
    'Colores': 22
}

servos = {}

# Configurar pines y objetos PWM
for key, pin in servo_pins.items():
    GPIO.setup(pin, GPIO.OUT)
    servo = GPIO.PWM(pin, 50)  # 50 Hz
    servo.start(0)
    servos[key] = servo

def mover_servo(clase):
    servo = servos[clase]
    servo.ChangeDutyCycle(7.5)  # Posición abierta (~90°)
    time.sleep(1)
    servo.ChangeDutyCycle(2.5)  # Posición cerrada (~0°)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)    # Detener señal

# ==== MODELO TFLITE ====
interpreter = tf.lite.Interpreter(model_path='/home/elog/my_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Clases
class_names = ['Blanca', 'Colores', 'Oscura']

# ==== CÁMARA ====
cam = cv2.VideoCapture(0)
cv2.namedWindow("Clasificador de Ropa")

while True:
    ret, frame = cam.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

    cv2.imshow("Clasificador de Ropa", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC
        print("Cerrando...")
        break
    elif k % 256 == 32:  # ESPACIO
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

        if pred_class in servos:
            mover_servo(pred_class)

        # Mostrar predicción
        color = (0, 255, 0) if pred_class == 'Blanca' else (255, 0, 0) if pred_class == 'Oscura' else (0, 165, 255)
        cv2.putText(img, f"{pred_class} ({confidence*100:.1f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar imagen
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Ropa: {pred_class}")
        plt.axis('off')
        plt.show()

# ==== LIMPIEZA ====
cam.release()
cv2.destroyAllWindows()
for servo in servos.values():
    servo.stop()
GPIO.cleanup()
