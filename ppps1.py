import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO

# Configuración de pines GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pines para los servos
servo_blanca_pin = 17
servo_oscura_pin = 27
servo_colores_pin = 22

# Configurar pines como salida PWM a 50Hz (para servos)
GPIO.setup(servo_blanca_pin, GPIO.OUT)
GPIO.setup(servo_oscura_pin, GPIO.OUT)
GPIO.setup(servo_colores_pin, GPIO.OUT)

servo_blanca = GPIO.PWM(servo_blanca_pin, 50)
servo_oscura = GPIO.PWM(servo_oscura_pin, 50)
servo_colores = GPIO.PWM(servo_colores_pin, 50)

# Iniciar PWM con duty cycle neutro (0)
servo_blanca.start(0)
servo_oscura.start(0)
servo_colores.start(0)

# Función para mover el servo (ej. abrir y cerrar tapa)
def mover_servo(servo):
    # 7.5% duty ≈ posición neutra, 5% ≈ cerrado, 10% ≈ abierto
    servo.ChangeDutyCycle(10)  # Abrir
    time.sleep(1.5)
    servo.ChangeDutyCycle(5)   # Cerrar
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)   # Apagar para evitar zumbidos

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path='/home/pi/my_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Clases
class_names = ['Oscura', 'Blanca', 'Colores']

# Iniciar cámara
cam = cv2.VideoCapture(0)
cv2.namedWindow("Clasificador de Ropa")

def activar_servo(clase):
    print(f"Activando servo para {clase}")
    if clase == "Blanca":
        mover_servo(servo_blanca)
    elif clase == "Oscura":
        mover_servo(servo_oscura)
    elif clase == "Colores":
        mover_servo(servo_colores)

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

        activar_servo(pred_class)

        color = (0, 255, 0) if pred_class == 'Blanca' else (255, 0, 0) if pred_class == 'Oscura' else (0, 165, 255)
        cv2.putText(img, f"{pred_class} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Ropa: {pred_class}")
        plt.axis('off')
        plt.show()

# Cleanup
servo_blanca.stop()
servo_oscura.stop()
servo_colores.stop()
GPIO.cleanup()
cam.release()
cv2.destroyAllWindows()
