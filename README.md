# clasificacion-de-ropa
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-opencv libatlas-base-dev -y
pip3 install numpy matplotlib
pip3 install tflite-runtime


sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev -S
sudo apt-get install -y build-essential cmake gfortran libjpeg-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev libblas-dev liblapack-dev 
sudo apt-get install python3-venv
python3 -m venv tf-env
source tf-env/bin/activate
pip install --upgrade pip
pip install tensorflow
pip install opencv-python
pip install matplotlib


python3 clasificador_servos.py
sudo apt install python3-rpi.gpio
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-opencv libatlas-base-dev -y
pip3 install numpy matplotlib
pip3 install tflite-runtime


sudo raspi-config
# Interfaces > Camera > Enable
nano clasificador_servos.py
import numpy as np
import cv2
import time
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO

# Pines GPIO para los servos
servo_pins = [17, 27, 22]  # Blanca, Colores, Oscura
GPIO.setmode(GPIO.BCM)
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)

servos = [GPIO.PWM(pin, 50) for pin in servo_pins]
for servo in servos:
    servo.start(0)

def mover_servo(index):
    print(f"Moviendo servo {index}...")
    servos[index].ChangeDutyCycle(7.5)
    time.sleep(1)
    servos[index].ChangeDutyCycle(2.5)
    time.sleep(0.5)
    servos[index].ChangeDutyCycle(0)

# Cargar modelo tflite
interpreter = tflite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Capturar imagen
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ No se pudo capturar imagen.")
    GPIO.cleanup()
    exit()

# Preprocesamiento
img = cv2.resize(frame, (224, 224)).astype(np.float32)
img = np.expand_dims(img, axis=0) / 255.0

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

class_names = ['Blanca', 'Colores', 'Oscura']
pred_index = int(np.argmax(output))
print(f"✅ Predicción: {class_names[pred_index]}")

mover_servo(pred_index)

# Limpieza
for servo in servos:
    servo.stop()
GPIO.cleanup()

sudo apt install python3-gpiozero

sudo systemctl start pigpiod
sudo systemctl enable pigpiod
