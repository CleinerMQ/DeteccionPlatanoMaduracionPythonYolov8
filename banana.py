import cv2
import math
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Función para verificar cámaras disponibles
def check_cameras():
    available_cams = []
    for i in range(10):  # Probar hasta 10 cámaras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()  # Liberar la cámara
    return available_cams

# Cargar el modelo YOLO
model = YOLO("./best.pt")

# Definir las clases según tu archivo data.yaml
classNames = ['Mal estado', 'Apto para chifles', 'No apto para chifles']

# Función para iniciar la cámara
def start_video():
    global cap, is_running
    cam_index = camera_combobox.current()
    cap = cv2.VideoCapture(available_cameras[cam_index])
    is_running = True
    update_frame()

# Función para detener el video
def stop_video():
    global is_running
    is_running = False
    if cap.isOpened():
        cap.release()
    video_label.config(image="")  # Limpiar la pantalla de video

# Función para actualizar el frame de video
def update_frame():
    global cap
    if is_running:
        success, img = cap.read()
        if success:
            # Obtener el valor de confianza del control deslizante
            confidence_threshold = threshold_slider.get() / 100

            # Aplicar detección y dibujar resultados en la imagen
            results = model(img, stream=True, conf=confidence_threshold)
            counts = {'Mal estado': 0, 'Apto para chifles': 0, 'No apto para chifles': 0}

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    # Filtrar solo las clases en classNames
                    if cls < len(classNames):
                        class_name = classNames[cls]
                        counts[class_name] += 1

                        # Definir color para cada clase
                        color = (139, 0, 0) if class_name == "Mal estado" else (0, 128, 0) if class_name == "Apto para chifles" else (0, 102, 204)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f'{class_name} {confidence}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convertir a formato de imagen compatible con tkinter
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            video_label.img_tk = img_tk
            video_label.config(image=img_tk)

            # Actualizar etiquetas de conteo
            label_mal_estado.config(text=f"Mal estado: {counts['Mal estado']}")
            label_apto.config(text=f"Apto para chifles: {counts['Apto para chifles']}")
            label_no_apto.config(text=f"No apto para chifles: {counts['No apto para chifles']}")

        # Continuar actualizando el video
        video_label.after(10, update_frame)

# Configurar la interfaz de usuario con tkinter
root = tk.Tk()
root.title("Detección de Plátanos")
root.geometry("900x700")
root.configure(bg="#f2f2f2")
root.resizable(False, False)  # Deshabilitar maximización de ventana

# Frame para seleccionar la cámara y los botones de control
control_frame = tk.Frame(root, bg="#2E4053")
control_frame.pack(fill="x", padx=10, pady=5)

# Lista de cámaras disponibles
available_cameras = check_cameras()
camera_combobox = ttk.Combobox(control_frame, values=available_cameras, state="readonly", font=("Arial", 12))
camera_combobox.set("Selecciona la cámara")
camera_combobox.pack(side="left", padx=5, pady=5)

# Botones de iniciar y detener
start_button = tk.Button(control_frame, text="Iniciar Video", command=start_video, bg="#5D6D7E", fg="white", font=("Arial", 12, "bold"))
start_button.pack(side="left", padx=5, pady=5)

stop_button = tk.Button(control_frame, text="Detener Video", command=stop_video, bg="#A93226", fg="white", font=("Arial", 12, "bold"))
stop_button.pack(side="left", padx=5, pady=5)

# Control deslizante para seleccionar el umbral de confianza
threshold_slider = tk.Scale(control_frame, from_=0, to=100, orient="horizontal", label="Umbral(%)", bg="#2E4053", fg="white", font=("Arial", 10, "bold"))
threshold_slider.set(50)  # Umbral inicial al 50%
threshold_slider.pack(side="right", padx=5, pady=5)

# Frame para mostrar el video
video_frame = tk.Frame(root, bg="#1C2833")
video_frame.pack(fill="both", expand=True, padx=10, pady=5)

video_label = tk.Label(video_frame)
video_label.pack(fill="both", expand=True)

# Frame para mostrar el conteo de objetos detectados
label_frame = tk.Frame(root, bg="#34495E")
label_frame.pack(fill="x", padx=10, pady=5)

label_mal_estado = tk.Label(label_frame, text="Mal estado: 0", fg="#B03A2E", bg="#34495E", font=("Arial", 12, "bold"))
label_mal_estado.pack(side="left", padx=10)

label_apto = tk.Label(label_frame, text="Apto para chifles: 0", fg="#1E8449", bg="#34495E", font=("Arial", 12, "bold"))
label_apto.pack(side="left", padx=10)

label_no_apto = tk.Label(label_frame, text="No apto para chifles: 0", fg="#2874A6", bg="#34495E", font=("Arial", 12, "bold"))
label_no_apto.pack(side="left", padx=10)

# Variables globales para la cámara y el estado de video
cap = None
is_running = False

# Iniciar la aplicación
root.mainloop()


