# Muestra el video con las detecciones y guarda las métricas en un archivo de texto.

import os
import time
import psutil
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2

videos_dir = "./cardumen_mediapipe/Data"
resultados_dir = "./cardumen_mediapipe/resultados"
modelo_mediapipe = "./cardumen_mediapipe/Models/efficientdet_lite0_pf16.tflite" # Ruta al modelo de MediaPipe

# Crear carpeta de resultados si no existe
os.makedirs(resultados_dir, exist_ok=True)

# Agregar timestamp al nombre del archivo
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(resultados_dir, f"metrics_results_mediapipe_{timestamp}.txt")

video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Abrir archivo en modo append para asegurar que cada escritura se mantenga
with open(output_file, "w", encoding="utf-8") as out_f:
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"\nIniciando procesamiento de: {video_file}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        total_person_score = 0
        total_person_detections = 0
        tiempo_inferencia_total = 0
        tiempo_procesamiento_total = 0

        current_process = psutil.Process(os.getpid())
        current_process.cpu_percent(interval=None)

        # Iniciar cronómetros
        start_cpu_time = time.process_time()
        start_wall_time = time.time()
        
        # Importante: Crear un nuevo detector para cada video para evitar problemas con los timestamps
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=modelo_mediapipe),
            max_results=10,
            score_threshold=0.3,
            running_mode=vision.RunningMode.VIDEO
        )
        detector = vision.ObjectDetector.create_from_options(options)
        
        # Usaremos el frame index como timestamp - esto garantiza incremento monótono
        for frame_index in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Usar el índice de frame multiplicado por 1000 para asegurar incremento consistente
            timestamp_ms = frame_index * 1000

            tiempo_inicio_inferencia = time.time()
            result = detector.detect_for_video(mp_image, timestamp_ms)
            tiempo_fin_inferencia = time.time()

            tiempo_inferencia = tiempo_fin_inferencia - tiempo_inicio_inferencia
            tiempo_inferencia_total += tiempo_inferencia
            tiempo_procesamiento_total += tiempo_inferencia

            # Dibujar las detecciones en el frame actual
            for detection in result.detections:
                categoria = detection.categories[0]
                if categoria.category_name.lower() == "person":
                    total_person_score += categoria.score
                    total_person_detections += 1
                
                # Dibujar el rectángulo y etiqueta para cada detección
                bbox = detection.bounding_box
                bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                
                # Rectángulo para la detección
                cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (100, 255, 0), 2)
                
                # Barra superior para el texto
                cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), (100, 255, 0), -1)
                
                # Texto con nombre y porcentaje
                label = f"{categoria.category_name} {categoria.score*100:.2f}%"
                cv2.putText(frame, label, (bbox_x + 5, bbox_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f"Procesando {video_file} frame {frame_index+1}/{frame_count}", end='\r')
            
            # Mostrar el frame con las detecciones
            cv2.imshow('Video', frame)
            
            # Salir al presionar ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Finalizar cronómetros
        end_cpu_time = time.process_time()
        end_wall_time = time.time()

        cpu_usage = current_process.cpu_percent(interval=None)
        cpu_time_used = end_cpu_time - start_cpu_time
        wall_time_elapsed = end_wall_time - start_wall_time

        average_score = total_person_score / total_person_detections if total_person_detections > 0 else 0

        # Guardar métricas
        out_f.write(f"Video: {video_file}\n")
        out_f.write(f"Uso de CPU: {cpu_usage/psutil.cpu_count():.4f}%\n")
        out_f.write(f"CPU Time Used: {cpu_time_used:.4f} seconds\n")
        out_f.write(f"Wall-Clock Time Elapsed: {wall_time_elapsed:.4f} seconds\n")
        out_f.write(f"Confianza promedio (solo de personas): {average_score:.4f}\n")
        out_f.write(f"Tiempo medio de inferencia: {tiempo_inferencia_total/max(frame_count,1):.4f}\n")
        out_f.write(f"Tiempo medio de procesamiento (solo inferencia en MediaPipe): {tiempo_procesamiento_total/max(frame_count,1):.4f}\n")
        out_f.write("-" * 40 + "\n")
        
        # Forzar la escritura inmediata al archivo
        out_f.flush()
        
        cap.release()
        print(f"\nCompletado: {video_file}")

# Cerrar todas las ventanas de OpenCV al finalizar
cv2.destroyAllWindows()
print(f"\nTodos los resultados guardados en {output_file}")