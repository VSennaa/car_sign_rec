import json
import cv2
import numpy as np
import easyocr
import Levenshtein
import csv
import re
from flask import Flask, render_template, Response
import os

app = Flask(__name__)

# Configuração e inicialização de modelos
def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

# Carregar veículos autorizados
def load_authorized_vehicles(csv_path):
    vehicles = {}
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if 'PLACA' in row and 'SERVIDOR' in row:
                    plate = re.sub(r'[^A-Z0-9]', '', row['PLACA'].upper())
                    vehicles[plate] = row['SERVIDOR'].strip()
    except Exception as e:
        print(f"[ERRO] Falha ao ler CSV: {e}")
    return vehicles

authorized_vehicles = load_authorized_vehicles(config["PARAMETROS_AUTORIZACAO"]["ARQUIVO_SERVIDORES_CSV"])

# Carregar modelos
print("[INFO] Carregando modelos...")
net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/mobilenet_iter_73000.caffemodel")
reader = easyocr.Reader(['pt'], gpu=False)
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

def is_valid_plate_format(text):
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned_text) == 7 and (re.match(r"^[A-Z]{3}[0-9]{4}$", cleaned_text) or re.match(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$", cleaned_text)):
        return True, cleaned_text
    return False, None

# Variável para controlar a fonte de vídeo atual
current_source = {'type': 'usb', 'url': config.get('CAMERA_RTSP_URL', '')}

@app.route('/update_config', methods=['POST'])
def update_config():
    from flask import request
    data = request.json
    current_source['type'] = data.get('sourceType', 'usb')
    current_source['url'] = data.get('rtspUrl', '')
    return {"status": "success"}, 200

def generate_frames():
    # Inicializa baseado na configuração atual
    cap = None
    if current_source['type'] == 'usb':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(current_source['url'])
        
    frame_count = 0
    
    while True:
        # Se a fonte mudou, precisa reiniciar a captura
        # (Para simplificar, isso será feito se o vídeo parar)
        success, frame = cap.read()
        if not success:
            # Tentar reconectar ou mudar fonte
            cap.release()
            if current_source['type'] == 'usb':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(current_source['url'])
            continue
        
        frame_count += 1
        # Processamento apenas em alguns frames
        if frame_count % config["PARAMETROS_PERFORMANCE"].get("FRAME_SKIP", 5) == 0:
            h, w = frame.shape[:2]
            r = config["PARAMETROS_PERFORMANCE"].get("FRAME_WIDTH", 800) / float(w)
            # Redimensionamento
            frame_proc = cv2.resize(frame, (config["PARAMETROS_PERFORMANCE"].get("FRAME_WIDTH", 800), int(h * r)))
            (h_proc, w_proc) = frame_proc.shape[:2]

            blob = cv2.dnn.blobFromImage(frame_proc, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > config["PARAMETROS_DETECCAO"].get("CONFIDENCE_THRESHOLD", 0.5):
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] in ["car", "bus", "motorbike"]:
                        box = detections[0, 0, i, 3:7] * np.array([w_proc, h_proc, w_proc, h_proc])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Extrair região de interesse (veículo)
                        vehicle_img = frame_proc[startY:endY, startX:endX]
                        if vehicle_img.size > 0:
                            ocr_results = reader.readtext(vehicle_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            
                            match_info = None
                            found_plate = None
                            
                            for (bbox, text, prob) in ocr_results:
                                if prob > config["PARAMETROS_DETECCAO"].get("OCR_CONFIDENCE_THRESHOLD", 0.4):
                                    is_valid, plate_text = is_valid_plate_format(text)
                                    if is_valid:
                                        # Verificar autorização
                                        for auth_plate, server_info in authorized_vehicles.items():
                                            if Levenshtein.distance(plate_text, auth_plate) <= config["PARAMETROS_AUTORIZACAO"].get("TOLERANCIA_MATCH", 1):
                                                match_info = server_info
                                                found_plate = plate_text
                                                break
                                        
                                        # Desenhar na imagem original redimensionada
                                        color = (0, 255, 0) if match_info else (0, 0, 255)
                                        label = f"Servidor {match_info}" if match_info else "Nao reconhecido"
                                        
                                        cv2.rectangle(frame_proc, (startX, startY), (endX, endY), color, 2)
                                        cv2.putText(frame_proc, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                        break
            frame = frame_proc # Substitui pelo frame processado
        
        # Codifica o frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
