import json
import cv2
import numpy as np
import easyocr
import Levenshtein
import csv
import re
from flask import Flask, render_template, Response, request
import os
import base64
from datetime import datetime
app = Flask(__name__)

# Configuração e inicialização de modelos
def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

# Carregar veículos autorizados
def load_authorized_vehicles(csv_path):
    vehicles = {}
    if not os.path.exists(csv_path):
        print(f"[AVISO] Arquivo '{csv_path}' não encontrado.")
        return {}
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                placa_raw = row.get('PLACA')
                servidor_raw = row.get('SERVIDOR')
                
                if placa_raw and servidor_raw:
                    plate = re.sub(r'[^A-Z0-9]', '', placa_raw.upper())
                    server_info = servidor_raw.strip()
                    
                    if plate and server_info:
                        vehicles[plate] = server_info
    except Exception as e:
        print(f"[ERRO] Falha ao ler CSV: {e}")
    return vehicles

authorized_vehicles = load_authorized_vehicles(config["PARAMETROS_AUTORIZACAO"]["ARQUIVO_SERVIDORES_CSV"])

# Carregar modelos globalmente
print("[INFO] Carregando modelos de IA...")
net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/mobilenet_iter_73000.caffemodel")
reader = easyocr.Reader(['pt'], gpu=False)
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

# Variável para controlar a detecção atual para notificação no frontend
last_detection = {"status": None, "nome": None, "placa": None, "timestamp": None}

@app.route('/get_detection')
def get_detection():
    return json.dumps(last_detection), 200

# Variável para controlar a fonte de vídeo atual
current_source = {'type': 'rtsp', 'url': config.get('CAMERA_RTSP_URL', '')}

@app.route('/update_config', methods=['POST'])
def update_config():
    global current_source
    data = request.json
    source_type = data.get('sourceType', 'usb')
    url = data.get('rtspUrl', '')
    
    # Trava: se for RTSP e a URL estiver vazia, ignora a mudança
    if source_type == 'rtsp' and not url.strip():
        print("[AVISO] Tentativa de configurar RTSP com URL vazia. Mudança ignorada.")
        return {"status": "error", "message": "URL vazia"}, 400
        
    current_source['type'] = source_type
    current_source['url'] = url
    return {"status": "success"}, 200

def is_valid_plate_format(text):
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned_text) != 7: return False, None
    if re.match(r"^[A-Z]{3}[0-9]{4}$", cleaned_text) or re.match(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$", cleaned_text):
        return True, cleaned_text
    return False, None

def find_plate_candidates_advanced(vehicle_img, params):
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    blur_kernel = tuple(params["GAUSSIAN_BLUR_KERNEL"])
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, 50, 200)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if params["MIN_ASPECT_RATIO"] < aspect_ratio < params["MAX_ASPECT_RATIO"] and w > params["MIN_PLATE_WIDTH"] and h > params["MIN_PLATE_HEIGHT"]:
            return vehicle_img[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def process_frame(frame):
    h, w = frame.shape[:2]
    scale = config["PARAMETROS_PERFORMANCE"].get("FRAME_WIDTH", 800) / float(w)
    frame_resized = cv2.resize(frame, (config["PARAMETROS_PERFORMANCE"].get("FRAME_WIDTH", 800), int(h * scale)))
    (h_r, w_r) = frame_resized.shape[:2]

    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > config["PARAMETROS_DETECCAO"].get("CONFIDENCE_THRESHOLD", 0.5):
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["car", "bus", "motorbike"]:
                box = detections[0, 0, i, 3:7] * np.array([w_r, h_r, w_r, h_r])
                (startX, startY, endX, endY) = box.astype("int")
                
                vehicle_img = frame_resized[startY:endY, startX:endX]
                if vehicle_img.size > 0:
                    plate_crop, plate_coords = None, None
                    if config["MODO_OPERACAO"].get("USE_ADVANCED_PLATE_FINDER", False):
                        plate_crop, plate_coords = find_plate_candidates_advanced(vehicle_img, config["PARAMETROS_DETECTOR_AVANCADO"])
                    else:
                        plate_crop = vehicle_img
                    
                    if plate_crop is not None:
                        ocr_results = reader.readtext(plate_crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        print(f"[DEBUG] OCR bruto: {ocr_results}")
                        
                        for (bbox, text, prob) in ocr_results:
                            if prob > config["PARAMETROS_DETECCAO"].get("OCR_CONFIDENCE_THRESHOLD", 0.4):
                                is_valid, plate_text = is_valid_plate_format(text)
                                if is_valid:
                                    # Verificar autorização
                                    match_info = None
                                    for auth_plate, server_info in authorized_vehicles.items():
                                        if Levenshtein.distance(plate_text, auth_plate) <= config["PARAMETROS_AUTORIZACAO"].get("TOLERANCIA_MATCH", 1):
                                            match_info = server_info
                                            break
                                    
                                    # Feedback Visual
                                    if match_info:
                                        last_detection.update({"status": "autorizado", "nome": match_info, "placa": plate_text, "timestamp": datetime.now().isoformat()})
                                        color = (0, 255, 0)
                                    else:
                                        last_detection.update({"status": "nao_autorizado", "nome": "Nao reconhecido", "placa": plate_text, "timestamp": datetime.now().isoformat()})
                                        color = (0, 0, 255)
                                    
                                    # Desenha caixa do carro
                                    cv2.rectangle(frame_resized, (startX, startY), (endX, endY), color, 2)
                                    # Removido cv2.putText para limpeza do frame
                                    
                                    # Desenha caixa da placa se encontrada
                                    if plate_coords:
                                        px, py, pw, ph = plate_coords
                                        abs_startX = startX + px
                                        abs_startY = startY + py
                                        cv2.rectangle(frame_resized, (abs_startX, abs_startY), (abs_startX + pw, abs_startY + ph), (0, 255, 255), 2)
                                    break
    return frame_resized


@app.route('/test_image', methods=['POST'])
def test_image():
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    processed_frame = process_frame(frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return {"image": base64_img}, 200

def generate_frames():
    global current_source
    cap = None
    if current_source['type'] == 'usb':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(current_source['url'])
        
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            if current_source['type'] == 'usb':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(current_source['url'])
            continue
        
        frame_count += 1
        if frame_count % config["PARAMETROS_PERFORMANCE"].get("FRAME_SKIP", 5) == 0:
            frame = process_frame(frame)
        
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
