import os
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime, timedelta
import csv
import json

# --- FUNÇÃO PARA CARREGAR CONFIGURAÇÃO ---
def load_config(config_path='config.json'):
    # ... (esta função permanece idêntica)
    default_config = {
        "CAMERA_RTSP_URL": "rtsp://usuario:senha@ip_da_camera/stream",
        "MODO_OPERACAO": {"DEBUG_MODE": True, "USE_ADVANCED_PLATE_FINDER": False},
        "PARAMETROS_PERFORMANCE": {"FRAME_SKIP": 5, "FRAME_WIDTH": 800},
        "PARAMETROS_DETECCAO": {"CONFIDENCE_THRESHOLD": 0.5, "OCR_CONFIDENCE_THRESHOLD": 0.4, "COOLDOWN_SEGUNDOS": 10},
        "PARAMETROS_DETECTOR_AVANCADO": {"GAUSSIAN_BLUR_KERNEL": [5, 5], "MIN_ASPECT_RATIO": 2.0, "MAX_ASPECT_RATIO": 4.5, "MIN_PLATE_WIDTH": 60, "MIN_PLATE_HEIGHT": 15}
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            print(f"[INFO] Arquivo '{config_path}' carregado com sucesso.")
            return json.load(f)
    except FileNotFoundError:
        print(f"[AVISO] Arquivo '{config_path}' não encontrado. Criando um arquivo padrão.")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        print(f"[ERRO] Por favor, edite o arquivo '{config_path}' com suas configurações e rode o script novamente.")
        return None
    except json.JSONDecodeError:
        print(f"[ERRO] O arquivo '{config_path}' contém um erro de sintaxe JSON e não pôde ser lido.")
        return None

# --- INICIALIZAÇÃO DE RECURSOS ---
config = load_config()
if config is None:
    exit()

PROTOTXT = "deploy.prototxt"
MODEL = "mobilenet_iter_73000.caffemodel"
CAPTURAS_DIR = "capturas"
CSV_PATH = os.path.join("placas", "placas.csv")

os.makedirs(CAPTURAS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

print("[INFO] Carregando modelo de detecção de objetos...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# ==============================================================================
# --- CORREÇÃO ESTÁ AQUI ---
# A lista de classes PRECISA ter todos os 21 itens com os quais o modelo foi treinado, na ordem correta.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# ==============================================================================

print("[INFO] Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'], gpu=False)
print("[INFO] EasyOCR pronto!")

# (O resto do código permanece exatamente o mesmo)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["imagem", "placa", "data_hora"])

placas_recentes = {}

def is_valid_plate_format(text):
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned_text) != 7:
        return False, None
    if re.match(r"^[A-Z]{3}[0-9]{4}$", cleaned_text) or re.match(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$", cleaned_text):
        return True, cleaned_text
    return False, None

def find_plate_candidates_advanced(vehicle_img, params):
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    blur_kernel = tuple(params["GAUSSIAN_BLUR_KERNEL"])
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(blurred, 50, 200)

    if config["MODO_OPERACAO"]["DEBUG_MODE"]:
        cv2.imshow("Debug - Edges", edges)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        if params["MIN_ASPECT_RATIO"] < aspect_ratio < params["MAX_ASPECT_RATIO"] and w > params["MIN_PLATE_WIDTH"] and h > params["MIN_PLATE_HEIGHT"]:
            plate_crop = vehicle_img[y:y+h, x:x+w]
            return plate_crop
            
    return None

def save_plate_to_csv(image_name, plate_text, timestamp):
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([image_name, plate_text, timestamp])

cap = cv2.VideoCapture(config["CAMERA_RTSP_URL"])
if not cap.isOpened():
    print(f"[ERRO] Não foi possível abrir o stream da câmera: {config['CAMERA_RTSP_URL']}")
    exit()

print("[INFO] Processando vídeo... Pressione 'q' para sair.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[AVISO] Stream finalizado ou quadro perdido.")
        break
    
    frame_count += 1
    if frame_count % config["PARAMETROS_PERFORMANCE"]["FRAME_SKIP"] != 0:
        continue

    h, w = frame.shape[:2]
    r = config["PARAMETROS_PERFORMANCE"]["FRAME_WIDTH"] / float(w)
    frame = cv2.resize(frame, (config["PARAMETROS_PERFORMANCE"]["FRAME_WIDTH"], int(h * r)), interpolation=cv2.INTER_AREA)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    agora = datetime.now()
    cooldown = config["PARAMETROS_DETECCAO"]["COOLDOWN_SEGUNDOS"]
    placas_a_remover = [p for p, t in placas_recentes.items() if (agora - t) > timedelta(seconds=cooldown)]
    for p in placas_a_remover:
        del placas_recentes[p]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        # Esta linha agora funcionará corretamente, pois a lista CLASSES tem o tamanho certo.
        if confidence > config["PARAMETROS_DETECCAO"]["CONFIDENCE_THRESHOLD"] and CLASSES[idx] in ["car", "bus", "motorbike"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            vehicle_img = frame[startY:endY, startX:endX]
            if vehicle_img.size == 0: continue

            if config["MODO_OPERACAO"]["DEBUG_MODE"]:
                cv2.imshow("Debug - Vehicle Crop", vehicle_img)

            image_to_ocr = None
            if config["MODO_OPERACAO"]["USE_ADVANCED_PLATE_FINDER"]:
                image_to_ocr = find_plate_candidates_advanced(vehicle_img, config["PARAMETROS_DETECTOR_AVANCADO"])
            else:
                image_to_ocr = vehicle_img
            
            if image_to_ocr is not None:
                if config["MODO_OPERACAO"]["DEBUG_MODE"]:
                    cv2.imshow("Debug - Image Sent to OCR", image_to_ocr)
                
                ocr_results = reader.readtext(image_to_ocr, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                for (bbox, text, prob) in ocr_results:
                    if prob > config["PARAMETROS_DETECCAO"]["OCR_CONFIDENCE_THRESHOLD"]:
                        is_valid, plate_text = is_valid_plate_format(text)
                        
                        if is_valid and plate_text not in placas_recentes:
                            placas_recentes[plate_text] = agora
                            
                            timestamp_obj = datetime.now()
                            timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
                            image_name = f"placa_{timestamp_obj.strftime('%Y%m%d_%H%M%S')}.png"
                            filepath = os.path.join(CAPTURAS_DIR, image_name)
                            
                            cv2.imwrite(filepath, vehicle_img)
                            save_plate_to_csv(image_name, plate_text, timestamp_str)
                            
                            print(f"[SUCESSO] Placa: {plate_text} | Conf: {prob:.2f} | Arquivo: {image_name}")

                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            cv2.putText(frame, plate_text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            break
    
    cv2.imshow("Detecção de Placas - Pressione 'q' para sair", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Finalizando...")
cap.release()
cv2.destroyAllWindows()
