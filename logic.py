import os
import cv2
import numpy as np
import easyocr
import re
import csv
import json
import Levenshtein
from datetime import datetime, timedelta

def load_config(config_path='config.json'):
    default_config = {
        "CAMERA_RTSP_URL": "rtsp://usuario:senha@ip_da_camera/stream",
        "MODO_OPERACAO": {"DEBUG_MODE": True, "USE_ADVANCED_PLATE_FINDER": False},
        "PARAMETROS_AUTORIZACAO": {"ARQUIVO_SERVIDORES_CSV": "servidores.csv", "TOLERANCIA_MATCH": 1},
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
        print(f"[ERRO] Por favor, edite o arquivo '{config_path}' com suas configurações.")
        return None
    except json.JSONDecodeError:
        print(f"[ERRO] O arquivo '{config_path}' contém um erro de sintaxe JSON.")
        return None

def load_authorized_vehicles(csv_path):
    if not os.path.exists(csv_path):
        print(f"[ERRO] Arquivo de servidores '{csv_path}' não encontrado.")
        return {}
    vehicles = {}
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';') 
            for row in reader:
                if 'PLACA' in row and 'SERVIDOR' in row:
                    plate = re.sub(r'[^A-Z0-9]', '', row['PLACA'].upper())
                    server_info = row['SERVIDOR'].strip()
                    if plate:
                        vehicles[plate] = server_info
        print(f"[INFO] {len(vehicles)} veículos autorizados carregados de '{csv_path}'.")
    except Exception as e:
        print(f"[ERRO] Falha ao ler o arquivo CSV: {e}")
        return {}
    return vehicles

def find_match_in_whitelist(detected_plate, authorized_vehicles, tolerance):
    for authorized_plate, server_info in authorized_vehicles.items():
        distance = Levenshtein.distance(detected_plate, authorized_plate)
        if distance <= tolerance:
            return {"servidor": server_info, "placa_autorizada": authorized_plate}
    return None

def trigger_release_action(detected_plate, match_info):
    print("----------------------------------------------------")
    print(f"[LIBERADO] Veículo autorizado detectado!")
    print(f"  > Servidor: {match_info['servidor']}")
    print(f"  > Placa na Lista: {match_info['placa_autorizada']}")
    print(f"  > Placa Detectada: {detected_plate}")
    print("----------------------------------------------------")

def is_valid_plate_format(text):
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned_text) != 7: return False, None
    if re.match(r"^[A-Z]{3}[0-9]{4}$", cleaned_text) or re.match(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$", cleaned_text):
        return True, cleaned_text
    return False, None

def find_plate_candidates_advanced(vehicle_img, params, config):
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    blur_kernel = tuple(params["GAUSSIAN_BLUR_KERNEL"])
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Aplicar limiar Canny com parâmetros ajustáveis
    edges = cv2.Canny(blurred, params.get("CANNY_LOW", 50), params.get("CANNY_HIGH", 200))
    
    # Operação morfológica para fechar buracos
    kernel_size = params.get("MORPH_KERNEL_SIZE", 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    if config["MODO_OPERACAO"]["DEBUG_MODE"]: 
        cv2.imshow("Debug - Edges", edges)
        cv2.imshow("Debug - Closed", closed)
        
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if params["MIN_ASPECT_RATIO"] < aspect_ratio < params["MAX_ASPECT_RATIO"] and w > params["MIN_PLATE_WIDTH"] and h > params["MIN_PLATE_HEIGHT"]:
            return vehicle_img[y:y+h, x:x+w]
    return None

def save_plate_to_csv(image_name, plate_text, timestamp, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "placas.csv"), mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([image_name, plate_text, timestamp])
