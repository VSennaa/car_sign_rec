import os
import cv2
import numpy as np
import easyocr
from datetime import datetime, timedelta
from logic import load_config, load_authorized_vehicles, find_match_in_whitelist, trigger_release_action, is_valid_plate_format, find_plate_candidates_advanced, save_plate_to_csv

config = load_config()
if config is None: exit()

authorized_vehicles = load_authorized_vehicles(config["PARAMETROS_AUTORIZACAO"]["ARQUIVO_SERVIDORES_CSV"])
PROTOTXT, MODEL = "deploy.prototxt", "mobilenet_iter_73000.caffemodel"
CAPTURAS_DIR = "capturas"
placas_recentes = {}

print("[INFO] Carregando modelo de detecção de objetos...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

print("[INFO] Inicializando EasyOCR...")
reader = easyocr.Reader(['pt'], gpu=False)

def run():
    global placas_recentes
    cap = cv2.VideoCapture(config["CAMERA_RTSP_URL"])
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o stream da câmera: {config['CAMERA_RTSP_URL']}")
        return

    print("[INFO] Processando vídeo... Pressione 'q' para sair.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: print("[AVISO] Stream finalizado."); break
        
        frame_count += 1
        if frame_count % config["PARAMETROS_PERFORMANCE"]["FRAME_SKIP"] != 0: continue

        h, w = frame.shape[:2]
        r = config["PARAMETROS_PERFORMANCE"]["FRAME_WIDTH"] / float(w)
        frame = cv2.resize(frame, (config["PARAMETROS_PERFORMANCE"]["FRAME_WIDTH"], int(h * r)), interpolation=cv2.INTER_AREA)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        agora = datetime.now()
        cooldown = config["PARAMETROS_DETECCAO"]["COOLDOWN_SEGUNDOS"]
        placas_recentes = {p: t for p, t in placas_recentes.items() if (agora - t) <= timedelta(seconds=cooldown)}

        for i in range(detections.shape[2]):
            confidence, idx = detections[0, 0, i, 2], int(detections[0, 0, i, 1])
            if confidence > config["PARAMETROS_DETECCAO"]["CONFIDENCE_THRESHOLD"] and CLASSES[idx] in ["car", "bus", "motorbike"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                vehicle_img = frame[startY:endY, startX:endX]
                if vehicle_img.size == 0: continue

                if config["MODO_OPERACAO"]["DEBUG_MODE"]: cv2.imshow("Debug - Vehicle Crop", vehicle_img)

                image_to_ocr = find_plate_candidates_advanced(vehicle_img, config["PARAMETROS_DETECTOR_AVANCADO"], config) if config["MODO_OPERACAO"]["USE_ADVANCED_PLATE_FINDER"] else vehicle_img
                
                if image_to_ocr is not None:
                    if config["MODO_OPERACAO"]["DEBUG_MODE"]: cv2.imshow("Debug - Image Sent to OCR", image_to_ocr)
                    ocr_results = reader.readtext(image_to_ocr, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    for (bbox, text, prob) in ocr_results:
                        if prob > config["PARAMETROS_DETECCAO"]["OCR_CONFIDENCE_THRESHOLD"]:
                            is_valid, plate_text = is_valid_plate_format(text)
                            if is_valid and plate_text not in placas_recentes:
                                placas_recentes[plate_text] = agora
                                timestamp_obj, timestamp_str = datetime.now(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                image_name = f"placa_{timestamp_obj.strftime('%Y%m%d_%H%M%S')}.png"
                                os.makedirs(CAPTURAS_DIR, exist_ok=True)
                                cv2.imwrite(os.path.join(CAPTURAS_DIR, image_name), vehicle_img)
                                save_plate_to_csv(image_name, plate_text, timestamp_str, "placas")
                                print(f"[SUCESSO] Placa detectada: {plate_text}")

                                match_info = find_match_in_whitelist(plate_text, authorized_vehicles, config["PARAMETROS_AUTORIZACAO"]["TOLERANCIA_MATCH"])
                                if match_info:
                                    trigger_release_action(plate_text, match_info)

                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                                cv2.putText(frame, plate_text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                break
        cv2.imshow("Detecção de Placas - Pressione 'q' para sair", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("[INFO] Finalizando...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
