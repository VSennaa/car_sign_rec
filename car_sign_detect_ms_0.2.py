import os
import cv2
import time
import numpy as np
import easyocr
import pandas as pd

# Classes do modelo
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]

# Caminhos do modelo
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Pastas e arquivos
capturas_dir = "capturas"
placas_dir = "placas"
csv_path = os.path.join(placas_dir, "placas.csv")
os.makedirs(capturas_dir, exist_ok=True)
os.makedirs(placas_dir, exist_ok=True)

# Inicializa EasyOCR
print("[INFO] Inicializando EasyOCR...")
reader = easyocr.Reader(['pt', 'en'])
print("[INFO] EasyOCR pronto!")

# Lista de palavras a serem filtradas (como "BRASIL", "BR", etc.)
PALAVRAS_FILTRO = ["BRASIL", "BR", "GOVERNO", "DE", "GOVERNMENT", "DEBRASIL"]

# CSV inicial
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["imagem", "placa"]).to_csv(csv_path, index=False)

# Função para obter o próximo nome sequencial de imagem
def get_next_image_name():
    arquivos = os.listdir(capturas_dir)
    imagens = [f for f in arquivos if f.startswith("img_") and f.endswith(".png")]
    if not imagens:
        return "img_1.png"
    numeros = [int(f.split('_')[1].split('.')[0]) for f in imagens]
    return f"img_{max(numeros) + 1}.png"

# Inicia a câmera
cap = cv2.VideoCapture(0)

NUM_TENTATIVAS = 5  # Número de tentativas para ler a placa

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Não foi possível ler da câmera.")
        break

    cv2.imshow("Detecção de Veículos", frame)  # Exibe o frame da câmera
    cv2.waitKey(1)

    (h, w) = frame.shape[:2]

    # Prepara a imagem para a rede neural
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label in ["car", "bus", "motorbike", "truck"]:  # Detecta veículos
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                placa_detectada = None

                # Tenta fazer a detecção de placa várias vezes
                for tentativa in range(NUM_TENTATIVAS):
                    ret, new_frame = cap.read()
                    if not ret:
                        break

                    vehicle_img = new_frame[startY:endY, startX:endX]

                    if vehicle_img.size == 0:
                        continue

                    filename = get_next_image_name()
                    filepath = os.path.join(capturas_dir, filename)
                    cv2.imwrite(filepath, vehicle_img)  # Salva a imagem do veículo
                    print(f"[INFO] Imagem salva: {filepath}")

                    # Realiza o OCR na imagem do veículo
                    ocr_results = reader.readtext(vehicle_img)

                    if ocr_results:
                        placa = ocr_results[0][1].upper().strip()

                        # Filtra palavras indesejadas
                        if any(palavra in placa for palavra in PALAVRAS_FILTRO):
                            print(f"[INFO] Ignorando placa com palavra filtrada: {placa}")
                            continue  # Ignora essa placa e tenta a próxima

                        placa_detectada = placa
                        print(f"[INFO] Placa detectada: {placa}")

                        # Adiciona a placa ao CSV
                        df = pd.read_csv(csv_path)
                        df = pd.concat([df, pd.DataFrame([[filename, placa]], columns=["imagem", "placa"])])
                        df.to_csv(csv_path, index=False)
