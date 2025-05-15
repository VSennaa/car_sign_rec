import os
import cv2
import time
import numpy as np

# Definir as classes de objetos do modelo MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", 
           "tvmonitor"]

# Caminho da pasta onde as imagens serão salvas
capturas_dir = "capturas"

# Cria a pasta "capturas" se não existir
os.makedirs(capturas_dir, exist_ok=True)

# Caminho para os arquivos do modelo
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

# Carregar o modelo Caffe
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Função para gerar o próximo nome de arquivo
def get_next_image_name():
    # Lista todos os arquivos na pasta "capturas"
    arquivos = os.listdir(capturas_dir)

    # Filtra apenas os arquivos com nome "img_*.png"
    imagens = [f for f in arquivos if f.startswith("img_") and f.endswith(".png")]

    # Se não houver nenhuma imagem, começa do 1
    if len(imagens) == 0:
        return "img_1.png"

    # Extrai os números dos arquivos e encontra o maior número
    numeros = [int(f.split('_')[1].split('.')[0]) for f in imagens]
    maior_numero = max(numeros)

    # Retorna o próximo número
    return f"img_{maior_numero + 1}.png"

# Inicializa a câmera
cap = cv2.VideoCapture(0)  # ou use 'rtsp://...' para IP Camera

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    
    # Prepara o frame para detecção
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_vehicle = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label in ["car", "bus", "motorbike", "truck"]:  # veículos
                detected_vehicle = True

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}",
                            (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if detected_vehicle:
        # Obter o próximo nome de arquivo
        filename = os.path.join(capturas_dir, get_next_image_name())
        
        # Salva a imagem em PNG
        cv2.imwrite(filename, frame)
        print(f"[INFO] Veículo detectado. Imagem salva como {filename}")
        time.sleep(1)  # evita capturar várias imagens por segundo

    # Mostra a imagem (opcional)
    cv2.imshow("Detecção de Veículos", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera
cap.release()
cv2.destroyAllWindows()
