import os
import cv2
import time
import numpy as np

# Lista de classes do MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]

# Caminhos para os arquivos do modelo
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

# Inicializa a rede neural
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Diretório onde as imagens serão salvas
capturas_dir = "capturas"
os.makedirs(capturas_dir, exist_ok=True)

# Gera nome sequencial da próxima imagem
def get_next_image_name():
    arquivos = os.listdir(capturas_dir)
    imagens = [f for f in arquivos if f.startswith("img_") and f.endswith(".png")]
    if len(imagens) == 0:
        return "img_1.png"
    numeros = [int(f.split('_')[1].split('.')[0]) for f in imagens]
    maior = max(numeros)
    return f"img_{maior + 1}.png"

# Inicializa a câmera
cap = cv2.VideoCapture(0)  # Altere para URL RTSP se for câmera IP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Pré-processa a imagem
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Verifica se é um veículo
            if label in ["car", "bus", "motorbike", "truck"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Garante que os limites estejam dentro da imagem
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                if endX > startX and endY > startY:
                    vehicle_image = frame[startY:endY, startX:endX]

                    if vehicle_image.size != 0:
                        filename = os.path.join(capturas_dir, get_next_image_name())
                        cv2.imwrite(filename, vehicle_image)
                        print(f"[INFO] Veículo detectado. Imagem salva como {filename}")
                        time.sleep(1)  # evita salvar muitos frames seguidos
                    else:
                        print("[AVISO] Imagem recortada vazia, ignorada.")
                else:
                    print("[AVISO] Coordenadas de recorte inválidas.")

    # Exibe a imagem com a detecção (opcional)
    cv2.imshow("Detecção de Veículos", frame)

    # Encerra ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
