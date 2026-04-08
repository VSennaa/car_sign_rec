import os
import customtkinter as ctk
import cv2
import numpy as np
import easyocr
import re
import json
import Levenshtein
from PIL import Image
from datetime import datetime, timedelta
from logic import load_config, load_authorized_vehicles, find_match_in_whitelist, trigger_release_action, is_valid_plate_format, find_plate_candidates_advanced, save_plate_to_csv

# Parâmetros de Rede
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay'

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Captura de Vídeo - Integrado")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")

        # Gestão do Loop
        self.is_camera_running = False
        self.frame_count = 0
        self.placas_recentes = {}
        self.status_timer = None
        self.last_status_time = datetime.now()

        # Carregar Recursos
        self.config = load_config()
        if self.config is None: exit()
        self.authorized_vehicles = load_authorized_vehicles(self.config["PARAMETROS_AUTORIZACAO"]["ARQUIVO_SERVIDORES_CSV"])
        
        PROTOTXT, MODEL = "deploy.prototxt", "mobilenet_iter_73000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        self.CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        self.reader = easyocr.Reader(['pt'], gpu=False)

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.btn_start = ctk.CTkButton(self.sidebar, text="Iniciar Câmera", command=self.start_camera)
        self.btn_start.pack(pady=10, padx=20)

        self.btn_pause = ctk.CTkButton(self.sidebar, text="Pausar", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(pady=10, padx=20)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="Parar Câmera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(pady=10, padx=20)

        self.btn_settings = ctk.CTkButton(self.sidebar, text="Configurações", command=self.open_settings)
        self.btn_settings.pack(pady=10, padx=20)

        self.entry_url = ctk.CTkEntry(self.sidebar, placeholder_text="URL RTSP, índice 0 ou caminho de arquivo")
        self.entry_url.pack(pady=10, padx=20)
        self.entry_url.insert(0, self.config.get('CAMERA_RTSP_URL', '0'))

        # Variáveis de estado
        self.is_paused = False

        # Video Label
        self.video_label = ctk.CTkLabel(self, text="Tela de Vídeo")
        self.video_label.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Status Label
        self.status_label = ctk.CTkLabel(self, text="", font=("Arial", 28, "bold"), corner_radius=10)
        self.status_label.place(relx=0.5, rely=0.9, anchor="center")

        self.cap = None

    def open_settings(self):
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("Configurações")
        settings_window.geometry("400x500")

        # Scrollable frame for many settings
        scroll = ctk.CTkScrollableFrame(settings_window)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # Dictionary to store entries for saving
        self.settings_entries = {}

        # Recursive function to build UI for dict
        def create_ui(data, parent, path=""):
            for key, value in data.items():
                if isinstance(value, dict):
                    section = ctk.CTkLabel(parent, text=key, font=("Arial", 14, "bold"))
                    section.pack(pady=(10, 0))
                    create_ui(value, parent, f"{path}{key}.")
                else:
                    lbl = ctk.CTkLabel(parent, text=f"{key}:")
                    lbl.pack(pady=(5, 0))
                    entry = ctk.CTkEntry(parent)
                    entry.insert(0, str(value))
                    entry.pack(fill="x")
                    # Store path to reconstruct dict
                    entry.path = f"{path}{key}"
                    self.settings_entries[entry.path] = entry

        create_ui(self.config, scroll)

        def save_settings():
            # Update self.config based on entries
            def update_dict(d, path, value):
                keys = path.split('.')
                for k in keys[:-1]:
                    d = d[k]
                
                val_str = str(value).strip()
                if val_str.lower() == "true": value = True
                elif val_str.lower() == "false": value = False
                elif val_str.startswith("[") and val_str.endswith("]"):
                    try: value = json.loads(val_str)
                    except: pass
                else:
                    try:
                        if "." in val_str: value = float(val_str)
                        else: value = int(val_str)
                    except: pass
                d[keys[-1]] = value

            for path, entry in self.settings_entries.items():
                update_dict(self.config, path, entry.get())
            
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            print("[INFO] Configurações salvas e recarregadas.")
            settings_window.destroy()

        ctk.CTkButton(settings_window, text="Salvar", command=save_settings).pack(pady=10)

    def show_status(self, text, bg_color):
        if self.status_timer:
            self.after_cancel(self.status_timer)
        self.status_label.configure(text=text, fg_color=bg_color, text_color="white")
        self.status_timer = self.after(5000, self.clear_status)

    def clear_status(self):
        self.status_label.configure(text="", fg_color="transparent")

    def start_camera(self):
        # Limpa o input e remove aspas
        source = self.entry_url.get().strip().strip('\"').strip("\'")
        
        # Lógica de Seleção de Fonte
        if source.isdigit():
            source = int(source)
        elif not source.startswith("rtsp") and not source.startswith("http"):
            if not os.path.exists(source):
                print(f"[ERRO] Arquivo ou fonte não encontrada: {source}")
                return
        
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.is_camera_running = True
            self.is_paused = False
            self.btn_start.configure(state="disabled")
            self.btn_pause.configure(state="normal", text="Pausar")
            self.btn_stop.configure(state="normal")
            self.update_frame()
        else:
            print("[ERRO] Não foi possível abrir o stream.")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.configure(text="Continuar" if self.is_paused else "Pausar")

    def stop_camera(self):
        self.is_camera_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="Pausar")
        self.btn_stop.configure(state="disabled")
        self.video_label.configure(text="Câmera Parada")

    def update_frame(self):
        if self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_camera()
                return

            if not self.is_paused:
                self.frame_count += 1
                
                # AI Logic with Frame Skip
                if self.frame_count % self.config["PARAMETROS_PERFORMANCE"]["FRAME_SKIP"] == 0:
                    h, w = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
                    self.net.setInput(blob)
                    detections = self.net.forward()

                    agora = datetime.now()
                    cooldown = self.config["PARAMETROS_DETECCAO"]["COOLDOWN_SEGUNDOS"]
                    self.placas_recentes = {p: t for p, t in self.placas_recentes.items() if (agora - t) <= timedelta(seconds=cooldown)}

                    for i in range(detections.shape[2]):
                        confidence, idx = detections[0, 0, i, 2], int(detections[0, 0, i, 1])
                        if confidence > self.config["PARAMETROS_DETECCAO"]["CONFIDENCE_THRESHOLD"] and self.CLASSES[idx] in ["car", "bus", "motorbike"]:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            vehicle_img = frame[startY:endY, startX:endX]
                            if vehicle_img.size == 0: continue

                            image_to_ocr = find_plate_candidates_advanced(vehicle_img, self.config["PARAMETROS_DETECTOR_AVANCADO"], self.config) if self.config["MODO_OPERACAO"]["USE_ADVANCED_PLATE_FINDER"] else vehicle_img
                            
                            if image_to_ocr is not None:
                                ocr_results = self.reader.readtext(image_to_ocr, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                                for (bbox, text, prob) in ocr_results:
                                    if prob > self.config["PARAMETROS_DETECCAO"]["OCR_CONFIDENCE_THRESHOLD"]:
                                        is_valid, plate_text = is_valid_plate_format(text)
                                        if is_valid and plate_text not in self.placas_recentes:
                                            self.placas_recentes[plate_text] = agora
                                            timestamp_obj, timestamp_str = datetime.now(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            image_name = f"placa_{timestamp_obj.strftime('%Y%m%d_%H%M%S')}.png"
                                            
                                            os.makedirs("capturas", exist_ok=True)
                                            cv2.imwrite(os.path.join("capturas", image_name), vehicle_img)
                                            save_plate_to_csv(image_name, plate_text, timestamp_str, "placas")
                                            print(f"[SUCESSO] Placa detectada: {plate_text}")

                                            match_info = find_match_in_whitelist(plate_text, self.authorized_vehicles, self.config["PARAMETROS_AUTORIZACAO"]["TOLERANCIA_MATCH"])
                                            
                                            # Status feedback with Cooldown
                                            if (datetime.now() - self.last_status_time).seconds > 5:
                                                if match_info:
                                                    trigger_release_action(plate_text, match_info)
                                                    self.show_status(f"AUTORIZADO: {match_info['servidor']}", "green")
                                                else:
                                                    self.show_status(f"NEGADO: {plate_text}", "red")
                                                self.last_status_time = datetime.now()

                                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                                            cv2.putText(frame, plate_text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                            break

            # Update UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(800, 600))
            
            self.current_image = ctk_image
            self.video_label.configure(image=self.current_image, text="")
            self.after(10, self.update_frame)

if __name__ == "__main__":
    app = App()
    app.mainloop()
