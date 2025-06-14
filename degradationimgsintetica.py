import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import urllib.request
import zipfile
from pathlib import Path  # al inicio del archivo si no está
import csv

# ------------------------
# MODELOS PREENTRENADOS LIGEROS
# ------------------------

class LightMotionBlurModel(torch.nn.Module):
    """Modelo ligero para desenfoque de movimiento (4 capas convolucionales)"""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 3, kernel_size=3, padding=1)
        
    def forward(self, x, intensity=0.5):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x  # El desenfoque ya se aplicó en las convoluciones

def apply_color_tone(image, tone="sepia", strength=0.6, grayscale_first=True):
    """
    Aplica un virado tonal a una imagen en escala de grises.
    - tone: 'sepia', 'blue', 'rose'
    - strength: cuánto mezcla con el color
    - grayscale_first: si True, convierte primero a escala de grises
    """
    if tone == "sepia":
        color = np.array([30, 66, 112])  # BGR
    elif tone == "blue":
        color = np.array([170, 120, 60])  # azul lavado
    elif tone == "rose":
        color = np.array([180, 150, 200])
    else:
        return image

    if grayscale_first:
        image = np.clip(image, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    color_layer = np.ones_like(image) * color
    blended = cv2.addWeighted(image.astype(np.float32), 1 - strength,
                              color_layer.astype(np.float32), strength, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_scratch_template(image, scratch_dir):
    scratch_files = list(scratch_dir.glob("*.jpg"))
    if not scratch_files:
        return image
    scratch = cv2.imread(str(random.choice(scratch_files)), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    scratch = cv2.resize(scratch, (w, h))
    M = cv2.getRotationMatrix2D((w//2, h//2), random.uniform(0, 360), random.uniform(0.5, 1.5))
    scratch = cv2.warpAffine(scratch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flip = random.choice([0, 1, -1, None])
    if flip is not None:
        scratch = cv2.flip(scratch, flip)
    mask = scratch.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask] * 3, axis=-1)  # Expande a (H, W, 3)

    alpha = 0.3
    scratch_layer = image.astype(np.float32) * (1 - alpha * mask_3ch)

    #scratch_layer = (1.0 - mask_3ch) * image.astype(np.float32)
    return np.clip(scratch_layer, 0, 255).astype(np.uint8)

def apply_vignette(image, strength=15, blur_kernel=51, alpha=0.8):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    norm_x = (x - cx) / cx
    norm_y = (y - cy) / cy
    radius = np.sqrt(norm_x**2 + norm_y**2)
    mask = 1 - np.clip(radius**strength, 0, 1)
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    mask_3ch = np.stack([mask]*3, axis=-1)
    vignette = image.astype(np.float32) * (1 - alpha) + image.astype(np.float32) * mask_3ch * alpha
    return np.clip(vignette, 0, 255).astype(np.uint8)


def apply_blobs(image, num_spots=30, size_range=(20, 60), intensity_range=(80, 180)):
    output = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    for _ in range(num_spots):
        radius = random.randint(*size_range)
        x = random.randint(radius, w - radius)
        y = random.randint(radius, h - radius)
        intensity = random.randint(*intensity_range)
        color = random.choice([-1, 1])
        blob_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(blob_mask, (x, y), radius, 1.0, -1, lineType=cv2.LINE_AA)
        blob_mask = cv2.GaussianBlur(blob_mask, (0, 0), sigmaX=radius/2, sigmaY=radius/2)
#        output += blob_mask[:, :, None] * color * intensity

        alpha = 0.3
        blob_effect = blob_mask[:, :, None] * color * intensity
        output = (1 - alpha) * output + alpha * (output + blob_effect)


    return np.clip(output, 0, 255).astype(np.uint8)


def apply_clean_bw_look(image, contrast=1.15, brightness=10, sharpen=True):
    """
    Convierte a B/N con buen contraste y bordes definidos.
    - contrast >1 = más contraste
    - brightness = brillo extra
    - sharpen = aplica máscara de enfoque
    """
    # Escala de grises y expansión a 3 canales
  # Escala de grises y expansión a 3 canales
    image = np.clip(image, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if sharpen:
        blur = cv2.GaussianBlur(gray_3ch, (0, 0), 1.0)
        sharpened = cv2.addWeighted(gray_3ch, 1.5, blur, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return gray_3ch


def apply_dirty_bw_look(image, noise_amount=0.03, contrast_boost=1.7):
    """
    Aplica un look sucio y agresivo de B/N con contraste extremo.
    """
    # Escala de grises
    image = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estiramiento de contraste
    gray = cv2.convertScaleAbs(gray, alpha=contrast_boost, beta=0)

    # Umbral adaptativo para perder tonos intermedios
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Añadir ruido blanco y negro (sal y pimienta)
    noise = np.random.rand(*binary.shape)
    binary[noise < noise_amount] = 0
    binary[noise > 1 - noise_amount] = 255

    # Volver a 3 canales
    dirty_bw = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return dirty_bw


def apply_flat_gray_look(image, desat_level=0.5, contrast=0.8, blur_radius=1.2):
        """
        Aplica una apariencia grisácea desaturada y suave.
        - desat_level: 0 (sin cambio), 1 (totalmente en escala de grises)
        - contrast: <1 para aplanar
        - blur_radius: para suavizar bordes
        """
        # Convertir a escala de grises
        image = np.clip(image, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Mezcla de desaturación
        blended = cv2.addWeighted(image.astype(np.float32), 1 - desat_level,
                                gray_3ch.astype(np.float32), desat_level, 0)

        # Contraste
        mean = np.mean(blended)
        flat = (blended - mean) * contrast + mean

        # Desenfoque
        blurred = cv2.GaussianBlur(flat, (0, 0), blur_radius)

        return np.clip(blurred, 0, 255).astype(np.uint8)
def apply_soft_bw_style(image, contrast=1.05, brightness=-5, blur_radius=1.2):
    """
    Estilo suave en blanco y negro como en películas antiguas.
    - contrast: cerca de 1 para mantener detalle sin exagerar
    - brightness: valor negativo para dar tono más oscuro
    - blur_radius: para suavizar bordes
    """
    # Convertir a gris
    image = np.clip(image, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Suavizado tipo película
    blurred = cv2.GaussianBlur(gray_3ch, (0, 0), blur_radius)

    return np.clip(blurred, 0, 255).astype(np.uint8)


# ------------------------
# CLASE PRINCIPAL
# ------------------------

class VideoDegrader:
    def __init__(self, input_path):
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {input_path}")
        
        if not self.cap.isOpened():
            print("❌ No se pudo abrir la secuencia")
        else:
            print("✅ Secuencia abierta")
            print("Frame count:", self.cap.get(cv2.CAP_PROP_FRAME_COUNT))



        # Definición de perfiles de degradación
        self.profiles = {
            "1920s": {"flicker_intensity": 0.25, "grain_size": 3, "scratch_speed": 2, "color_shift": (0.7, 0.6, 0.5), "motion_blur": 0.4, "burn_prob": 0.02},
            "1950s": {"flicker_intensity": 0.15, "grain_size": 2, "scratch_speed": 3, "color_shift": (0.9, 0.8, 0.7), "motion_blur": 0.3, "burn_prob": 0.01},
            "1970s": {"flicker_intensity": 0.1, "grain_size": 1, "scratch_speed": 4, "color_shift": (0.8, 0.9, 0.8), "motion_blur": 0.2, "burn_prob": 0.005},
            "vhs":   {"flicker_intensity": 0.08, "grain_size": 2, "scratch_speed": 5, "color_shift": (0.85, 0.85, 1.1), "motion_blur": 0.1, "burn_prob": 0.001, "vhs_effect": True}
        }
        # Configuración global
        self.apply_vignette_globally = True
        self.blob_prob = 0.1
        self.scratch_prob = 0.3
        self.scratch_dir = Path("/home/laura/CycleGAN/00Databases/plantillasScratches")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 15
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Modelo de desenfoque de movimiento
        self.motion_blur_model = LightMotionBlurModel()
        self.motion_blur_model.eval()
        
        # Estados para coherencia temporal
        self.prev_grain = None
        self.jitter_offset = (0, 0)
        self.burn_mask = None
        self.burn_lifetime = 0
        self.color_tone = "rose"
        self.color_tone_strength = 0.2
        self.apply_clean_bw = False
        self.apply_soft_bw = True
        self.scratch_count = 3
        self.scratch_positions = [random.randint(0, self.width) for _ in range(self.scratch_count)]

        self.input_path = input_path  # Para tener el patrón disponible en otros métodos




    # ------------------------
    # GENERARA MULTIPLES VERSIONES DE UN VIDEO
    # ------------------------

    

    def generate_versions(self, n, output_prefix, csv_path, description="", output_dir="video_variaciones"):
        """
        Genera n versiones degradadas y guarda la configuración en un CSV.
        """
        fieldnames = [
            'description', 'output_file', 'era', 'apply_vignette_globally',
            'blob_prob', 'scratch_prob', 'color_tone', 'color_tone_strength',
            'apply_clean_bw', 'apply_soft_bw', 'scratch_count','scratch_positions' 
        ]

        csv_full_path = os.path.join(output_dir, csv_path)
        with open(csv_full_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n):
                # Aleatorizar configuración
                era = random.choice(list(self.profiles.keys()))
                self.apply_vignette_globally =  random.choice([True, False])
                self.blob_prob = round(random.uniform(0, 0.2), 2)
                self.scratch_prob = round(random.uniform(0, 0.5), 2)
                self.color_tone = random.choice(['sepia', 'blue', 'rose', None])
                self.color_tone_strength = round(random.uniform(0, 1), 2)
                self.apply_clean_bw = random.choice([True, False])
                self.apply_soft_bw = random.choice([True, False])
                self.scratch_count = random.randint(1, 5)
                self.scratch_positions = [random.randint(0, self.width) for _ in range(self.scratch_count)]


               
                output_file = os.path.join(output_dir, f"{output_prefix}_{i+1:03d}.mp4")

                # Aplicar degradación con era seleccionada
                # Reiniciamos el VideoCapture para cada corrida
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.apply_temporal_degradation(output_file, era=era)


                output_name = f"{output_prefix}_{i+1:03d}.mp4"
                output_file = os.path.join(output_dir, output_name)

                # Guardar configuración en CSV
                writer.writerow({
                    'description': description,
                    'output_file': output_name,
                    'era': era,
                    'apply_vignette_globally': self.apply_vignette_globally,
                    'blob_prob': self.blob_prob,
                    'scratch_prob': self.scratch_prob,
                    'color_tone': self.color_tone,
                    'color_tone_strength': self.color_tone_strength,
                    'apply_clean_bw': self.apply_clean_bw,
                    'apply_soft_bw': self.apply_soft_bw,
                    'scratch_count': self.scratch_count,
                    'scratch_positions': ','.join(map(str, self.scratch_positions))

                })

    def recreate_from_csv(self, csv_path, output_dir="."):

        """
        Lee un CSV con configuraciones y recrea cada video.
        """
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Aplicar configuración leída
                era = row['era']
                self.apply_vignette_globally = row['apply_vignette_globally'] == 'True'
                self.blob_prob = float(row['blob_prob'])
                self.scratch_prob = float(row['scratch_prob'])
                self.color_tone = row['color_tone'] if row['color_tone'] != 'None' else None
                self.color_tone_strength = float(row['color_tone_strength'])
                self.apply_clean_bw = row['apply_clean_bw'] == 'True'
                self.apply_soft_bw = row['apply_soft_bw'] == 'True'
                self.scratch_count = int(row['scratch_count'])
                self.scratch_positions = list(map(int, row['scratch_positions'].split(',')))


                nombre_base = row['output_file']
                nombre_sin_ext, ext = os.path.splitext(nombre_base)
                output_file = os.path.join(output_dir, f"{nombre_sin_ext}R{ext}")

                #output_file = os.path.join(output_dir, row['output_file']) 
                # Reiniciar posición de lectura
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.apply_temporal_degradation(output_file, era=era)


    def apply_temporal_degradation(self, output_path, era="1950s"):
        """
        Aplica degradación temporal al video completo y guarda el resultado.
        """


        print("✅ Secuencia abierta en apply_temporal_degradation")
        print("Frame count:", self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        config = self.profiles[era]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        flicker_phase = random.uniform(0, 2 * np.pi)
        os.makedirs("debug_frames", exist_ok=True)

        original_positions = self.scratch_positions.copy()  # ⚠️ COPIA para no modificar la lista original


        for frame_idx in tqdm(range(self.frame_count), desc=f"Degradando video ({era})"):
            ret, frame = self.cap.read()
            if not ret:
                print("no hay ret")
                break
            frame = frame.astype(np.float32)
            debug_imgs = {"original": frame.copy()}

            # 1. Flicker
            flicker_val = 1 - config["flicker_intensity"] * abs(np.sin(flicker_phase + frame_idx * 0.1))
            frame = frame * flicker_val
            debug_imgs["flicker"] = frame.copy()

            # 2. Grain
            grain = np.random.normal(0, 5 * config["grain_size"], (self.height, self.width, 3))
            grain = np.clip(grain, -30, 30)
            if self.prev_grain is not None:
                grain = 0.7 * grain + 0.3 * self.prev_grain
            frame = frame * 0.9 + grain * 0.1
            self.prev_grain = grain
            debug_imgs["grain"] = frame.copy()

            # 3. Scratches (rayas)
            scratch_mask = np.zeros((self.height, self.width, 3), dtype=np.float32)
            for i in range(self.scratch_count):
                pos = (original_positions[i] + frame_idx * config["scratch_speed"]) % self.width
                scratch_width = max(1, int(self.width * 0.005))
                scratch_mask[:, pos:pos + scratch_width, :] = 1.0
            frame = frame * (1 - 0.3 * scratch_mask) + 30 * scratch_mask
            debug_imgs["scratches"] = frame.copy()

            # 4. Color shift
            frame = self._apply_color_shift(frame, config["color_shift"])
            debug_imgs["color_shift"] = frame.copy()

            # 5. Jitter
            if frame_idx % 10 == 0:
                self.jitter_offset = (random.randint(-3, 3), random.randint(-3, 3))
            frame = self._shift_frame(frame, self.jitter_offset)
            debug_imgs["jitter"] = frame.copy()

            # 6. Film burn
            if random.random() < config.get("burn_prob", 0):
                self.burn_mask = self._create_burn_mask()
                self.burn_lifetime = random.randint(10, 30)
            if self.burn_mask is not None and self.burn_lifetime > 0:
                frame = self._simulate_film_burn(frame, self.burn_mask)
                self.burn_lifetime -= 1
                dx, dy = random.randint(-2, 2), random.randint(-2, 2)
                self.burn_mask = self._shift_mask(self.burn_mask, dx, dy)
                debug_imgs["burn"] = frame.copy()

            # 7. VHS effect
            if config.get("vhs_effect", False):
                frame = self._vhs_effect(frame)
                debug_imgs["vhs"] = frame.copy()

            # 8. Motion blur (opcional)
            # if config["motion_blur"] > 0:
            #     frame = self._apply_motion_blur(frame, config["motion_blur"])
            #     debug_imgs["motion_blur"] = frame.copy()

            # 9. Scratch template aleatorio
            if random.random() < self.scratch_prob:
                frame = apply_scratch_template(frame, self.scratch_dir)
                debug_imgs["scratch_template"] = frame.copy()

            # 10. Viñeta global
            if self.apply_vignette_globally:
                frame = apply_vignette(frame)
                debug_imgs["vignette"] = frame.copy()

            # 11. Blobs
            if random.random() < self.blob_prob:
                frame = apply_blobs(frame, num_spots=10)
                debug_imgs["blobs"] = frame.copy()

            # 12. Color tone
            if self.color_tone is not None:
                frame = apply_color_tone(frame, tone=self.color_tone, strength=self.color_tone_strength)
                debug_imgs["tonal"] = frame.copy()

            # 13. Clean B/W look
            if self.apply_clean_bw:
                frame = apply_clean_bw_look(frame)
                debug_imgs["clean_bw"] = frame.copy()

            # 14. Soft B/W style
            if self.apply_soft_bw:
                frame = apply_soft_bw_style(frame)
                debug_imgs["soft_bw"] = frame.copy()

            # Guardar imágenes de depuración cada 10 frames
            if frame_idx % 10 == 0:
                for step_name, step_img in debug_imgs.items():
                    debug_path = os.path.join("debug_frames", f"frame{frame_idx:04d}_{step_name}.jpg")
                    cv2.imwrite(debug_path, np.clip(step_img, 0, 255).astype(np.uint8))
                print(f"[DEBUG] Guardadas imágenes del frame {frame_idx} en 'debug_frames/'.")

            # Escribir frame final
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.write(frame)
            

        #self.cap.release()
        out.release()
        print(f"Video degradado guardado en: {output_path}")
    
    # ------------------------
    # MÉTODOS AUXILIARES
    # ------------------------
    
    def _apply_color_shift(self, frame, shift):
        b, g, r = cv2.split(frame)
        b = b * shift[0]
        g = g * shift[1]
        r = r * shift[2]
        return cv2.merge([b, g, r])
    
    def _shift_frame(self, frame, offset):
        M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        return cv2.warpAffine(frame, M, (self.width, self.height))
    
    def _apply_motion_blur(self, frame, intensity):
        with torch.no_grad():
            img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
            blurred = self.motion_blur_model(img_tensor, intensity)
            blurred = blurred.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            return blurred * 255.0


            
        return blurred
    
    def _create_burn_mask(self, size_ratio=0.3):
        """Crea una máscara de quemado con forma elíptica"""
        mask = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Posición aleatoria
        cx = random.randint(0, self.width)
        cy = random.randint(0, self.height)
        
        # Tamaño aleatorio
        w = int(self.width * size_ratio * random.uniform(0.5, 1.5))
        h = int(self.height * size_ratio * random.uniform(0.5, 1.5))
        
        # Crear elipse
        cv2.ellipse(mask, (cx, cy), (w//2, h//2), 0, 0, 360, (1.0, 1.0, 1.0), -1)
        
        # Suavizar bordes
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        return mask
    
    def _simulate_film_burn(self, frame, burn_mask):
        """Aplica efecto de quemado en la película"""
        # Intensidad aleatoria
        intensity = random.uniform(0.1, 0.3)
        
        # Aplicar efecto (oscurecimiento + cambio de color)
        burned = frame * (1 - burn_mask * intensity)
        
        # Añadir tinte amarillo/naranja
        b, g, r = cv2.split(burned)
        b = b * 0.9  # Reducir azul
        g = g * 1.1  # Aumentar verde
        
        return cv2.merge([b, g, r])
    
    def _shift_mask(self, mask, dx, dy):
        """Desplaza la máscara manteniendo bordes"""
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(mask, M, (self.width, self.height))
        
        # Rellenar bordes con cero
        if dx > 0: shifted[:, :dx] = 0
        if dx < 0: shifted[:, dx:] = 0
        if dy > 0: shifted[:dy, :] = 0
        if dy < 0: shifted[dy:, :] = 0
            
        return shifted
    
    def _vhs_effect(self, frame):
        """Efecto de distorsión VHS (bleeding de color)"""
        # Separar canales
        b, g, r = cv2.split(frame)
        
        # Desfasar canal azul (bleeding característico)
        offset_x = random.randint(1, 3)
        offset_y = random.randint(0, 1)
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        b = cv2.warpAffine(b, M, (self.width, self.height))
        
        # Añadir ruido de croma
        chroma_noise = np.random.normal(0, 10, (self.height, self.width))
        r = r + chroma_noise
        g = g + chroma_noise
        
        return cv2.merge([b, g, r])

# ------------------------
# FUNCIÓN PARA DESCARGAR DAVIS
# ------------------------

def download_davis_dataset(output_dir="DAVIS"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    zip_path = os.path.join(output_dir, "davis.zip")
    
    if not os.path.exists(zip_path):
        print("Descargando dataset DAVIS (480p)...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Descomprimiendo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            
    davis_path = os.path.join(output_dir, "DAVIS")
    return {
        "JPEGImages": os.path.join(davis_path, "JPEGImages", "480p"),
        "Annotations": os.path.join(davis_path, "Annotations", "480p")
    }

def procesar_carpeta_personalizada(
    carpeta_entrada, carpeta_videos, carpeta_fotogramas, num_versiones=1
):
    """
    Procesa una carpeta con subcarpetas (cada una es una escena).
    Por cada subcarpeta genera num_versiones videos degradados y el CSV de config,
    y guarda todos los fotogramas degradados en otra carpeta.
    """
    carpeta_entrada = Path(carpeta_entrada)
    carpeta_videos = Path(carpeta_videos)
    carpeta_fotogramas = Path(carpeta_fotogramas)
    carpeta_videos.mkdir(exist_ok=True, parents=True)
    carpeta_fotogramas.mkdir(exist_ok=True, parents=True)

    escenas = [f for f in carpeta_entrada.iterdir() if f.is_dir()]
    print(f"🔍 Se encontraron {len(escenas)} escenas en {carpeta_entrada}")

    for escena in escenas:
        print(f"🎞️ Procesando escena: {escena.name}")

        # Buscar todos los .jpg y .png de la subcarpeta, ordenados
        fotogramas = sorted(list(escena.glob("*.jpg")) + list(escena.glob("*.png")))
        if len(fotogramas) == 0:
            print(f"⚠️ No se encontraron imágenes en {escena}, se omite.")
            continue

        # Crear un archivo temporal de video a partir de los fotogramas
        temp_video = carpeta_videos / f"{escena.name}_tempinput.mp4"
        frame0 = cv2.imread(str(fotogramas[0]))
        height, width = frame0.shape[:2]
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        for img_path in fotogramas:
            img = cv2.imread(str(img_path))
            if img is not None and img.shape[:2] == (height, width):
                vw.write(img)
            else:
                print(f"[WARN] Imagen inválida o de tamaño distinto: {img_path}")
        vw.release()

        degrader = VideoDegrader(str(temp_video))

        output_prefix = f"{escena.name}_style"
        csv_filename = f"{escena.name}.csv"

        degrader.generate_versions(
            n=num_versiones,
            output_prefix=output_prefix,
            csv_path=csv_filename,
            description=f"Degradación automática para {escena.name}",
            output_dir=str(carpeta_videos)
        )

        # -- GUARDAR FOTOGRAMAS DEGRADADOS (primera versión por defecto) --
        print(f"   🖼️ Guardando fotogramas degradados en {carpeta_fotogramas/escena.name}")
        carpeta_out_escena = carpeta_fotogramas / escena.name
        carpeta_out_escena.mkdir(exist_ok=True, parents=True)
        # Abrir el primer video degradado generado (puedes cambiar el índice si quieres otra versión)
        degradado_path = carpeta_videos / f"{output_prefix}_001.mp4"
        cap = cv2.VideoCapture(str(degradado_path))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(carpeta_out_escena / f"{frame_idx:05d}.jpg"), frame)
            frame_idx += 1
        cap.release()

        print(f"✅ Finalizado: {escena.name} → {output_prefix}_###.mp4 + {csv_filename} + {frame_idx} fotogramas degradados")

        # Elimina el archivo de video temporal
        try:
            temp_video.unlink()
        except Exception as e:
            print(f"[WARN] No se pudo borrar {temp_video}: {e}")

    print("🎬 Todas las escenas han sido degradadas y exportadas como videos y fotogramas.")



# ------------------------
# EJECUCIÓN PRINCIPAL
# ------------------------
# === MODO PRINCIPAL CON MENÚ INTERACTIVO SIMPLE ===

def main():
    print("Selecciona modo de operación:")
    print("1. Procesar DAVIS automáticamente")
    print("2. Procesar carpeta personalizada")
    modo = input("Elige 1 o 2: ").strip()

    if modo == "1":
        davis_paths = download_davis_dataset()
        annotations_base = davis_paths["Annotations"]
        images_base = davis_paths["JPEGImages"]

        output_root = "video_variaciones"
        os.makedirs(output_root, exist_ok=True)

        # Leer subcarpetas desde la carpeta de anotaciones
        video_names = [f.name for f in sorted(Path(annotations_base).iterdir()) if f.is_dir()]
        print(f"🔍 Se encontraron {len(video_names)} secuencias en Annotations/480p")

        for name in video_names:
            print(f"🎞️ Procesando secuencia: {name}")

            video_dir = os.path.join(images_base, name)
            frame_pattern = os.path.join(video_dir, "%05d.jpg")

            if not Path(video_dir).exists():
                print(f"⚠️ Carpeta de imágenes no encontrada para {name}, se omite.")
                continue

            degrader = VideoDegrader(frame_pattern)

            output_prefix = f"{name}_style"
            csv_filename = f"{name}.csv"

            degrader.generate_versions(
                n=1,
                output_prefix=output_prefix,
                csv_path=csv_filename,
                description=f"Degradación automática para {name}",
                output_dir=output_root
            )

            print(f"✅ Finalizado: {name} → {output_prefix}_###.mp4 + {csv_filename}")

        print("🎬 Todas las secuencias DAVIS han sido degradadas.")

    elif modo == "2":
        carpeta_entrada = input(
            "Ruta a la carpeta raíz de escenas (subcarpetas):\n"
            "Ejemplo: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/ORIGINALES\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/ORIGINALES"
        
        carpeta_videos = input(
            "Carpeta de salida para videos y CSV:\n"
            "Ejemplo: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/Videos\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/Videos"
        
        carpeta_fotogramas = input(
            "Carpeta de salida para fotogramas degradados (uno por escena):\n"
            "Ejemplo: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/Fotogramas\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/Fotogramas"
        
        num_versiones = int(
            input("¿Cuántas versiones degradadas por escena quieres generar? [default=1]: ") or "1"
        )
        procesar_carpeta_personalizada(
            carpeta_entrada, carpeta_videos, carpeta_fotogramas, num_versiones=num_versiones
        )

    else:
        print("Opción inválida. Ejecuta de nuevo el script.")

if __name__ == "__main__":
    main()