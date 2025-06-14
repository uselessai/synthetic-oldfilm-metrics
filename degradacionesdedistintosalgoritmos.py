import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

# Carpetas
input_root = Path("/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/train")
output_root = Path("/home/laura/CycleGAN/00Databases/REDS/COMPARACION/ALGORITMO")
output_root.mkdir(parents=True, exist_ok=True)

# Funciones de degradación (por paper)
def add_gaussian_noise(image, sigma=25):  # UIR-LoRA, Wan et al.
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_blur(image, ksize=5):  # VRT, Wan et al., UIR-LoRA
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def compress_jpeg(image, quality=40):  # Wan et al., UIR-LoRA
    _, encimg = cv2.imencode('.png', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encimg, 1)

def adjust_brightness_contrast(image, brightness=1.2, contrast=0.9):  # Wan et al.
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def downsample_upsample(image, scale=0.5):  # Wan et al., VRT
    h, w = image.shape[:2]
    down = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return up

# Proceso por escena
for scene_dir in input_root.iterdir():
    if scene_dir.is_dir():
        output_scene = output_root / scene_dir.name
        output_scene.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(scene_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            degraded = img.copy()
            degraded = add_gaussian_noise(degraded)
            degraded = apply_blur(degraded)
            degraded = compress_jpeg(degraded)
            degraded = adjust_brightness_contrast(degraded)
            degraded = downsample_upsample(degraded)

            cv2.imwrite(str(output_scene / img_path.name), degraded)

print("Degradación completa. Archivos guardados en:", output_root)
