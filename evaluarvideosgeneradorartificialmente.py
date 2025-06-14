import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
import torch
import lpips
import piq
import torchvision.transforms as T
from PIL import Image

from basicsr.metrics.niqe import calculate_niqe


# Detectar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Paths a las carpetas
paths = {
    "reales_antiguas": Path("/home/laura/CycleGAN/00Databases/escenas_videosantiguos"),
    "degradadas_propias": Path("/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/Fotogramas"),
    "degradadas_RRTN": Path("/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/lq/train"),
    "degradadas_algoritmo": Path("/home/laura/CycleGAN/00Databases/REDS/COMPARACION/ALGORITMO"),
    "real_calidad": Path("/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/train"),
    
    
    
}


# Inicializar LPIPS en GPU si hay
loss_fn = lpips.LPIPS(net='alex').to(device)
to_tensor = T.ToTensor()

# Funciones auxiliares
def load_gray(p): return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
def load_rgb(p): 
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def entropy(img): return shannon_entropy(img)

def flicker(seq):
    if len(seq) < 2: return None
    means = [np.mean(f) for f in seq]
    return np.std(np.abs(np.diff(means))) / (np.mean(means) + 1e-8)

def lpips_dist(a, b):
    t1 = torch.tensor(a).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    t2 = torch.tensor(b).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        return loss_fn(t1, t2).item()


def niqe_score(img):
    """
    Calcula el puntaje NIQE de una imagen. Convierte a RGB si es necesario.

    Args:
        img (np.ndarray): Imagen en formato numpy, tipo uint8, puede ser en escala de grises o RGB.

    Returns:
        float: Puntaje NIQE (menor es mejor calidad).
    """
    #print("[DEBUG] Iniciando cálculo de NIQE")

    if img is None:
        #print("[ERROR] La imagen es None")
        raise ValueError("La imagen es None")

    #print(f"[DEBUG] Tipo de dato: {img.dtype}, forma: {img.shape}")

    if img.dtype != np.uint8:
        #print("[ERROR] Tipo de dato incorrecto")
        raise ValueError("La imagen debe tener dtype=uint8")

    if img.ndim == 2:
        #print("[DEBUG] Imagen en escala de grises detectada, expandiendo a 3 canales")
        img = np.expand_dims(img, axis=2)

    if img.shape[2] == 1:
        #print("[DEBUG] Imagen con 1 canal, replicando para crear RGB")
        img = np.repeat(img, 3, axis=2)

    if img.shape[2] == 4:
        #print("[DEBUG] Imagen con 4 canales (RGBA), descartando canal alfa")
        img = img[:, :, :3]

    if img.shape[2] != 3:
        #print(f"[ERROR] La imagen no tiene 3 canales después de procesar: tiene {img.shape[2]}")
        raise ValueError(f"La imagen debe tener 3 canales después de la conversión, pero tiene {img.shape[2]}")

    #print("[DEBUG] Normalizando imagen a float32 en [0, 1]")
    img = img.astype(np.float32) / 255.0

    #print("[DEBUG] Llamando a calculate_niqe()")
    score = calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y')

    #print(f"[DEBUG] Puntaje NIQE: {score}")
    return score

def brisque(img):
    if img.ndim == 2:
        img = Image.fromarray(img.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
    return piq.brisque(to_tensor(img).unsqueeze(0), data_range=1.0).item()

# Evaluación por video
def collect_stats_per_video(video_root, domain_name):
    results = []
    for scene in sorted(video_root.iterdir()):
        if not scene.is_dir():
            continue
        print(f"[{domain_name.upper()}] Procesando video: {scene.name}")

        frames = sorted([f for f in scene.glob("*") if f.suffix.lower() in [".jpg", ".png"]])

        if not frames:
            print(f"[{domain_name.upper()}] {scene.name}: sin fotogramas")
            continue

        gray_seq = [load_gray(f) for f in frames if load_gray(f) is not None]
        rgb_seq = [load_rgb(f) for f in frames if load_rgb(f) is not None]

        fi = flicker(gray_seq)

        lpips_vals = []
        for i in range(len(rgb_seq) - 1):
            if rgb_seq[i] is not None and rgb_seq[i + 1] is not None:
                try:
                    lpips_vals.append(lpips_dist(rgb_seq[i], rgb_seq[i + 1]))
                except Exception as e:
                    print(f"[LPIPS ERROR] {scene.name}, frame {i}: {e}")

        brisques = [brisque(r) for r in rgb_seq if r is not None]
        entropies = [entropy(g) for g in gray_seq if g is not None]
        niqes = []
        for i, r in enumerate(rgb_seq):
            if r is not None:
                try:
                    score = niqe_score(r)
                    #print(f"[NIQE] Frame {i} → {score}")
                    niqes.append(score)
                except Exception as e:
                    print(f"[ERROR] Falló el NIQE en frame {i}: {e}")
            else:
                print(f"[WARNING] Frame {i} es None y fue ignorado.")


        results.append({
            "domain": domain_name,
            "scene": scene.name,
            "frames": len(frames),
            "lpips_mean": np.mean(lpips_vals) if lpips_vals else None,
            "brisque_mean": np.mean(brisques) if brisques else None,
            "entropy_mean": np.mean(entropies) if entropies else None,
            "flicker_index": fi,
            "niqe_mean": np.mean(niqes) if niqes else None,
            "niqe_std": np.std(niqes) if niqes else None,

        })

    return results

# Ejecutar evaluación por grupo
all_metrics = []
for key in paths:
    all_metrics += collect_stats_per_video(paths[key], key)

# Guardar CSV
df = pd.DataFrame(all_metrics)
df.to_csv("comparacion_por_similitud_con_reales.csv", index=False)
print("\n✅ Resultados guardados en: comparacion_por_similitud_con_reales.csv")
