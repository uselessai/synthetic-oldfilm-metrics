import os
from glob import glob
import numpy as np
import cv2
from PIL import Image
import sys

# Agrega la ruta del módulo util.py a PYTHONPATH
sys.path.append("/home/laura/CycleGAN/RRTN-old-film-restoration")

# Importa la función de degradación
from VP_code.data.Data_Degradation.util import degradation_video_list_4

# Rutas base
REDS_ROOT = "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/ORIGINALES"
TEXTURE_ROOT = "/home/laura/CycleGAN/RRTN-old-film-restoration/texture_templates/noise_data"
OUTPUT_ROOT = "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN"

def process_scene(scene_folder):
    img_paths = sorted(glob(os.path.join(scene_folder, "*.png")))
    imgs = [cv2.imread(p).astype(np.float32)/255.0 for p in img_paths]
    degraded, gt = degradation_video_list_4(imgs, texture_url=TEXTURE_ROOT)
    return img_paths, degraded, gt

def main():
    for split in ("train", "val"):
        split_in = os.path.join(REDS_ROOT, split)
        split_out_lq = os.path.join(OUTPUT_ROOT, "lq", split)
        split_out_gt = os.path.join(OUTPUT_ROOT, "gt", split)

        for scene_name in sorted(os.listdir(split_in)):
            scene_in = os.path.join(split_in, scene_name)
            if not os.path.isdir(scene_in): continue

            scene_lq = os.path.join(split_out_lq, scene_name)
            scene_gt = os.path.join(split_out_gt, scene_name)
            os.makedirs(scene_lq, exist_ok=True)
            os.makedirs(scene_gt, exist_ok=True)

            paths, lq_imgs, gt_imgs = process_scene(scene_in)
            for path, lq, gt in zip(paths, lq_imgs, gt_imgs):
                name = os.path.basename(path)
                cv2.imwrite(os.path.join(scene_lq, name), (lq*255).astype(np.uint8))
                cv2.imwrite(os.path.join(scene_gt, name), (gt*255).astype(np.uint8))

            print(f"[✓] Procesado {split}/{scene_name}: {len(paths)} frames")

if __name__ == "__main__":
    main()
