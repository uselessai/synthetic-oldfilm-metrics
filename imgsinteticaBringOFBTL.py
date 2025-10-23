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
import math
import json
import zlib
import shutil


# ===============================
# PRESETS SOLO CON OVERLAYS EXPORTABLES
# ===============================
OVERLAY_PRESETS = {
"scratches_proc_mixed": {
        "proc_scratches": True,
        "params": {
        "num_bundles": 1,                 # menos grupos
        "angle_range": (-60, 60),         # ángulos aleatorios amplios
        "spacing_range": (40, 80),        # más espacio entre rayas
        "width_range": (1, 2),            # finas
        "intensity_range": (0.7, 0.95),   # bastante visibles
        "polarity_white_prob": 0.5,       # mitad blancas, mitad negras
        "dash_freq_range": (70.0, 100.0), # rayas casi continuas
        "dash_depth_range": (0.05, 0.15), # poco dash
        "noise_strength_range": (0.1, 0.2),
        "blur_ksize_range": (3, 5),       # blur pequeño
        "blur_sigma_range": (0.3, 0.6),
        "extras_count_range": (5, 10),    # algunas rayas sueltas
        "extras_length_range": (40, 100),
        "warp_amplitude_range": (0.9, 2.6),
        "warp_sigma_range": (9.0, 18.0),
        "edge_halo_strength_range": (0.14, 0.26),
        "core_gamma_range": (0.95, 1.25),
        "local_variation_strength": 0.4,
        "seed": None
    }
    },
 # solo ejemplos; ajusta a tu gusto
     "scratches_proc_light": {
        "proc_scratches": True,
        "params": {
            "num_bundles": 1,
            "angle_range": (-40, 40),
            "spacing_range": (30, 50),
            "width_range": (1, 2),
            "intensity_range": (0.35, 0.6),
            "polarity_white_prob": 0.6,
            "extras_count_range": (15, 35),
            "warp_amplitude_range": (0.4, 1.4),
            "warp_sigma_range": (8.0, 16.0),
            "edge_halo_strength_range": (0.1, 0.2),
            "core_gamma_range": (0.9, 1.15),
            "local_variation_strength": 0.25,
        }
    },
    "scratches_proc_medium": {
        "proc_scratches": True,
        "params": {
            "num_bundles": 2,
            "angle_range": (-45, 45),
            "spacing_range": (24, 40),
            "width_range": (1, 2),
            "intensity_range": (0.45, 0.75),
            "polarity_white_prob": 0.6,
            "extras_count_range": (25, 60),
            "warp_amplitude_range": (0.6, 2.0),
            "warp_sigma_range": (9.0, 19.0),
            "edge_halo_strength_range": (0.12, 0.24),
            "core_gamma_range": (0.95, 1.25),
            "local_variation_strength": 0.35,
        }
    },
    "scratches_proc_grid": {
        "proc_scratches": True,
        "params": {
            "num_bundles": 2,                 # dos familias “cruzadas”
            "angle_range": (-35, 65),         # deja rango amplio para que crucen
            "spacing_range": (18, 32),
            "width_range": (1, 2),
            "intensity_range": (0.55, 0.8),
            "dash_freq_range": (18.0, 24.0),
            "dash_depth_range": (0.35, 0.5),
            "noise_strength_range": (0.4, 0.6),
            "blur_ksize_range": (7, 11),
            "blur_sigma_range": (1.0, 1.6),
            "extras_count_range": (80, 140),
            "extras_length_range": (20, 60),
            "polarity_white_prob": 0.6,
            "warp_amplitude_range": (0.5, 1.6),
            "warp_sigma_range": (10.0, 22.0),
            "edge_halo_strength_range": (0.15, 0.3),
            "core_gamma_range": (1.05, 1.35),
            "local_variation_strength": 0.45,
        }
    },

    "vignette_soft": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"vignette_alpha": 0.6, "vignette_strength": 12}
    },
    "vignette_strong": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"vignette_alpha": 0.9, "vignette_strength": 20}
    },
    "scratches_light": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 1.0,
        "scratch_count": (3, 6),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"scratch_alpha": 0.3}
    },
    "scratches_heavy": {
        "era": "1920s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 1.0,
        "scratch_count": (12, 20),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"scratch_alpha": 0.8}
    },
    "blobs_soft": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"blob_intensity_boost": 1.5, "blob_size": (20, 60)}
    },
    "blobs_large": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"blob_intensity_boost": 3.0, "blob_size": (60, 120)}
    },
    "burn_small": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"burn_prob": 1.0, "burn_intensity": 0.3}
    },
    "burn_large": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"burn_prob": 1.0, "burn_intensity": 0.6}
    },
    "combo_scratches_blobs": {
        "era": "1920s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 1.0,
        "scratch_count": (5, 10),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"scratch_alpha": 0.5, "blob_size": (30, 80)}
    },
    "combo_vignette_burn": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"vignette_alpha": 0.8, "vignette_strength": 15, "burn_prob": 1.0, "burn_intensity": 0.4}
    },
    "combo_vignette_blobs_soft": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.7,
            "vignette_strength": 14,
            "blob_size": (18, 45)
        }
    },
    "combo_vignette_blobs_heavy": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.85,
            "vignette_strength": 20,
            "blob_size": (35, 90)
        }
    },
    "combo_blobs_burn_soft": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "burn_prob": 1.0,
            "burn_intensity": 0.3,
            "blob_size": (22, 60)
        }
    },
    "combo_blobs_burn_intense": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "burn_prob": 1.0,
            "burn_intensity": 0.55,
            "blob_size": (40, 110)
        }
    },
    "combo_vignette_blobs_burn": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.82,
            "vignette_strength": 18,
            "burn_prob": 1.0,
            "burn_intensity": 0.42,
            "blob_size": (26, 75)
        }
    },
    "combo_vignette_blobs_burn_sepia": {
        "era": "1920s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "sepia",
        "color_tone_strength": 0.65,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.78,
            "vignette_strength": 22,
            "burn_prob": 1.0,
            "burn_intensity": 0.38,
            "blob_size": (24, 68)
        }
    },
    "combo_vignette_blobs_burn_blue": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "blue",
        "color_tone_strength": 0.6,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.8,
            "vignette_strength": 16,
            "burn_prob": 1.0,
            "burn_intensity": 0.35,
            "blob_size": (20, 60)
        }
    },
    "combo_vignette_blobs_burn_rose": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "rose",
        "color_tone_strength": 0.55,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.76,
            "vignette_strength": 17,
            "burn_prob": 1.0,
            "burn_intensity": 0.4,
            "blob_size": (20, 65)
        }
    },
    "combo_soft_bw_vignette_blobs": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": True,
        "override_profile": {
            "vignette_alpha": 0.72,
            "vignette_strength": 21,
            "blob_size": (18, 52)
        }
    },
    "combo_clean_bw_vignette_burn": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 0.6,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": True,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.88,
            "vignette_strength": 19,
            "burn_prob": 1.0,
            "burn_intensity": 0.48,
            "blob_size": (16, 42)
        }
    },
    "combo_vignette_burn_double": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 0.3,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.86,
            "vignette_strength": 24,
            "burn_prob": 1.0,
            "burn_intensity": 0.6
        }
    },
    "combo_blue_softbw_blobs": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "blue",
        "color_tone_strength": 0.55,
        "apply_clean_bw": False,
        "apply_soft_bw": True,
        "override_profile": {
            "blob_size": (25, 70)
        }
    },
    "combo_rose_cleanbw_burn": {
        "era": "1920s",
        "apply_vignette_globally": True,
        "blob_prob": 0.4,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "rose",
        "color_tone_strength": 0.6,
        "apply_clean_bw": True,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.68,
            "vignette_strength": 23,
            "burn_prob": 1.0,
            "burn_intensity": 0.32,
            "blob_size": (18, 50)
        }
    },
    "combo_scratches_vignette_soft": {
        "era": "1930s",
        "apply_vignette_globally": True,
        "blob_prob": 0.0,
        "scratch_prob": 1.0,
        "scratch_count": (8, 14),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.74,
            "vignette_strength": 19,
            "scratch_alpha": 0.55
        },
        "proc_scratches": True,
        "params": {
            "num_bundles": 2,
            "angle_range": (-28, 28),
            "spacing_range": (18, 34),
            "spacing_jitter_range": (0.18, 0.42),
            "width_range": (1, 2),
            "intensity_range": (0.42, 0.68),
            "dash_freq_range": (22.0, 34.0),
            "dash_depth_range": (0.18, 0.38),
            "noise_strength_range": (0.25, 0.45),
            "blur_ksize_range": (5, 9),
            "blur_sigma_range": (0.8, 1.3),
            "longitudinal_sigma_range": (12.0, 28.0),
            "longitudinal_dropout_range": (0.08, 0.2),
            "longitudinal_variation_strength_range": (0.32, 0.55),
            "warp_amplitude_range": (0.4, 1.5),
            "warp_sigma_range": (8.0, 16.0),
            "edge_halo_strength_range": (0.12, 0.22),
            "core_gamma_range": (0.95, 1.18),
            "extras_count_range": (16, 42),
            "extras_length_range": (24, 62)
        }
    },
    "combo_scratches_vignette_burn": {
        "era": "1920s",
        "apply_vignette_globally": True,
        "blob_prob": 0.0,
        "scratch_prob": 1.0,
        "scratch_count": (10, 18),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.82,
            "vignette_strength": 21,
            "burn_prob": 1.0,
            "burn_intensity": 0.36,
            "scratch_alpha": 0.62
        },
        "proc_scratches": True,
        "params": {
            "num_bundles": 3,
            "angle_range": (-32, 32),
            "spacing_range": (16, 30),
            "spacing_jitter_range": (0.22, 0.46),
            "width_range": (1, 2),
            "intensity_range": (0.48, 0.78),
            "dash_freq_range": (20.0, 32.0),
            "dash_depth_range": (0.22, 0.42),
            "noise_strength_range": (0.3, 0.5),
            "blur_ksize_range": (5, 9),
            "blur_sigma_range": (0.9, 1.45),
            "longitudinal_sigma_range": (14.0, 30.0),
            "longitudinal_dropout_range": (0.1, 0.24),
            "longitudinal_variation_strength_range": (0.35, 0.6),
            "warp_amplitude_range": (0.5, 1.8),
            "warp_sigma_range": (9.0, 18.0),
            "edge_halo_strength_range": (0.16, 0.26),
            "core_gamma_range": (0.98, 1.24),
            "extras_count_range": (20, 48),
            "extras_length_range": (28, 70)
        }
    },
    "combo_scratches_blobs_vignette": {
        "era": "1940s",
        "apply_vignette_globally": True,
        "blob_prob": 1.0,
        "scratch_prob": 1.0,
        "scratch_count": (6, 12),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.78,
            "vignette_strength": 18,
            "scratch_alpha": 0.52,
            "blob_size": (22, 58)
        },
        "proc_scratches": True,
        "params": {
            "num_bundles": 2,
            "angle_range": (-26, 26),
            "spacing_range": (20, 38),
            "spacing_jitter_range": (0.16, 0.4),
            "width_range": (1, 2),
            "intensity_range": (0.4, 0.68),
            "dash_freq_range": (22.0, 34.0),
            "dash_depth_range": (0.2, 0.4),
            "noise_strength_range": (0.28, 0.48),
            "blur_ksize_range": (5, 9),
            "blur_sigma_range": (0.9, 1.35),
            "longitudinal_sigma_range": (10.0, 26.0),
            "longitudinal_dropout_range": (0.1, 0.22),
            "longitudinal_variation_strength_range": (0.32, 0.56),
            "warp_amplitude_range": (0.45, 1.6),
            "warp_sigma_range": (8.0, 16.0),
            "edge_halo_strength_range": (0.14, 0.24),
            "core_gamma_range": (0.95, 1.18),
            "extras_count_range": (18, 44),
            "extras_length_range": (26, 66)
        }
    },
    "combo_scratches_blobs_burn": {
        "era": "1920s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,
        "scratch_prob": 1.0,
        "scratch_count": (7, 14),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "burn_prob": 1.0,
            "burn_intensity": 0.4,
            "scratch_alpha": 0.6,
            "blob_size": (26, 72)
        },
        "proc_scratches": True,
        "params": {
            "num_bundles": 3,
            "angle_range": (-35, 35),
            "spacing_range": (18, 34),
            "spacing_jitter_range": (0.2, 0.46),
            "width_range": (1, 2),
            "intensity_range": (0.45, 0.75),
            "dash_freq_range": (20.0, 30.0),
            "dash_depth_range": (0.22, 0.45),
            "noise_strength_range": (0.32, 0.52),
            "blur_ksize_range": (5, 9),
            "blur_sigma_range": (1.0, 1.5),
            "longitudinal_sigma_range": (14.0, 32.0),
            "longitudinal_dropout_range": (0.12, 0.26),
            "longitudinal_variation_strength_range": (0.36, 0.62),
            "warp_amplitude_range": (0.6, 2.0),
            "warp_sigma_range": (10.0, 18.0),
            "edge_halo_strength_range": (0.16, 0.28),
            "core_gamma_range": (1.0, 1.26),
            "extras_count_range": (22, 54),
            "extras_length_range": (30, 76)
        }
    },
    "combo_scratches_blue_leak": {
        "era": "1950s",
        "apply_vignette_globally": True,
        "blob_prob": 0.0,
        "scratch_prob": 1.0,
        "scratch_count": (6, 12),
        "color_tone": "blue",
        "color_tone_strength": 0.6,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {
            "vignette_alpha": 0.7,
            "vignette_strength": 17,
            "scratch_alpha": 0.5
        },
        "proc_scratches": True,
        "params": {
            "num_bundles": 2,
            "angle_range": (-24, 24),
            "spacing_range": (22, 40),
            "spacing_jitter_range": (0.14, 0.38),
            "width_range": (1, 2),
            "intensity_range": (0.38, 0.64),
            "dash_freq_range": (24.0, 36.0),
            "dash_depth_range": (0.18, 0.36),
            "noise_strength_range": (0.24, 0.42),
            "blur_ksize_range": (5, 9),
            "blur_sigma_range": (0.8, 1.2),
            "longitudinal_sigma_range": (10.0, 24.0),
            "longitudinal_dropout_range": (0.08, 0.2),
            "longitudinal_variation_strength_range": (0.3, 0.52),
            "warp_amplitude_range": (0.4, 1.4),
            "warp_sigma_range": (8.0, 14.0),
            "edge_halo_strength_range": (0.12, 0.22),
            "core_gamma_range": (0.95, 1.16),
            "extras_count_range": (14, 36),
            "extras_length_range": (22, 58)
        }
    }
}


SCRATCH_PRESET_ALIAS = {
    "only_scratches": "scratches_proc_mixed",
    "scratches_light": "scratches_proc_light",
    "scratches_heavy": "scratches_proc_mixed",
    "combo_scratches_blobs": "scratches_proc_medium",
    "scratches": "scratches_proc_medium",
    "scratches_flicker": "scratches_proc_mixed"
}

DEFAULT_PROC_SCRATCH_ALIAS = "scratches_proc_medium"


ATOMIC_PRESETS = {
    "only_vignette": {
        "era": "1950s",
        "apply_vignette_globally": True,  # viñeta siempre
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"vignette_alpha": 1.0, "vignette_strength": 25}
    },
    "only_scratches": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 1.0,      # siempre
        "scratch_count": (12, 20),# muchas rayas
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"scratch_alpha": 0.8}  # opacidad más fuerte
    },
    "only_flicker": {
        "era": "1920s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"flicker_intensity": 0.7}  # mucho parpadeo
    },
    "only_blobs": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 1.0,         # siempre
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"blob_intensity_boost": 3.0, "blob_size": (40, 120)}
    },
    "only_burn": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"burn_prob": 1.0, "burn_intensity": 0.5}
    },
    "only_soft_bw": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": True,
        "override_profile": {"soft_bw_blur": 2.5, "soft_bw_contrast": 1.2}
    },
    "only_clean_bw": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": True,
        "apply_soft_bw": False,
        "override_profile": {"clean_bw_contrast": 1.5, "clean_bw_brightness": 20}
    },
    "only_vhs": {
        "era": "vhs",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {"vhs_effect": True, "vhs_offset": 5, "vhs_chroma_noise": 30}
    },
    "only_sepia": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "sepia",
        "color_tone_strength": 1.0,   # máximo
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {}
    },
    "only_blue": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "blue",
        "color_tone_strength": 1.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {}
    },
    "only_rose": {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": "rose",
        "color_tone_strength": 1.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {}
    }
}



DEGRADATION_PRESETS = {
    "scratches": {
        "era": "1950s", "apply_vignette_globally": False, "blob_prob": 0.05,
        "scratch_prob": 0.95, "scratch_count": (3, 6),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {"flicker_intensity": 0.10}
    },
    "flicker": {
        "era": "1920s", "apply_vignette_globally": False, "blob_prob": 0.0,
        "scratch_prob": 0.0, "scratch_count": (0, 0),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {"flicker_intensity": 0.35}
    },
    "blobs": {
        "era": "1950s", "apply_vignette_globally": False, "blob_prob": 0.8,
        "scratch_prob": 0.1, "scratch_count": (1, 3),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {}
    },
    "vignette": {
        "era": "1950s", "apply_vignette_globally": True, "blob_prob": 0.0,
        "scratch_prob": 0.0, "scratch_count": (0, 0),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {}
    },
    "sepia": {
        "era": "1920s", "apply_vignette_globally": True, "blob_prob": 0.0,
        "scratch_prob": 0.1, "scratch_count": (1, 2),
        "color_tone": "sepia", "color_tone_strength": 0.5,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {}
    },
    "blue": {
        "era": "1950s", "apply_vignette_globally": True, "blob_prob": 0.0,
        "scratch_prob": 0.0, "scratch_count": (0, 0),
        "color_tone": "blue", "color_tone_strength": 0.45,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {}
    },
    "rose": {
        "era": "1950s", "apply_vignette_globally": True, "blob_prob": 0.0,
        "scratch_prob": 0.0, "scratch_count": (0, 0),
        "color_tone": "rose", "color_tone_strength": 0.35,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {}
    },
    "clean_bw": {
        "era": "1950s", "apply_vignette_globally": False, "blob_prob": 0.0,
        "scratch_prob": 0.1, "scratch_count": (1, 2),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": True, "apply_soft_bw": False,
        "override_profile": {}
    },
    "soft_bw": {
        "era": "1920s", "apply_vignette_globally": True, "blob_prob": 0.0,
        "scratch_prob": 0.0, "scratch_count": (0, 0),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": True,
        "override_profile": {}
    },
    "vhs": {
        "era": "vhs", "apply_vignette_globally": False, "blob_prob": 0.0,
        "scratch_prob": 0.0, "scratch_count": (0, 0),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {}
    },
    "burn": {
        "era": "1950s", "apply_vignette_globally": False, "blob_prob": 0.0,
        "scratch_prob": 0.1, "scratch_count": (1, 2),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {"burn_prob": 0.06}
    },
    "scratches_flicker": {
        "era": "1920s", "apply_vignette_globally": True, "blob_prob": 0.05,
        "scratch_prob": 0.9, "scratch_count": (3, 6),
        "color_tone": None, "color_tone_strength": 0.0,
        "apply_clean_bw": False, "apply_soft_bw": False,
        "override_profile": {"flicker_intensity": 0.3}
    },
}


# ===============================
# DATASETS DEFINIDOS (5 combos)
# ===============================
FIXED_DATASETS = {
    "CORE_20s": "custom:scratches+flicker+sepia+vignette",
    "DIRT_BW": "custom:blobs+vignette+soft_bw",
    "CLEANBW_SCR": "custom:clean_bw+scratches",
    "COLOR_LEAK": "custom:burn+blue+vignette",
    "VHS_TRANSFER": "custom:vhs+flicker"
}


# -------------------------
# GENERADOR EXTERNO DE CAPAS DE ARTEFACTOS (OVERLAYS PNG RGBA)
# -------------------------
def procesar_carpeta_overlays(
    carpeta_entrada,
    carpeta_salida,
    tipos=None,
    presets=None,
    export_rrtn_texture=False,
    only_rrtn_texture=False,
    rrtn_background=245.0,
    rrtn_darkness=200.0,
    rrtn_alpha_gamma=0.75,
    rrtn_blur_sigma=0.0,
    rrtn_noise_sigma=0.0
):
    """
    Recorre escenas (subcarpetas) como en la opción 4,
    pero en vez de degradar genera overlays RGBA + CSV.
    """
    if presets is None:
        presets = ATOMIC_PRESETS
    if tipos is None:
        tipos = list(presets.keys())

    if only_rrtn_texture:
        export_rrtn_texture = True

    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(exist_ok=True, parents=True)

    escenas = [f for f in carpeta_entrada.iterdir() if f.is_dir()]
    print(f"🔍 Se encontraron {len(escenas)} escenas en {carpeta_entrada}")

    for escena in escenas:
        print(f"🎞️ Procesando escena: {escena.name}")
        fotogramas = sorted(list(escena.glob("*.jpg")) + list(escena.glob("*.png")))
        if not fotogramas:
            print(f"⚠️ No se encontraron imágenes en {escena}, se omite.")
            continue

        # Carpeta de salida para esta escena
        carpeta_out = carpeta_salida / escena.name
        carpeta_out.mkdir(parents=True, exist_ok=True)

        # Tomamos el primer frame como base
        frame0 = fotogramas[0]

        # Generar overlays (reutilizamos la función hecha antes)
        csv_out = generar_overlays_imagen(
            frame0,
            output_dir=carpeta_out,
            presets=presets,
            scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches"),
            csv_name=f"{escena.name}.csv",
            export_rrtn_texture=export_rrtn_texture,
            only_rrtn_texture=only_rrtn_texture,
            rrtn_background=rrtn_background,
            rrtn_darkness=rrtn_darkness,
            rrtn_alpha_gamma=rrtn_alpha_gamma,
            rrtn_blur_sigma=rrtn_blur_sigma,
            rrtn_noise_sigma=rrtn_noise_sigma
        )

        print(f"✅ Escena {escena.name} → overlays en {carpeta_out}, CSV: {csv_out}")

    print("🎬 Todas las escenas han sido procesadas en modo overlays.")


def procesar_carpeta_overlays_coherentes(
    carpeta_entrada,
    carpeta_salida,
    scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches"),
    include_vignette=True,
    include_scratches=True,
    include_blobs=True,
    target_size=None,
    guardar_metadata=True,
    export_rrtn_texture=False,
    only_rrtn_texture=False,
    rrtn_background=245.0,
    rrtn_darkness=200.0,
    rrtn_alpha_gamma=0.75,
    rrtn_blur_sigma=0.0,
    rrtn_noise_sigma=0.0
):
    """
    Genera una plantilla RGBA por fotograma combinando artefactos seleccionados.
    - Mantiene la viñeta coherente por escena.
    - Permite variaciones suaves de arañazos/blobs por fotograma.
    - Guarda la salida en subcarpetas numéricas correlativas bajo carpeta_salida.
    """
    if only_rrtn_texture:
        export_rrtn_texture = True

    carpeta_entrada = Path(carpeta_entrada)
    carpeta_salida = Path(carpeta_salida)
    scratch_dir = Path(scratch_dir)

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    scratch_proc_presets = [
        (name, data["params"])
        for name, data in OVERLAY_PRESETS.items()
        if isinstance(data, dict) and data.get("proc_scratches") and isinstance(data.get("params"), dict)
    ]
    if not scratch_proc_presets:
        scratch_proc_presets = [("procedural_default", {})]

    escenas = sorted([f for f in carpeta_entrada.iterdir() if f.is_dir()])
    if not escenas:
        print(f"⚠️ No se encontraron escenas en {carpeta_entrada}")
        return

    existentes = []
    for d in carpeta_salida.iterdir():
        if d.is_dir() and d.name.isdigit():
            try:
                existentes.append(int(d.name))
            except ValueError:
                continue
    siguiente_indice = max(existentes, default=0) + 1

    print(f"🔍 Escenas detectadas: {len(escenas)}")

    for escena in escenas:
        print(f"\n🎞️ Generando plantillas para escena: {escena.name}")
        fotogramas = sorted(list(escena.glob("*.png")) + list(escena.glob("*.jpg")))
        if not fotogramas:
            print(f"   ⚠️ No se encontraron fotogramas en {escena}, se omite.")
            continue

        frame0 = cv2.imread(str(fotogramas[0]), cv2.IMREAD_UNCHANGED)
        if frame0 is None:
            print(f"   ⚠️ No se pudo leer {fotogramas[0].name}, se omite.")
            continue

        if frame0.ndim == 2:
            height, width = frame0.shape
        else:
            height, width = frame0.shape[:2]

        if target_size is not None:
            height, width = target_size

        white_rgb = np.ones((height, width, 3), dtype=np.float32) * 255.0
        black_rgb = np.zeros((height, width, 3), dtype=np.float32)

        escena_rel = str(escena.relative_to(carpeta_entrada))
        escena_seed = zlib.crc32(escena_rel.encode("utf-8")) & 0xFFFFFFFF
        rng_escena = random.Random(escena_seed)

        vignette_mask = None
        vignette_params = None
        if include_vignette:
            apply_vignette_scene = rng_escena.random() < 0.60
            if apply_vignette_scene:
                vignette_strength = rng_escena.uniform(18.0, 26.0)
                vignette_alpha = rng_escena.uniform(0.55, 0.9)
                vignette_mask = _make_vignette_mask(height, width, strength=vignette_strength)
                vignette_params = {
                    "alpha": float(vignette_alpha),
                    "strength": float(vignette_strength),
                    "enabled": True
                }
            else:
                vignette_params = {"enabled": False}

        scratch_params = None
        scratch_state = {
            "rgba": None,
            "alpha": None,
            "rgb": None,
            "preset": None,
            "seed": None,
            "meta": None
        }
        if include_scratches:
            scratch_alpha = rng_escena.uniform(0.35, 0.65)
            scratch_dropout = rng_escena.uniform(0.04, 0.16)
            scratch_shift = rng_escena.randint(0, 2)
            scratch_sigma = rng_escena.uniform(0.6, 1.8)
            scratch_params = {
                "alpha": float(min(max(scratch_alpha, 0.05), 0.95)),
                "dropout": float(max(scratch_dropout, 0.0)),
                "shift_pixels": int(max(scratch_shift, 0)),
                "blur_sigma": float(max(scratch_sigma, 0.0)),
                "refresh_prob": float(rng_escena.uniform(0.18, 0.35)),
                "absence_prob": float(rng_escena.uniform(0.08, 0.25)),
                "last_preset": None,
                "last_seed": None,
                "details": None
            }

        blob_alpha_base = None
        blob_color_base = None
        blob_params = None
        if include_blobs:
            blob_count = rng_escena.randint(6, 14)
            blob_min = rng_escena.randint(14, 32)
            blob_max = rng_escena.randint(blob_min + 4, blob_min + 28)
            blob_alpha_base, blob_color_base = _make_blobs_mask(
                height, width,
                count=blob_count,
                size_range=(blob_min, blob_max),
                polarity=None,
                rng=rng_escena
            )
            blob_params = {
                "count": int(blob_count),
                "size_min": int(blob_min),
                "size_max": int(blob_max)
            }

        carpeta_escena = carpeta_salida / f"{siguiente_indice:04d}"
        while carpeta_escena.exists():
            siguiente_indice += 1
            carpeta_escena = carpeta_salida / f"{siguiente_indice:04d}"
        carpeta_escena.mkdir(parents=True, exist_ok=True)
        indice_actual = siguiente_indice
        siguiente_indice += 1

        total_frames = len(fotogramas)
        for idx, frame_path in enumerate(tqdm(fotogramas, desc=f"{escena.name}", leave=False)):
            frame_seed = zlib.crc32(f"{escena_seed}_{idx}".encode("utf-8")) & 0xFFFFFFFF
            rng_frame = np.random.default_rng(frame_seed)

            overlay = np.zeros((height, width, 4), dtype=np.float32)

            if vignette_mask is not None and vignette_params and vignette_params.get("enabled", True):
                overlay = _overlay_max_alpha(
                    overlay,
                    black_rgb,
                    vignette_mask[..., None] * vignette_params["alpha"]
                )

            if scratch_params:
                refresh_needed = (
                    scratch_state["rgba"] is None
                    or rng_frame.random() < scratch_params["refresh_prob"]
                )
                if refresh_needed:
                    if rng_frame.random() < scratch_params["absence_prob"]:
                        scratch_state.update({"rgba": None, "alpha": None, "rgb": None, "preset": None, "seed": None, "meta": None})
                        scratch_params["last_preset"] = None
                        scratch_params["last_seed"] = None
                        scratch_params["details"] = None
                    else:
                        preset_index = int(rng_frame.integers(0, len(scratch_proc_presets)))
                        preset_name, preset_params = scratch_proc_presets[preset_index]
                        scratch_kwargs = dict(preset_params)
                        scratch_kwargs.pop("seed", None)
                        scratch_seed = int(rng_frame.integers(0, 2**31 - 1))
                        scratch_overlay_rgba, scratch_meta = generate_scratches_overlay(
                            height, width, seed=scratch_seed, **scratch_kwargs
                        )
                        scratch_rgba = scratch_overlay_rgba.astype(np.float32)
                        scratch_state.update({
                            "rgba": scratch_rgba,
                            "alpha": np.clip(scratch_rgba[..., 3] / 255.0, 0.0, 1.0),
                            "rgb": scratch_rgba[..., :3],
                            "preset": preset_name,
                            "seed": scratch_seed,
                            "meta": scratch_meta
                        })
                        scratch_params["last_preset"] = preset_name
                        scratch_params["last_seed"] = scratch_seed
                        scratch_params["details"] = scratch_meta

            if (
                scratch_state["alpha"] is not None
                and scratch_state["rgb"] is not None
                and scratch_params
                and scratch_state["alpha"].max() > 0
            ):
                scratch_alpha_map = scratch_state["alpha"].copy()
                scratch_rgb_map = scratch_state["rgb"].copy()
                shift = scratch_params["shift_pixels"]
                if shift > 0:
                    tx = int(rng_frame.integers(-shift, shift + 1))
                    ty = int(rng_frame.integers(-shift, shift + 1))
                    if tx != 0 or ty != 0:
                        M = np.float32([[1, 0, float(tx)], [0, 1, float(ty)]])
                        scratch_alpha_map = cv2.warpAffine(
                            scratch_alpha_map,
                            M,
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101
                        )
                        scratch_rgb_map = cv2.warpAffine(
                            scratch_rgb_map,
                            M,
                            (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101
                        )
                dropout = scratch_params["dropout"]
                if dropout > 0:
                    attenuation = 1.0 - dropout * rng_frame.random((height, width))
                    scratch_alpha_map = scratch_alpha_map * attenuation
                    scratch_rgb_map = scratch_rgb_map * attenuation[..., None]

                blur_sigma = scratch_params["blur_sigma"]
                if blur_sigma > 0:
                    scratch_alpha_map = cv2.GaussianBlur(scratch_alpha_map, (0, 0), blur_sigma)
                    scratch_rgb_map = cv2.GaussianBlur(scratch_rgb_map, (0, 0), blur_sigma)
                scratch_alpha_map = np.clip(scratch_alpha_map, 0.0, 1.0)
                scratch_rgb_map = np.clip(scratch_rgb_map, 0.0, 255.0)
                scratch_alpha_gain = min(
                    max(scratch_params["alpha"] * float(rng_frame.uniform(0.85, 1.15)), 0.02),
                    0.98
                )
                if scratch_alpha_map.max() > 1e-3:
                    overlay = _overlay_max_alpha(
                        overlay,
                        scratch_rgb_map,
                        scratch_alpha_map[..., None] * scratch_alpha_gain
                    )

            if blob_alpha_base is not None and blob_params and blob_alpha_base.max() > 0:
                blob_mask = np.clip(blob_alpha_base * float(rng_frame.uniform(0.8, 1.2)), 0.0, 1.0)
                overlay = _overlay_max_alpha(
                    overlay,
                    blob_color_base * 255.0,
                    blob_mask[..., None] * 0.6
                )
                overlay = _overlay_max_alpha(
                    overlay,
                    black_rgb,
                    blob_mask[..., None] * 0.18
                )

            overlay_rgb = np.clip(overlay[..., :3], 0.0, 255.0)
            overlay_alpha = np.clip(overlay[..., 3], 0.0, 1.0)
            overlay_u8 = np.zeros_like(overlay, dtype=np.uint8)
            overlay_u8[..., :3] = overlay_rgb.astype(np.uint8)
            overlay_u8[..., 3] = (overlay_alpha * 255.0).astype(np.uint8)

            out_path = None
            if not only_rrtn_texture:
                out_path = carpeta_escena / frame_path.name
                cv2.imwrite(str(out_path), overlay_u8)

            if export_rrtn_texture:
                rrtn_texture = _rgba_to_rrtn_texture(
                    overlay_u8,
                    background=rrtn_background,
                    darkness=rrtn_darkness,
                    alpha_gamma=rrtn_alpha_gamma,
                    blur_sigma=rrtn_blur_sigma,
                    noise_sigma=rrtn_noise_sigma
                )
                if out_path is not None:
                    rrtn_path = out_path if only_rrtn_texture else out_path.with_name(f"{out_path.stem}_rrtn.png")
                else:
                    base_name = frame_path.name
                    rrtn_path = carpeta_escena / base_name
                    if not only_rrtn_texture:
                        rrtn_path = rrtn_path.with_name(f"{Path(base_name).stem}_rrtn.png")
                cv2.imwrite(str(rrtn_path), rrtn_texture)

        if guardar_metadata:
            metadata = {
                "scene_name": escena.name,
                "scene_index": f"{indice_actual:04d}",
                "seed": int(escena_seed),
                "frame_count": int(total_frames),
                "width": int(width),
                "height": int(height),
                "artifacts": {}
            }
            if vignette_params:
                metadata["artifacts"]["vignette"] = vignette_params
            if scratch_params:
                metadata["artifacts"]["scratches"] = scratch_params
            if blob_params:
                metadata["artifacts"]["blobs"] = blob_params
            if export_rrtn_texture:
                metadata["artifacts"]["rrtn_texture"] = {
                    "background": rrtn_background,
                    "darkness": rrtn_darkness,
                    "alpha_gamma": rrtn_alpha_gamma,
                    "blur_sigma": rrtn_blur_sigma,
                    "noise_sigma": rrtn_noise_sigma,
                    "suffix": "" if only_rrtn_texture else "_rrtn.png"
                }

            with open(carpeta_escena / "metadata.json", "w") as fmeta:
                json.dump(metadata, fmeta, indent=2)

        print(f"   ✅ {total_frames} plantillas guardadas en {carpeta_escena}")

    print("\n🎬 Plantillas coherentes generadas para todas las escenas.")


def _color_from_tone(tone):
    # BGR aproximados, coherentes con tus funciones
    if tone == "sepia":
        return np.array([30, 66, 112], dtype=np.float32)
    elif tone == "blue":
        return np.array([170, 120, 60], dtype=np.float32)
    elif tone == "rose":
        return np.array([180, 150, 200], dtype=np.float32)
    return None

def _overlay_max_alpha(dst_rgba, src_rgb, src_alpha):
    """Compone un src (RGB + alpha escalar o mapa) sobre RGBA destino con 'max' en alfa y mezcla normal."""
    h, w = dst_rgba.shape[:2]
    if np.isscalar(src_alpha):
        A = np.full((h, w, 1), float(src_alpha), dtype=np.float32)
    else:
        A = src_alpha.astype(np.float32)
        if A.ndim == 2: A = A[..., None]
    C = src_rgb.astype(np.float32)
    # Porter-Duff "over": out = src*A + dst*(1-A). Aquí acumulamos y dejamos alpha como max(actual, A)
    dst_rgb = dst_rgba[..., :3].astype(np.float32)
    dst_a   = dst_rgba[..., 3:4].astype(np.float32)
    out_rgb = C * A + dst_rgb * (1.0 - A)
    out_a   = np.maximum(dst_a, A)
    out = dst_rgba.copy()
    out[..., :3] = out_rgb
    out[..., 3:4] = out_a
    return out

def _rgba_to_rrtn_texture(
    rgba_img,
    background=245.0,
    darkness=200.0,
    alpha_gamma=0.75,
    blur_sigma=0.0,
    noise_sigma=0.0
):
    """
    Convierte un overlay RGBA en una textura estilo RRTN (grises sin canal alfa).
    El alpha controla la oscuridad de las líneas sobre un fondo casi blanco.
    """
    if rgba_img.ndim != 3:
        raise ValueError("Se esperaba una imagen de 3 dimensiones para generar la textura estilo RRTN.")

    h, w = rgba_img.shape[:2]
    if rgba_img.shape[2] == 4:
        alpha = rgba_img[..., 3].astype(np.float32) / 255.0
        rgb = rgba_img[..., :3].astype(np.uint8)
    elif rgba_img.shape[2] == 3:
        alpha = np.ones((h, w), dtype=np.float32)
        rgb = rgba_img.astype(np.uint8)
    else:
        raise ValueError("La imagen debe tener 3 (RGB) o 4 canales (RGBA).")

    if alpha_gamma != 1.0:
        alpha = np.power(np.clip(alpha, 0.0, 1.0), float(alpha_gamma))
    else:
        alpha = np.clip(alpha, 0.0, 1.0)

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    contrast = 1.0 + 0.6 * (0.5 - gray)  # aumenta líneas oscuras, reduce las claras
    strength = np.clip(alpha * contrast, 0.0, 1.0) * float(darkness)
    texture = np.full((h, w), float(background), dtype=np.float32) - strength

    if blur_sigma and blur_sigma > 0:
        texture = cv2.GaussianBlur(texture, (0, 0), blur_sigma)

    if noise_sigma and noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, size=(h, w)).astype(np.float32)
        texture = texture + noise

    return np.clip(texture, 0, 255).astype(np.uint8)

def _make_vignette_mask(h, w, strength=15, blur_kernel=51):
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    nx = (x - cx) / max(cx, 1)
    ny = (y - cy) / max(cy, 1)
    r = np.sqrt(nx**2 + ny**2)
    mask = 1 - np.clip(r**strength, 0, 1)
    mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    return 1.0 - mask  # 0 centro, 1 bordes

def _make_scratch_mask_from_template(h, w, scratch_dir, rot=None, scale=None, flip=None, rng=None):
    rng = rng or random
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(Path(scratch_dir).glob(pattern))
    if not files:
        return np.zeros((h, w), dtype=np.float32), None
    path = rng.choice(files)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (w, h))
    if rot is None: rot = rng.uniform(0, 360)
    if scale is None: scale = rng.uniform(0.5, 1.5)
    M = cv2.getRotationMatrix2D((w//2, h//2), rot, scale)
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if flip is None: flip = rng.choice([0, 1, -1, None])
    if flip is not None:
        m = cv2.flip(m, flip)
    m = m.astype(np.float32) / 255.0
    # Normaliza y asegura contraste; invierte si la mayoría es fondo claro.
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    if m.mean() > 0.55:
        m = 1.0 - m
    m = np.clip(m, 0.0, 1.0)
    # Realza rayas estrechas: sube contraste en altas frecuencias
    m = cv2.GaussianBlur(m, (0, 0), 0.6)  # suaviza ruido pequeño
    m = np.power(m, 0.8, dtype=np.float32)
    return m, path.name

def _make_blobs_mask(h, w, count=10, size_range=(20,60), polarity=None, rng=None):
    """
    Genera máscaras de blobs robustas a imágenes pequeñas:
    - Ajusta (clamp) el radio para que quepa: radius <= (min(h,w)-2)//2
    - Si el rango pedido no cabe, reduce automáticamente.
    - Evita rangos vacíos para randint en x,y.
    """
    rng = rng or random
    bmin, bmax = size_range
    # radio máximo que cabe
    max_fit = max(1, (min(h, w) - 2) // 2)
    if max_fit < 1:
        # Imagen demasiado pequeña: sin blobs
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w, 3), dtype=np.float32)

    # clamp del rango pedido al que cabe
    adj_min = max(1, min(bmin, max_fit))
    adj_max = max(1, min(bmax, max_fit))
    if adj_min > adj_max:
        adj_min = adj_max  # colapsa al máximo posible

    alpha = np.zeros((h, w), dtype=np.float32)
    color = np.zeros((h, w, 3), dtype=np.float32)

    for _ in range(max(0, int(count))):
        radius = rng.randint(adj_min, adj_max)

        # rangos válidos para el centro
        x_left  = radius
        x_right = max(radius, w - radius)
        y_top   = radius
        y_bot   = max(radius, h - radius)

        # si no hay espacio horizontal/vertical, ubicamos al centro
        if x_right <= x_left:
            cx = w // 2
        else:
            cx = rng.randint(x_left, x_right)

        if y_bot <= y_top:
            cy = h // 2
        else:
            cy = rng.randint(y_top, y_bot)

        pol = rng.choice([-1, 1]) if polarity is None else polarity

        m = np.zeros((h, w), dtype=np.float32)
        cv2.circle(m, (cx, cy), radius, 1.0, -1, lineType=cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=radius/2, sigmaY=radius/2)

        a = 0.3  # opacidad base
        alpha = np.clip(alpha + m * a, 0, 1)

        if pol > 0:
            # claros → sumar blanco
            color += np.dstack([m, m, m])

    color = np.clip(color, 0, 1)
    return alpha, color





def _motion_blur(img, ksize=9, angle_deg=0.0, sigma=0.0):
    """Blur lineal en dirección angle_deg (usa depthwise separable aproximado)."""
    k = max(3, int(ksize) | 1)
    rad = math.radians(angle_deg % 180)
    # kernel direccional aproximado: proyectamos un 1D gauss a los ejes
    # (más barato que construir kernel rotado explícito)
    if abs(math.sin(rad)) > abs(math.cos(rad)):
        tmp = cv2.GaussianBlur(img, (1, k), sigmaX=0, sigmaY=sigma)
        out = cv2.GaussianBlur(tmp, (k, 1), sigmaX=sigma, sigmaY=0)
    else:
        tmp = cv2.GaussianBlur(img, (k, 1), sigmaX=sigma, sigmaY=0)
        out = cv2.GaussianBlur(tmp, (1, k), sigmaX=0, sigmaY=sigma)
    return out

def _fbm_grunge(h, w, octaves=4, base_sigma=3.0, gain=0.5, seed=None):
    """fBm rápido con gaussian blur multi-escala sobre ruido blanco."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        base = rng.rand(h, w).astype(np.float32)
    else:
        base = np.random.rand(h, w).astype(np.float32)
    acc = np.zeros_like(base)
    amp = 1.0
    sigma = base_sigma
    for _ in range(octaves):
        acc += amp * cv2.GaussianBlur(base, (0, 0), sigma)
        sigma *= 0.5
        amp *= gain
    acc = acc - acc.min()
    acc = acc / (acc.max() + 1e-8)
    return acc

def generate_scratches_overlay(
    h, w,
    seed=None,
    # --- control general ---
    num_bundles=2,                 # familias de líneas (pocas = "menos scratches")
    angle_range=(-45, 45),         # ángulos aleatorios por familia
    polarity_white_prob=0.65,      # prob. de que una familia sea blanca (resto negra)
    # --- forma de las líneas ---
    spacing_range=(28, 46),        # separación (más grande = menos denso)
    spacing_jitter_range=(0.1, 0.45),
    width_range=(1, 2),            # grosor de las líneas
    intensity_range=(0.35, 0.75),  # opacidad por familia
    dash_freq_range=(20.0, 30.0),  # frecuencia de “dashed”
    dash_depth_range=(0.25, 0.45), # profundidad del “dashed”
    noise_strength_range=(0.25, 0.45),  # grunge por familia
    blur_ksize_range=(5, 9),       # motion blur lineal
    blur_sigma_range=(0.8, 1.4),
    # --- variación longitudinal y tintes ---
    longitudinal_sigma_range=(12.0, 32.0),
    longitudinal_dropout_range=(0.08, 0.24),
    longitudinal_variation_strength_range=(0.35, 0.65),
    white_tint_range=((215, 222, 232), (245, 249, 255)),
    black_tint_range=((6, 8, 10), (26, 28, 34)),
    # --- rayas sueltas finas ---
    extras_count_range=(20, 60),
    extras_length_range=(10, 40),
    # --- deformaciones y variaciones ---
    warp_amplitude_range=(0.3, 2.5),
    warp_sigma_range=(8.0, 18.0),
    edge_halo_strength_range=(0.12, 0.28),
    core_gamma_range=(0.95, 1.3),
    local_variation_strength=0.35,
    angles_used=None
):
    """
    Devuelve overlay RGBA (uint8) con scratches blancos/negros y alpha.
    Añade curvaturas suaves, halos y variación local para un aspecto más orgánico.
    """
    import math
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = np.random.RandomState(seed)

    H, W = h, w
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    white_min = np.array(white_tint_range[0], dtype=np.float32)
    white_max = np.array(white_tint_range[1], dtype=np.float32)
    black_min = np.array(black_tint_range[0], dtype=np.float32)
    black_max = np.array(black_tint_range[1], dtype=np.float32)

    if angles_used is None:
        angles_used = []
    else:
        angles_used = list(angles_used)

    warp_used = []
    halo_used = []
    intensities_used = []
    polarity_used = []
    dropout_used = []
    variation_used = []
    spacing_jitter_used = []
    centers_used = []
    macro_traces = []

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def _warp_mask(mask, amp, sigma):
        if amp <= 0:
            return mask
        noise_x = rng.rand(H, W).astype(np.float32) - 0.5
        noise_y = rng.rand(H, W).astype(np.float32) - 0.5
        if sigma > 0:
            noise_x = cv2.GaussianBlur(noise_x, (0, 0), sigma)
            noise_y = cv2.GaussianBlur(noise_y, (0, 0), sigma)
        map_x = (xx + noise_x * amp).astype(np.float32)
        map_y = (yy + noise_y * amp).astype(np.float32)
        return cv2.remap(mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    def _line_family_mask(angle_deg, spacing_rng, width_rng,
                          dash_freq_rng, dash_depth_rng, jitter_strength):
        """Genera un mapa con familias de rayas, usando posiciones irregulares."""
        ang = math.radians(angle_deg)
        x_rot = xx * math.cos(ang) + yy * math.sin(ang)  # coordenada perpendicular a la raya
        t_coord = xx * (-math.sin(ang)) + yy * (math.cos(ang))  # coordenada a lo largo de la raya

        spacing_base = max(rng.uniform(*spacing_rng), 1.0)
        extent_min = float(x_rot.min())
        extent_max = float(x_rot.max())

        centers = []
        pos = extent_min - rng.uniform(0, spacing_base * 1.5)
        while pos < extent_max + spacing_base:
            if rng.rand() > 0.18:
                centers.append(pos)
            step = spacing_base * rng.uniform(0.55, 1.65)
            pos += max(step, 1.0)

        if not centers:
            return np.zeros((H, W), dtype=np.float32), []

        centers = np.array(centers, dtype=np.float32)
        n_lines = centers.shape[0]

        line_map = np.zeros((H, W), dtype=np.float32)
        line_meta = []

        width_field = rng.uniform(width_rng[0], width_rng[1], size=(H, W)).astype(np.float32)
        width_field = cv2.GaussianBlur(width_field, (0, 0), 2.0)

        jitter_strength = max(float(jitter_strength), 0.0)
        if jitter_strength > 1e-4:
            # Jitter conforme a la coordenada longitudinal (suavizado para evitar artefactos duros)
            t_min = float(t_coord.min())
            t_max = float(t_coord.max())
            span = max(1, int(np.ceil(t_max - t_min)) + 5)
            t_idx = np.clip(np.round(t_coord - t_min).astype(np.int32), 0, span - 1)
            streak_profile = rng.randn(span).astype(np.float32)
            blur_sigma = spacing_base * 0.4 + 1.0
            if blur_sigma > 0:
                streak_profile = cv2.GaussianBlur(streak_profile[:, None], (0, 0), blur_sigma).ravel()
            prof_max = float(np.max(np.abs(streak_profile))) if span > 0 else 0.0
            if prof_max > 1e-5:
                streak_profile = streak_profile / prof_max
        else:
            t_idx = None
            streak_profile = None

        for idx, center in enumerate(centers):
            width_pix = rng.uniform(width_rng[0], width_rng[1]) * rng.uniform(0.8, 1.8)

            dist = np.abs(x_rot - center)
            base_mask = np.clip(1.0 - dist / (width_pix + width_field + 1e-6), 0.0, 1.0)
            if base_mask.max() < 1e-4:
                continue

            dash_freq = rng.uniform(*dash_freq_rng)
            dash_depth = rng.uniform(*dash_depth_rng)
            dash_phase = rng.uniform(0, 2 * math.pi)
            dash = 0.5 + 0.5 * np.sin((t_coord / max(dash_freq, 1e-3) * 2 * math.pi) + dash_phase)
            dash = 1.0 - dash_depth * (1.0 - dash)
            mask = base_mask * dash.astype(np.float32)

            if jitter_strength > 1e-4 and streak_profile is not None:
                offset = streak_profile[t_idx] * spacing_base * jitter_strength
                dist_j = np.abs((x_rot - offset) - center)
                mask = np.clip(1.0 - dist_j / (width_pix + width_field + 1e-6), 0.0, 1.0) * dash.astype(np.float32)

            if rng.rand() < 0.35:
                breakup = rng.rand(H, W).astype(np.float32)
                breakup = cv2.GaussianBlur(breakup, (0, 0), rng.uniform(3.0, 7.0))
                mask *= np.clip(0.7 + 0.6 * (breakup - breakup.min()) / (breakup.max() - breakup.min() + 1e-6), 0.0, 1.2)

            line_map = np.maximum(line_map, mask)
            line_meta.append(float(center))

        return np.clip(line_map, 0, 1), line_meta

    def _apply_color_alpha(dst_rgba, color_rgb, alpha_map):
        A = alpha_map.astype(np.float32)
        if A.ndim == 2:
            A = A[..., None]
        C = color_rgb.astype(np.float32)
        dst_rgb = dst_rgba[..., :3]
        dst_a = dst_rgba[..., 3:4]
        out_rgb = C * A + dst_rgb * (1.0 - A)
        out_a = np.maximum(dst_a, A)
        dst_rgba[..., :3] = out_rgb
        dst_rgba[..., 3:4] = out_a

    def _make_color(min_vals, max_vals, sigma=1.2, jitter=0.14):
        base = np.array([rng.uniform(lo, hi) for lo, hi in zip(min_vals, max_vals)], dtype=np.float32)
        color = np.tile(base, (H, W, 1))
        noise = rng.rand(H, W).astype(np.float32)
        if sigma > 0:
            noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        mod = 1.0 + (noise - 0.5) * 2.0 * jitter
        return np.clip(color * mod[..., None], 0, 255.0)

    # familias principales
    for _ in range(num_bundles):
        angle = rng.uniform(*angle_range)
        angles_used.append(float(angle))
        intensity = rng.uniform(*intensity_range)
        intensities_used.append(float(intensity))
        noise_strength = rng.uniform(*noise_strength_range)
        ksize = int(rng.uniform(*blur_ksize_range)) | 1
        sigma = rng.uniform(*blur_sigma_range)
        warp_amp = rng.uniform(*warp_amplitude_range)
        warp_sig = rng.uniform(*warp_sigma_range)
        spacing_jitter = rng.uniform(*spacing_jitter_range)
        halo_strength = rng.uniform(*edge_halo_strength_range)
        warp_used.append(float(warp_amp))
        halo_used.append(float(halo_strength))
        spacing_jitter_used.append(float(spacing_jitter))

        # máscara base de líneas
        M, centers = _line_family_mask(angle, spacing_range, width_range, dash_freq_range, dash_depth_range, spacing_jitter)
        centers_used.append(centers)

        # cortes y variación longitudinal para romper la continuidad perfecta
        sigma_long = max(0.0, rng.uniform(*longitudinal_sigma_range))
        dropout_target = np.clip(rng.uniform(*longitudinal_dropout_range), 0.0, 0.95)
        var_strength = np.clip(rng.uniform(*longitudinal_variation_strength_range), 0.0, 1.0)
        dropout_used.append(float(dropout_target))
        variation_used.append(float(var_strength))
        ang_rad = math.radians(angle)
        axis_coord = xx * (-math.sin(ang_rad)) + yy * math.cos(ang_rad)
        axis_min = float(axis_coord.min())
        axis_max = float(axis_coord.max())
        span = max(1, int(np.ceil(axis_max - axis_min)) + 3)
        axis_idx = np.clip(np.round(axis_coord - axis_min).astype(np.int32), 0, span - 1)

        if span > 1 and dropout_target > 0:
            drop_profile = rng.rand(span).astype(np.float32)
            if sigma_long > 0:
                drop_profile = cv2.GaussianBlur(drop_profile[:, None], (0, 0), sigma_long * 0.5 + 0.5).ravel()
            profile_range = drop_profile.max() - drop_profile.min()
            if profile_range < 1e-5:
                gating_profile = np.ones_like(drop_profile, dtype=np.float32)
            else:
                drop_profile = (drop_profile - drop_profile.min()) / (profile_range + 1e-8)
                gating_profile = np.clip((drop_profile - dropout_target) / max(1e-4, 1.0 - dropout_target), 0.0, 1.0)
            if gating_profile.max() < 1e-3:
                gating_profile[:] = 1.0
            gating_map = gating_profile[axis_idx]
            if sigma_long > 0:
                gating_map = cv2.GaussianBlur(gating_map.astype(np.float32), (0, 0), sigma_long * 0.15 + 0.15)
            M = M * np.clip(gating_map, 0.0, 1.0)

        if span > 1 and var_strength > 0:
            var_profile = rng.rand(span).astype(np.float32)
            if sigma_long > 0:
                var_profile = cv2.GaussianBlur(var_profile[:, None], (0, 0), sigma_long * 0.6 + 0.8).ravel()
            profile_range = var_profile.max() - var_profile.min()
            if profile_range < 1e-5:
                variation_map = np.ones_like(axis_idx, dtype=np.float32)
            else:
                var_profile = (var_profile - var_profile.min()) / (profile_range + 1e-8)
                variation_map = var_profile[axis_idx]
            M = M * (1.0 - var_strength + var_strength * variation_map.astype(np.float32))

        # grunge (fBm rápido)
        G = _fbm_grunge(H, W, octaves=4, base_sigma=4.0, gain=0.55, seed=seed ^ rng.randint(1, 1 << 30))
        M = M * (0.7 + noise_strength * (G * 0.8 + 0.2))

        if local_variation_strength > 0:
            jitter = rng.rand(H, W).astype(np.float32) - 0.5
            jitter = cv2.GaussianBlur(jitter, (0, 0), 2.5)
            M = M * (1.0 + local_variation_strength * jitter)

        M = np.clip(M, 0, 1)

        # ondulación leve de la familia
        M = _warp_mask(M, warp_amp, warp_sig)
        M = np.clip(M, 0, 1)

        # motion blur en dirección del scratch
        M = _motion_blur(M, ksize=ksize, angle_deg=angle, sigma=sigma)
        if M.max() > 1e-6:
            M = M - M.min()
            M = M / (M.max() + 1e-8)

        gamma = rng.uniform(*core_gamma_range)
        M = np.clip(np.power(M + 1e-6, gamma), 0, 1)
        M = np.clip(M * intensity, 0, 1)

        # color por polaridad (con halo para dar volumen)
        is_white = bool(rng.rand() < polarity_white_prob)
        polarity_used.append("white" if is_white else "black")

        rim_sigma = max(0.6, sigma * 0.6 + 0.8)
        rim = cv2.GaussianBlur(M, (0, 0), rim_sigma)
        rim = np.clip(rim * halo_strength, 0, 1)

        bright_color = _make_color(white_min, white_max, sigma=1.2, jitter=0.16)
        dark_color = _make_color(black_min, black_max, sigma=1.6, jitter=0.1)

        if is_white:
            _apply_color_alpha(overlay, bright_color, M)
            if halo_strength > 0:
                shadow = np.clip(rim * 0.6, 0, 1)
                _apply_color_alpha(overlay, np.clip(dark_color * 1.1, 0, 255.0), shadow)
        else:
            _apply_color_alpha(overlay, dark_color, M)
            if halo_strength > 0:
                highlight = np.clip(rim * 0.6, 0, 1) * 0.5
                _apply_color_alpha(overlay, np.clip(bright_color * 0.85, 0, 255.0), highlight)

    # macro scratches largos tipo "raspones" que cruzan la imagen
    macro_count = rng.randint(0, 3)
    for _ in range(macro_count):
        macro_mask = np.zeros((H, W), dtype=np.float32)
        x0 = rng.randint(-W // 4, int(W * 1.25))
        y0 = rng.randint(-H // 6, int(H * 1.1))
        x1 = x0 + rng.randint(int(0.35 * W), int(1.3 * W)) * rng.choice([-1, 1])
        y1 = y0 + rng.randint(-int(0.3 * H), int(0.3 * H) + 1)

        n_ctrl = rng.randint(2, 4)
        ctrl = []
        for j in range(n_ctrl):
            alpha = j / max(1, n_ctrl - 1)
            cx = x0 + (x1 - x0) * alpha + rng.uniform(-W * 0.08, W * 0.08)
            cy = y0 + (y1 - y0) * alpha + rng.uniform(-H * 0.08, H * 0.08)
            ctrl.append([int(cx), int(cy)])

        pts = np.array(ctrl, dtype=np.int32).reshape((-1, 1, 2))
        thickness = max(1, int(rng.uniform(1.0, 3.8)))
        cv2.polylines(macro_mask, [pts], False, 1.0, thickness=thickness, lineType=cv2.LINE_AA)
        macro_mask = cv2.GaussianBlur(macro_mask, (0, 0), rng.uniform(1.2, 3.6))
        if macro_mask.max() < 1e-4:
            continue

        macro_noise = rng.rand(H, W).astype(np.float32)
        macro_noise = cv2.GaussianBlur(macro_noise, (0, 0), rng.uniform(3.5, 7.5))
        macro = macro_mask * np.clip(0.5 + 0.7 * (macro_noise - macro_noise.min()) / (macro_noise.max() - macro_noise.min() + 1e-6), 0.2, 1.4)
        macro = np.clip(macro, 0.0, 1.0)
        macro_alpha = macro * rng.uniform(0.25, 0.55)

        if rng.rand() < 0.7:
            macro_color = _make_color(white_min, white_max, sigma=1.0, jitter=0.15)
            macro_polarity = "white"
        else:
            macro_color = _make_color(black_min, black_max, sigma=1.3, jitter=0.1)
            macro_polarity = "black"

        _apply_color_alpha(overlay, macro_color, macro_alpha)
        macro_traces.append({
            "points": [tuple(p) for p in ctrl],
            "thickness": thickness,
            "polarity": macro_polarity,
            "alpha": float(np.max(macro_alpha))
        })

    # rayas sueltas (algunas negras, otras blancas)
    A_ex = np.zeros((H, W), dtype=np.float32)
    C_ex = np.zeros((H, W, 3), dtype=np.float32)
    extras_white = _make_color(white_min, white_max, sigma=0.9, jitter=0.1)
    extras_black = _make_color(black_min, black_max, sigma=0.9, jitter=0.08)
    extras_count = max(0, int(rng.randint(*extras_count_range)))
    for _ in range(extras_count):
        x0 = rng.randint(0, W)
        y0 = rng.randint(0, H)
        length = rng.randint(*extras_length_range)
        angle = rng.uniform(*angle_range)
        rad = np.deg2rad(angle)
        jitter_mag = max(length * 0.25, 1.0)
        mid_x = int(np.clip(x0 + (length * 0.5) * math.cos(rad) + rng.uniform(-jitter_mag, jitter_mag) * 0.1, 0, W - 1))
        mid_y = int(np.clip(y0 + (length * 0.5) * math.sin(rad) + rng.uniform(-jitter_mag, jitter_mag) * 0.1, 0, H - 1))
        x1 = int(np.clip(x0 + length * math.cos(rad) + rng.uniform(-2, 2), 0, W - 1))
        y1 = int(np.clip(y0 + length * math.sin(rad) + rng.uniform(-2, 2), 0, H - 1))
        lw = rng.choice([1, 1, 2])

        # máscara fina con ligera curvatura
        tmp = np.zeros((H, W), dtype=np.float32)
        pts = np.array([[x0, y0], [mid_x, mid_y], [x1, y1]], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(tmp, [pts], False, 1.0, thickness=lw, lineType=cv2.LINE_AA)
        tmp = cv2.GaussianBlur(tmp, (0, 0), 0.8)

        if rng.rand() < polarity_white_prob:
            C_ex += extras_white * (tmp[..., None] * 0.9)
        else:
            C_ex += extras_black * (tmp[..., None] * 0.7)
        A_ex = np.maximum(A_ex, tmp * 0.85)

    if A_ex.max() > 0:
        A_ex = cv2.GaussianBlur(A_ex, (3, 3), 0.8)
        A_ex = np.clip(A_ex, 0, 1) * 0.35
        C_ex = np.clip(C_ex, 0, 255.0)
        overlay_rgb = overlay[..., :3]
        overlay_a = overlay[..., 3:4]
        overlay_rgb = C_ex * A_ex[..., None] + overlay_rgb * (1.0 - A_ex[..., None])
        overlay_a = np.maximum(overlay_a, A_ex[..., None])
        overlay[..., :3] = overlay_rgb
        overlay[..., 3:4] = overlay_a

    overlay[..., :3] = np.clip(overlay[..., :3], 0, 255.0)
    overlay[..., 3] = np.clip(overlay[..., 3], 0, 1.0)

    out = overlay.copy()
    out[..., 3] = np.clip(out[..., 3] * 255.0, 0, 255.0)
    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)

    return out_uint8, {
        "seed": int(seed),
        "angles": angles_used,
        "spacing_range": (float(spacing_range[0]), float(spacing_range[1])),
        "spacing_jitter": spacing_jitter_used,
        "line_centers": [list(map(float, c)) for c in centers_used],
        "macro_traces": macro_traces,
        "intensity_used": intensities_used,
        "warp_amplitude": warp_used,
        "edge_halo_strength": halo_used,
        "polarities": polarity_used,
        "longitudinal_dropout": dropout_used,
        "longitudinal_variation": variation_used,
    }


def _resolve_procedural_scratch_params(preset_name, preset):
    """
    Devuelve parámetros para generate_scratches_overlay según el preset solicitado.
    Prefiere alias definidos en SCRATCH_PRESET_ALIAS y cae a un estilo por defecto.
    """
    alias_name = None
    if preset.get("proc_scratches", False):
        params = dict(preset.get("params", {}))
        if params:
            return params
        alias_name = preset_name
    else:
        alias_name = SCRATCH_PRESET_ALIAS.get(preset_name, DEFAULT_PROC_SCRATCH_ALIAS)

    alias_data = OVERLAY_PRESETS.get(alias_name, {})
    if alias_data.get("proc_scratches"):
        return dict(alias_data.get("params", {}))

    fallback = OVERLAY_PRESETS.get(DEFAULT_PROC_SCRATCH_ALIAS, {})
    return dict(fallback.get("params", {}))


def _ensure_unique_path(path):
    """
    Si el archivo ya existe, genera un nuevo nombre con sufijo incremental.
    """
    if not path.exists():
        return path
    counter = 1
    base_stem = path.stem
    while True:
        candidate = path.with_name(f"{base_stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def recolectar_restauraciones_imagen(
    imagen_original,
    carpeta_output_root=Path("/home/laura/CycleGAN/RRTN-old-film-restoration/OUTPUT"),
    carpeta_salida=Path("/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/combinacionpaper/imgoriginal_restaurada"),
    subcarpeta_resultados="test_results_30_rec1"
):
    """
    Busca todas las restauraciones generadas por RRTN para una imagen específica
    dentro de OUTPUT/*/rrtn_net_*/rrtn/<subcarpeta_resultados>/...
    Copia la imagen original (sufijo _original) y cada restauración con sufijo
    _<batch>_<modelo> al directorio de salida.
    """
    imagen_original = Path(imagen_original)
    if not imagen_original.exists():
        raise FileNotFoundError(f"Imagen original no encontrada: {imagen_original}")

    carpeta_output_root = Path(carpeta_output_root)
    if not carpeta_output_root.exists() or not carpeta_output_root.is_dir():
        raise FileNotFoundError(f"Carpeta OUTPUT inválida: {carpeta_output_root}")

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Copiar original primero
    destino_original = carpeta_salida / f"{imagen_original.stem}_original{imagen_original.suffix}"
    destino_original = _ensure_unique_path(destino_original)
    shutil.copy2(imagen_original, destino_original)

    partes = imagen_original.parts
    # Genera sufijos crecientes (sin incluir parte absoluta)
    posibles_sufijos = [Path(*partes[i:]) for i in range(1, len(partes))]

    hallazgos = []
    for batch_dir in sorted(carpeta_output_root.iterdir()):
        if not batch_dir.is_dir():
            continue
        for modelo_dir in sorted(batch_dir.iterdir()):
            if not modelo_dir.is_dir():
                continue
            test_dir = modelo_dir / "rrtn" / subcarpeta_resultados
            if not test_dir.exists() or not test_dir.is_dir():
                continue

            match_path = None
            match_rel = None
            for sufijo in posibles_sufijos:
                candidato = test_dir / sufijo
                if candidato.exists():
                    match_path = candidato
                    match_rel = sufijo
                    break
            if match_path is None:
                continue

            dest_name = f"{imagen_original.stem}_{batch_dir.name}_{modelo_dir.name}{match_path.suffix}"
            destino = carpeta_salida / dest_name
            destino = _ensure_unique_path(destino)
            shutil.copy2(match_path, destino)

            hallazgos.append({
                "batch": batch_dir.name,
                "modelo": modelo_dir.name,
                "origen": match_path,
                "destino": destino,
                "relativo": match_rel
            })

    return destino_original, hallazgos


def generar_overlays_imagen(
    image_path,
    output_dir="/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/overlaysimg",
    presets=None,
    scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches"),
    csv_name="overlays_meta.csv",
    use_atomic=True,
    compose_and_save=True,  # NEW: genera también imagen compuesta
    export_rrtn_texture=False,
    only_rrtn_texture=False,
    rrtn_background=245.0,
    rrtn_darkness=200.0,
    rrtn_alpha_gamma=0.75,
    rrtn_blur_sigma=0.0,
    rrtn_noise_sigma=0.0
):
    """
    Genera UNA capa de artefactos por preset (PNG RGBA con fondo transparente)
    + CSV con parámetros para reproducir. Similar a opción 5 pero sin hornear en la imagen.
    """
    if only_rrtn_texture:
        export_rrtn_texture = True

    presets = presets if presets is not None else (ATOMIC_PRESETS if use_atomic else DEGRADATION_PRESETS)
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = cv2.imread(str(image_path))
    if base is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    h, w = base.shape[:2]
    base_png = output_dir / f"{image_path.stem}_base_original.png"
    cv2.imwrite(str(base_png), base)

     # CSV (añadimos composed_path)  # NEW
    fieldnames = [
        "base_image", "preset", "overlay_path", "composed_path",
        "apply_vignette_globally", "vignette_alpha", "vignette_strength",
        "scratch_alpha", "scratch_template", "scratch_rotation", "scratch_scale", "scratch_flip",
        "blobs_count", "blob_size_min", "blob_size_max",
        "color_tone", "color_tone_strength",
        "burn_applied", "burn_intensity", "burn_cx", "burn_cy", "burn_w", "burn_h",
        "rrtn_texture_path",
        "notes"
    ]
    csv_path = output_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, preset in presets.items():
            rng_seed = random.randint(0, 2**31-1)  # guardamos seed por si quieres reproducir exacto
            random.seed(rng_seed)
            np.random.seed(rng_seed & 0xFFFFFFFF)

            # Overlay vacío RGBA [0,0,0,0]
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            rrtn_texture_rel = ""
            override_profile = preset.get("override_profile", {})
            notes_parts = [f"seed={rng_seed}"]

            # --- Vignette como oscurecimiento en bordes (negro con alfa radial) ---
            apply_vignette = preset.get("apply_vignette_globally", False)
            vignette_strength = override_profile.get("vignette_strength", 25 if name.startswith("only_") else 15)
            vignette_alpha = override_profile.get("vignette_alpha", 0.8 if apply_vignette else 0.0)
            if apply_vignette:
                vm = _make_vignette_mask(h, w, strength=vignette_strength)
                black = np.zeros((h, w, 3), dtype=np.float32)
                overlay = _overlay_max_alpha(overlay, black, (vm[..., None] * vignette_alpha))

            # --- Scratches procedurales (sin plantillas externas) ---
            scratch_alpha_override = override_profile.get("scratch_alpha")
            scratch_template_csv = None
            scratch_alpha_csv = scratch_alpha_override if scratch_alpha_override is not None else ""
            wants_scratch_overlay = (
                preset.get("proc_scratches", False)
                or preset.get("scratch_prob", 0) > 0
                or scratch_alpha_override is not None
            )
            if wants_scratch_overlay:
                scratch_params = _resolve_procedural_scratch_params(name, preset)
                scratch_params.setdefault("seed", rng_seed)
                ov_rgba, meta = generate_scratches_overlay(h, w, **scratch_params)
                alpha_map = ov_rgba[..., 3:4].astype(np.float32) / 255.0
                if scratch_alpha_override is not None:
                    alpha_map = np.clip(alpha_map * float(scratch_alpha_override), 0.0, 1.0)
                    scratch_alpha_csv = float(scratch_alpha_override)
                else:
                    scratch_alpha_csv = scratch_params.get("intensity_range", (0.35, 0.75))

                overlay = _overlay_max_alpha(
                    overlay,
                    ov_rgba[..., :3].astype(np.float32),
                    alpha_map
                )

                scratch_template_csv = "procedural"
                notes_parts.append(
                    "scratch_meta={}".format({
                        "seed": meta.get("seed"),
                        "angles": meta.get("angles"),
                        "spacing_range": meta.get("spacing_range"),
                        "spacing_jitter": meta.get("spacing_jitter"),
                        "macro_traces": meta.get("macro_traces"),
                        "warp": meta.get("warp_amplitude"),
                        "halo": meta.get("edge_halo_strength"),
                        "intensities": meta.get("intensity_used"),
                        "polarities": meta.get("polarities"),
                        "dropout": meta.get("longitudinal_dropout"),
                        "long_var": meta.get("longitudinal_variation")
                    })
                )

            # --- Blobs claros/oscuros aproximados como mezcla con blanco/negro ---
            blobs_count = 0
            bmin, bmax = 20, 60
            if preset.get("blob_prob", 0) > 0:
                blobs_count = 10
                if "blob_size" in override_profile:
                    bmin, bmax = override_profile["blob_size"]
                alpha_b, color_b = _make_blobs_mask(h, w, count=blobs_count, size_range=(bmin, bmax))
                if alpha_b.max() > 0:
                    overlay = _overlay_max_alpha(overlay, (color_b * 255.0), alpha_b[..., None]*0.6)
                    black = np.zeros((h, w, 3), dtype=np.float32)
                    overlay = _overlay_max_alpha(overlay, black, alpha_b[..., None]*0.2)

            # --- Burn (quemado) como elipse naranja semi-transparente ---
            burn_applied = False; burn_intensity = 0.0; burn_cx=burn_cy=burn_w=burn_h=0
            burn_prob = override_profile.get("burn_prob", 0.0)
            if burn_prob > 0:
                burn_applied = True
                burn_intensity = override_profile.get("burn_intensity", 0.25)
                mask = np.zeros((h, w), dtype=np.float32)
                cx = random.randint(0, w-1); cy = random.randint(0, h-1)
                rw = int(w * 0.3 * random.uniform(0.5, 1.5))
                rh = int(h * 0.3 * random.uniform(0.5, 1.5))
                cv2.ellipse(mask, (cx, cy), (max(1,rw//2), max(1,rh//2)), 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (101,101), 0)
                orange = np.array([0, 128, 255], dtype=np.float32)  # BGR
                overlay = _overlay_max_alpha(overlay, np.tile(orange,(h,w,1)), (mask[...,None]*burn_intensity))
                burn_cx, burn_cy, burn_w, burn_h = cx, cy, rw, rh

            # --- Tono de color como capa sólida del color con alfa ---
            tone = preset.get("color_tone", None)
            tone_str = preset.get("color_tone_strength", 0.0)
            if tone is not None and tone_str > 0:
                c = _color_from_tone(tone)
                if c is not None:
                    overlay = _overlay_max_alpha(overlay, np.tile(c,(h,w,1)), float(tone_str))





            # Guardar overlay RGBA
            # Guardar overlay RGBA
            out_png = output_dir / f"{image_path.stem}_{name}_overlay.png"

            # >>> FIX: escalar alfa de 0–1 a 0–255 antes de convertir a uint8
            rgba_save = overlay.copy()                     # overlay es float32
            rgba_save[..., 3] = np.clip(rgba_save[..., 3] * 255.0, 0, 255)

            rgba_uint8 = np.clip(rgba_save, 0, 255).astype(np.uint8)
            overlay_rel = ""
            if not only_rrtn_texture:
                cv2.imwrite(str(out_png), rgba_uint8)
                overlay_rel = out_png.name

            rrtn_texture_rel = ""
            if export_rrtn_texture:
                rrtn_texture = _rgba_to_rrtn_texture(
                    rgba_uint8,
                    background=rrtn_background,
                    darkness=rrtn_darkness,
                    alpha_gamma=rrtn_alpha_gamma,
                    blur_sigma=rrtn_blur_sigma,
                    noise_sigma=rrtn_noise_sigma
                )
                rrtn_path = out_png if only_rrtn_texture else out_png.with_name(f"{out_png.stem}_rrtn.png")
                cv2.imwrite(str(rrtn_path), rrtn_texture)
                rrtn_texture_rel = rrtn_path.name
            if only_rrtn_texture:
                overlay_rel = rrtn_texture_rel

            # NEW: componer con la base y guardar compuesto
            composed_rel = ""
            if compose_and_save and not only_rrtn_texture:
                base_f = base.astype(np.float32)
                ov_f   = rgba_uint8.astype(np.float32)
                alpha  = ov_f[..., 3:4] / 255.0          # (H,W,1)
                comp   = ov_f[..., :3] * alpha + base_f * (1.0 - alpha)
                comp_path = output_dir / f"{image_path.stem}_{name}_composed.png"
                cv2.imwrite(str(comp_path), np.clip(comp, 0, 255).astype(np.uint8))
                composed_rel = comp_path.name

            # Escribir fila CSV (incluye composed_path)  # NEW
            writer.writerow({
                "base_image": str(base_png.name),
                "preset": name,
                "overlay_path": overlay_rel,
                "composed_path": composed_rel,
                "apply_vignette_globally": apply_vignette,
                "vignette_alpha": float(vignette_alpha),
                "vignette_strength": float(vignette_strength),
                "scratch_alpha": scratch_alpha_csv,
                "scratch_template": scratch_template_csv,
                "scratch_rotation": "", "scratch_scale": "", "scratch_flip": "",
                "blobs_count": int(blobs_count), "blob_size_min": bmin, "blob_size_max": bmax,
                "color_tone": preset.get("color_tone", None),
                "color_tone_strength": preset.get("color_tone_strength", 0.0),
                "burn_applied": burn_applied,
                "burn_intensity": float(burn_intensity),
                "burn_cx": burn_cx, "burn_cy": burn_cy, "burn_w": burn_w, "burn_h": burn_h,
                "rrtn_texture_path": rrtn_texture_rel,
                "notes": " | ".join(notes_parts)
            })

    print(f"🎉 Overlays y compuestos guardados en: {output_dir}")
    print(f"🧾 CSV: {csv_path}")
    return str(csv_path)


# ------------------------
# -------------------------
# /FIN GENERADOR EXTERNO DE CAPAS DE ARTEFACTOS (OVERLAYS PNG RGBA)
# -------------------------


def componer_base_overlay(
    base_path,
    overlay_path,
    out_path,
    overlay_mode="auto",
    grayscale_strength=0.6
):
    """
    Compone una base BGR con una plantilla:
    - 'rgba': interpreta overlay con canal alfa (BGRA) y hace mezcla Porter-Duff.
    - 'grayscale': overlay en escala de grises (fondo claro, arañazos oscuros).
    - 'auto': detecta según canales.
    """
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"No se pudo leer imagen base: {base_path}")
    base_h, base_w = base.shape[:2]

    ov = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    if ov is None:
        raise FileNotFoundError(f"No se pudo leer overlay: {overlay_path}")

    detected_mode = overlay_mode
    if overlay_mode == "auto":
        if ov.ndim == 3 and ov.shape[2] == 4:
            detected_mode = "rgba"
        else:
            detected_mode = "grayscale"

    if detected_mode == "rgba":
        if ov.ndim != 3 or ov.shape[2] not in (3, 4):
            raise ValueError("Overlay RGBA debe tener 3 o 4 canales.")

        if ov.shape[2] == 4:
            ov_rgb = ov[..., :3]
            ov_alpha = ov[..., 3:4]
        else:
            ov_rgb = ov
            ov_alpha = np.full(ov_rgb.shape[:2] + (1,), 255, dtype=ov.dtype)

        ov_h, ov_w = ov_rgb.shape[:2]
        if (ov_h, ov_w) != (base_h, base_w):
            ov_rgb = cv2.resize(ov_rgb, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            ov_alpha = cv2.resize(ov_alpha, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
            if ov_alpha.ndim == 2:
                ov_alpha = ov_alpha[..., None]

        base_f = base.astype(np.float32)
        ov_rgb = ov_rgb.astype(np.float32)
        ov_alpha = ov_alpha.astype(np.float32) / 255.0
        comp = ov_rgb * ov_alpha + base_f * (1.0 - ov_alpha)
        cv2.imwrite(str(out_path), np.clip(comp, 0, 255).astype(np.uint8))
        return out_path

    if detected_mode == "grayscale":
        if ov.ndim == 3:
            ov_gray = cv2.cvtColor(ov, cv2.COLOR_BGR2GRAY)
        elif ov.ndim == 2:
            ov_gray = ov
        else:
            raise ValueError("Overlay en escala de grises debe tener 1 o 3 canales.")

        if ov_gray.shape[:2] != (base_h, base_w):
            ov_gray = cv2.resize(ov_gray, (base_w, base_h), interpolation=cv2.INTER_LINEAR)

        base_f = base.astype(np.float32) / 255.0
        ov_norm = ov_gray.astype(np.float32) / 255.0
        scratch_mask = (1.0 - ov_norm) * float(np.clip(grayscale_strength, 0.0, 1.0))
        comp = np.clip(base_f - scratch_mask[..., None], 0.0, 1.0)
        cv2.imwrite(str(out_path), (comp * 255.0).astype(np.uint8))
        return out_path

    raise ValueError(f"overlay_mode desconocido: {overlay_mode}")


def combinar_carpeta_base_overlay(
    carpeta_base,
    carpeta_overlays,
    carpeta_salida,
    extensiones_base=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    extensiones_overlay=(".png",),
    overlay_mode="auto",
    grayscale_strength=0.6,
    assign_mode="name",
    reverse_sort=False
):
    """
    Combina imágenes base con overlays RGBA usando nombres coincidentes.
    Guarda la composición en `carpeta_salida` con el mismo nombre del fotograma.
    """
    carpeta_base = Path(carpeta_base)
    carpeta_overlays = Path(carpeta_overlays)
    carpeta_salida = Path(carpeta_salida)

    if not carpeta_base.exists() or not carpeta_base.is_dir():
        raise FileNotFoundError(f"Carpeta base no válida: {carpeta_base}")
    if not carpeta_overlays.exists() or not carpeta_overlays.is_dir():
        raise FileNotFoundError(f"Carpeta de overlays no válida: {carpeta_overlays}")

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    base_files = [
        p for p in sorted(carpeta_base.iterdir())
        if p.is_file() and p.suffix.lower() in extensiones_base
    ]
    if not base_files:
        print(f"⚠️ No se encontraron imágenes base en {carpeta_base}")
        return

    overlay_files = [
        p for p in sorted(carpeta_overlays.iterdir(), reverse=reverse_sort)
        if p.is_file() and p.suffix.lower() in extensiones_overlay
    ]
    if not overlay_files:
        print(f"⚠️ No se encontraron overlays válidos en {carpeta_overlays}")
        return

    compuestos = 0
    faltantes = []
    errores = []

    if assign_mode == "sorted":
        base_iter = sorted(base_files, reverse=reverse_sort)
        count = min(len(base_iter), len(overlay_files))
        if count == 0:
            print("⚠️ No hay pares coincidentes para combinar.")
            return
        if len(base_iter) != len(overlay_files):
            print(f"⚠️ Conteo distinto: {len(base_iter)} bases vs {len(overlay_files)} overlays. Se combinarán {count}.")
        pairs = zip(base_iter[:count], overlay_files[:count])
    else:
        # modo por nombre (default)
        overlay_by_name = {p.name: p for p in overlay_files}
        overlay_by_stem = {}
        for p in overlay_files:
            overlay_by_stem.setdefault(p.stem, p)
        pairs = []
        for base_path in base_files:
            overlay_path = overlay_by_name.get(base_path.name)
            if overlay_path is None:
                overlay_path = overlay_by_stem.get(base_path.stem)
            if overlay_path is None:
                faltantes.append(base_path.name)
                continue
            pairs.append((base_path, overlay_path))

    for base_path, overlay_path in pairs:
        out_path = carpeta_salida / base_path.name
        try:
            componer_base_overlay(
                base_path,
                overlay_path,
                out_path,
                overlay_mode=overlay_mode,
                grayscale_strength=grayscale_strength
            )
            compuestos += 1
        except Exception as exc:
            errores.append((base_path.name, str(exc)))

    print(f"✅ Combinaciones realizadas: {compuestos}")
    if faltantes:
        lista = ", ".join(faltantes[:5])
        if len(faltantes) > 5:
            lista += ", ..."
        print(f"⚠️ No se encontró overlay para {len(faltantes)} archivos: {lista}")
    if errores:
        print("⚠️ Errores durante la combinación:")
        for nombre, msg in errores[:5]:
            print(f"   - {nombre}: {msg}")
        if len(errores) > 5:
            print(f"   ... y {len(errores) - 5} más.")


def combinar_subcarpetas_base_overlay(
    carpeta_base_root,
    carpeta_overlays_root,
    carpeta_salida_root,
    extensiones_base=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    extensiones_overlay=(".png",),
    overlay_mode="auto",
    grayscale_strength=0.6,
    assign_mode="name",
    reverse_sort=False
):
    """
    Itera por subcarpetas (escenas) y combina los fotogramas con sus overlays
    manteniendo los nombres de carpeta y archivo.
    """
    carpeta_base_root = Path(carpeta_base_root)
    carpeta_overlays_root = Path(carpeta_overlays_root)
    carpeta_salida_root = Path(carpeta_salida_root)
    carpeta_salida_root.mkdir(parents=True, exist_ok=True)

    if not carpeta_base_root.exists() or not carpeta_base_root.is_dir():
        raise FileNotFoundError(f"Carpeta base no válida: {carpeta_base_root}")
    if not carpeta_overlays_root.exists() or not carpeta_overlays_root.is_dir():
        raise FileNotFoundError(f"Carpeta de overlays no válida: {carpeta_overlays_root}")

    escenas_base = [p for p in carpeta_base_root.iterdir() if p.is_dir()]
    escenas_over = [p for p in carpeta_overlays_root.iterdir() if p.is_dir()]

    if not escenas_base:
        print(f"⚠️ No se encontraron subcarpetas en {carpeta_base_root}")
        return
    if not escenas_over:
        print(f"⚠️ No se encontraron subcarpetas en {carpeta_overlays_root}")
        return

    if assign_mode == "sorted":
        escenas_base.sort(reverse=reverse_sort)
        escenas_over.sort(reverse=reverse_sort)
        count = min(len(escenas_base), len(escenas_over))
        if count == 0:
            print("⚠️ No hay pares de escenas para combinar.")
            return
        if len(escenas_base) != len(escenas_over):
            print(f"⚠️ Conteo distinto: {len(escenas_base)} escenas base vs {len(escenas_over)} escenas de overlay. Se combinarán {count}.")
        scene_pairs = list(zip(escenas_base[:count], escenas_over[:count]))
    else:
        escenas_base.sort()
        escena_over_map = {p.name: p for p in escenas_over}
        scene_pairs = []
        faltantes = []
        for escena_base in escenas_base:
            overlay_scene = escena_over_map.get(escena_base.name)
            if overlay_scene is None:
                faltantes.append(escena_base.name)
                continue
            scene_pairs.append((escena_base, overlay_scene))
        if faltantes:
            lista = ", ".join(faltantes[:5])
            if len(faltantes) > 5:
                lista += ", ..."
            print(f"⚠️ No se encontraron overlays para {len(faltantes)} escenas: {lista}")

    procesadas = 0
    for escena_base, overlay_scene in scene_pairs:
        salida_scene = carpeta_salida_root / escena_base.name
        salida_scene.mkdir(parents=True, exist_ok=True)

        print(f"🎞️ Combinando escena base {escena_base.name} con overlay {overlay_scene.name}...")
        combinar_carpeta_base_overlay(
            escena_base,
            overlay_scene,
            salida_scene,
            extensiones_base=extensiones_base,
            extensiones_overlay=extensiones_overlay,
            overlay_mode=overlay_mode,
            grayscale_strength=grayscale_strength,
            assign_mode=assign_mode,
            reverse_sort=reverse_sort
        )
        procesadas += 1

    print(f"✅ Escenas procesadas: {procesadas}")


def crear_video_desde_frames(
    carpeta_frames,
    salida_video=None,
    fps=15,
    ordenar_por_nombre=True
):
    """
    Crea un video MP4 con los fotogramas de `carpeta_frames`.
    Se usa la resolución del primer frame válido encontrado.
    """
    carpeta_frames = Path(carpeta_frames)
    if not carpeta_frames.exists() or not carpeta_frames.is_dir():
        raise FileNotFoundError(f"Carpeta de frames inválida: {carpeta_frames}")

    if salida_video is None:
        salida_video = carpeta_frames / f"{carpeta_frames.name}.mp4"
    else:
        salida_video = Path(salida_video)

    frames = [
        p for p in carpeta_frames.iterdir() if p.is_file()
    ]
    if not frames:
        raise ValueError(f"No se encontraron imágenes en {carpeta_frames}")
    if ordenar_por_nombre:
        frames.sort()

    frame0 = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if frame0 is None:
        raise ValueError(f"No se pudo leer el primer frame: {frames[0]}")

    height, width = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(salida_video), fourcc, fps, (width, height))

    escritos = 0
    for frame_path in frames:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Frame inválido: {frame_path}, se omite.")
            continue
        if img.shape[:2] != (height, width):
            print(f"⚠️ Tamaño inconsistente en {frame_path}, se omite.")
            continue
        writer.write(img)
        escritos += 1

    writer.release()
    if escritos == 0:
        raise RuntimeError("No se escribió ningún frame en el video.")

    print(f"🎬 Video guardado en: {salida_video} ({escritos} frames, {width}x{height}, {fps} FPS)")
    return str(salida_video)


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
                    debug_path = os.path.join("debug_frames", f"frame{frame_idx:08d}_{step_name}.png")
                    cv2.imwrite(debug_path, np.clip(step_img, 0, 255).astype(np.uint8))
                print(f"[DEBUG] Guardadas imágenes del frame {frame_idx}.")

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
            cv2.imwrite(str(carpeta_out_escena / f"{frame_idx:08d}.png"), frame)
            frame_idx += 1
        cap.release()

        print(f"✅ Finalizado: {escena.name} → {output_prefix}_###.mp4 + {csv_filename} + {frame_idx} fotogramas degradados")

        # Elimina el archivo de video temporal
        try:
            temp_video.unlink()
        except Exception as e:
            print(f"[WARN] No se pudo borrar {temp_video}: {e}")

    print("🎬 Todas las escenas han sido degradadas y exportadas como videos y fotogramas.")

# métodos para que sean diferentes artefactos

def procesar_carpeta_por_tipos_compatible(
    carpeta_entrada, carpeta_videos, carpeta_fotogramas, tipos=None, presets=None, keep_original_names=True
):

    """
    MISMO flujo que opción 2:
    - Lee subcarpetas con imágenes (escenas).
    - Crea un temp video por escena.
    - Genera salidas en las MISMAS carpetas base que opción 2,
      pero en subcarpetas <escena>_<tipo>.
    """
    # <-- NUEVO: permitir presets personalizados (e.g., con custom:)
    if presets is None:
        presets = DEGRADATION_PRESETS
    if tipos is None:
        tipos = list(presets.keys())

    carpeta_entrada = Path(carpeta_entrada)
    carpeta_videos = Path(carpeta_videos)
    carpeta_fotogramas = Path(carpeta_fotogramas)
    carpeta_videos.mkdir(exist_ok=True, parents=True)
    carpeta_fotogramas.mkdir(exist_ok=True, parents=True)

    escenas = [f for f in carpeta_entrada.iterdir() if f.is_dir()]
    print(f"🔍 Se encontraron {len(escenas)} escenas en {carpeta_entrada}")

    for escena in escenas:
        print(f"🎞️ Procesando escena: {escena.name}")

        # 1) Igual que opción 2: crear temp video desde frames
        fotogramas = sorted(list(escena.glob("*.jpg")) + list(escena.glob("*.png")))
        if len(fotogramas) == 0:
            print(f"⚠️ No se encontraron imágenes en {escena}, se omite.")
            continue

        frame0 = cv2.imread(str(fotogramas[0]))
        if frame0 is None:
            print(f"⚠️ Imagen inválida en {fotogramas[0]}, se omite.")
            continue

        height, width = frame0.shape[:2]
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = carpeta_videos / f"{escena.name}_tempinput.mp4"
        vw = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        for img_path in fotogramas:
            img = cv2.imread(str(img_path))
            if img is not None and img.shape[:2] == (height, width):
                vw.write(img)
            else:
                print(f"[WARN] Imagen inválida o tamaño distinto: {img_path}")
        vw.release()

        # 2) Para cada tipo → crear subcarpetas <escena>_<tipo> en videos y frames
        for tipo in tipos:
            if tipo not in presets:
                print(f"⚠️ Tipo '{tipo}' no encontrado en presets. Se omite.")
                continue

            preset = presets[tipo]
            if keep_original_names:
                # Mantener nombres/carpetas originales
                carpeta_videos_tipo = carpeta_videos / escena.name
                carpeta_frames_tipo = carpeta_fotogramas / escena.name
                carpeta_videos_tipo.mkdir(exist_ok=True, parents=True)
                carpeta_frames_tipo.mkdir(exist_ok=True, parents=True)

                video_out = carpeta_videos_tipo / f"{escena.name}.mp4"
                csv_out   = carpeta_videos_tipo / f"{escena.name}.csv"
            else:
                subnombre = f"{escena.name}_{tipo}"
                carpeta_videos_tipo = carpeta_videos / subnombre
                carpeta_frames_tipo = carpeta_fotogramas / subnombre
                carpeta_videos_tipo.mkdir(exist_ok=True, parents=True)
                carpeta_frames_tipo.mkdir(exist_ok=True, parents=True)

                video_out = carpeta_videos_tipo / f"{subnombre}.mp4"
                csv_out   = carpeta_videos_tipo / f"{subnombre}.csv"

            # Instanciar y configurar degrader como en opción 2, pero con preset
            degrader = VideoDegrader(str(temp_video))
            degrader.apply_vignette_globally = preset["apply_vignette_globally"]
            degrader.blob_prob = preset["blob_prob"]
            degrader.scratch_prob = preset["scratch_prob"]
            min_sc, max_sc = preset["scratch_count"]
            degrader.scratch_count = random.randint(min_sc, max_sc) if max_sc > 0 else 0
            degrader.scratch_positions = [random.randint(0, degrader.width) for _ in range(degrader.scratch_count)]
            degrader.color_tone = preset["color_tone"]
            degrader.color_tone_strength = preset["color_tone_strength"]
            degrader.apply_clean_bw = preset["apply_clean_bw"]
            degrader.apply_soft_bw = preset["apply_soft_bw"]

            era = preset["era"]
            original_profile = degrader.profiles[era].copy()
            try:
                degrader.profiles[era].update(preset.get("override_profile", {}))

                # 3) Salidas (misma lógica que opción 2, pero en subcarpeta de tipo)
             
                degrader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                degrader.apply_temporal_degradation(str(video_out), era=era)

                # CSV de config
                with open(csv_out, 'w', newline='') as fcsv:
                    writer = csv.DictWriter(
                        fcsv,
                        fieldnames=[
                            'description','output_file','era','degradation_type',
                            'apply_vignette_globally','blob_prob','scratch_prob',
                            'color_tone','color_tone_strength','apply_clean_bw',
                            'apply_soft_bw','scratch_count','scratch_positions'
                        ]
                    )
                    writer.writeheader()
                    writer.writerow({
                        'description': f'Preset {tipo} para {escena.name}',
                        'output_file': video_out.name,
                        'era': era,
                        'degradation_type': tipo,
                        'apply_vignette_globally': degrader.apply_vignette_globally,
                        'blob_prob': degrader.blob_prob,
                        'scratch_prob': degrader.scratch_prob,
                        'color_tone': degrader.color_tone,
                        'color_tone_strength': degrader.color_tone_strength,
                        'apply_clean_bw': degrader.apply_clean_bw,
                        'apply_soft_bw': degrader.apply_soft_bw,
                        'scratch_count': degrader.scratch_count,
                        'scratch_positions': ','.join(map(str, degrader.scratch_positions)),
                    })

                # Guardar fotogramas degradados (igual que opción 2)
                cap = cv2.VideoCapture(str(video_out))
                idx = 0
                while True:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(str(carpeta_frames_tipo / f"{idx:08d}.png"), fr)
                    idx += 1
                cap.release()

                print(f"✅ {escena.name} [{tipo}] → {video_out} + {csv_out} + {idx} frames")
            finally:
                degrader.profiles[era] = original_profile

        # 4) borrar temp
        try:
            temp_video.unlink()
        except Exception as e:
            print(f"[WARN] No se pudo borrar {temp_video}: {e}")

    print("🎬 Todas las escenas han sido procesadas por TIPOS con el mismo flujo que la opción 2.")


def build_custom_preset(tokens):
    # Valores base neutros
    preset = {
        "era": "1950s",
        "apply_vignette_globally": False,
        "blob_prob": 0.0,
        "scratch_prob": 0.0,
        "scratch_count": (0, 0),
        "color_tone": None,
        "color_tone_strength": 0.0,
        "apply_clean_bw": False,
        "apply_soft_bw": False,
        "override_profile": {}
    }

    tokens = [t.strip().lower() for t in tokens if t.strip()]
    for t in tokens:
        if t == "scratches":
            preset["scratch_prob"] = max(preset["scratch_prob"], 0.9)
            preset["scratch_count"] = (3, 6)
        elif t == "flicker":
            preset["override_profile"]["flicker_intensity"] = max(
                preset["override_profile"].get("flicker_intensity", 0.0), 0.3
            )
            preset["era"] = "1920s"
        elif t == "blobs":
            preset["blob_prob"] = max(preset["blob_prob"], 0.8)
        elif t == "vignette":
            preset["apply_vignette_globally"] = True
        elif t == "sepia":
            preset["color_tone"] = "sepia"
            preset["color_tone_strength"] = max(preset["color_tone_strength"], 0.5)
        elif t == "blue":
            preset["color_tone"] = "blue"
            preset["color_tone_strength"] = max(preset["color_tone_strength"], 0.45)
        elif t == "rose":
            preset["color_tone"] = "rose"
            preset["color_tone_strength"] = max(preset["color_tone_strength"], 0.35)
        elif t == "clean_bw":
            preset["apply_clean_bw"] = True
        elif t == "soft_bw":
            preset["apply_soft_bw"] = True
        elif t == "vhs":
            preset["era"] = "vhs"
        elif t == "burn":
            preset["override_profile"]["burn_prob"] = max(
                preset["override_profile"].get("burn_prob", 0.0), 0.06
            )
        # Puedes extender con más tokens: "jitter_strong", "motion_blur", etc.

    return preset


def generar_degradaciones_imagen(
    image_path,
    output_dir="/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degradacionesimg",
    presets=None,
    scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches")
):
    """
    Genera una imagen por cada preset de degradación a partir de una imagen de entrada.
    - Guarda también la imagen original en la misma carpeta.
    - Los nombres siguen el patrón: <base>_original.png y <base>_<preset>.png
    """
    if presets is None:
        presets = DEGRADATION_PRESETS

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar imagen de entrada
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    h, w = img.shape[:2]
    base = image_path.stem

    # Guardar original
    out_orig = output_dir / f"{base}_original.png"
    cv2.imwrite(str(out_orig), img)

    # Crear vídeo temporal de 1 frame con la imagen (para reutilizar el pipeline)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_dir = output_dir / f"__tmp_{base}"
    temp_dir.mkdir(exist_ok=True)
    temp_video_in = temp_dir / f"{base}_in.mp4"
    vw = cv2.VideoWriter(str(temp_video_in), fourcc, 15, (w, h))
    vw.write(img)
    vw.release()

    # Para cada preset, aplicar la misma lógica de VideoDegrader y extraer frame
    for tipo, preset in presets.items():
        # Video temporal de salida
        temp_video_out = temp_dir / f"{base}_{tipo}.mp4"

        # Instanciar y configurar degrader
        degrader = VideoDegrader(str(temp_video_in))
        degrader.scratch_dir = scratch_dir  # asegurar ruta
        degrader.apply_vignette_globally = preset["apply_vignette_globally"]
        degrader.blob_prob = preset["blob_prob"]
        degrader.scratch_prob = preset["scratch_prob"]

        # scratch_count puede ser tupla (min,max)
        min_sc, max_sc = preset["scratch_count"]
        degrader.scratch_count = random.randint(min_sc, max_sc) if max_sc > 0 else 0
        degrader.scratch_positions = [random.randint(0, degrader.width) for _ in range(degrader.scratch_count)]

        degrader.color_tone = preset["color_tone"]
        degrader.color_tone_strength = preset["color_tone_strength"]
        degrader.apply_clean_bw = preset["apply_clean_bw"]
        degrader.apply_soft_bw = preset["apply_soft_bw"]

        # era y posibles overrides
        era = preset["era"]
        original_profile = degrader.profiles[era].copy()
        try:
            degrader.profiles[era].update(preset.get("override_profile", {}))
            degrader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            degrader.apply_temporal_degradation(str(temp_video_out), era=era)
        finally:
            degrader.profiles[era] = original_profile

        # Extraer el primer (y único) frame y guardarlo como PNG
        cap = cv2.VideoCapture(str(temp_video_out))
        ret, frm = cap.read()
        cap.release()
        if ret and frm is not None:
            out_png = output_dir / f"{base}_{tipo}.png"
            cv2.imwrite(str(out_png), frm)
            print(f"✅ Guardado: {out_png}")
        else:
            print(f"⚠️ No se pudo leer salida para preset '{tipo}'")

    # Limpieza temporal
    try:
        for f in temp_dir.glob("*"):
            try: f.unlink()
            except: pass
        temp_dir.rmdir()
    except Exception as e:
        print(f"[WARN] Limpieza de temporales con advertencias: {e}")

    print(f"🎉 Listo. Imágenes en: {output_dir}")


# ------------------------
# EJECUCIÓN PRINCIPAL
# ------------------------
# === MODO PRINCIPAL CON MENÚ INTERACTIVO SIMPLE ===

def main():
    print("Selecciona modo de operación:")
    print("1. Procesar DAVIS automáticamente")
    print("2. Procesar carpeta personalizada (aleatorio con CSV)")
    print("3. Procesar carpeta personalizada POR TIPOS (misma entrada/salida que opción 2)")
    print("4. Generar DIRECTAMENTE las 5 bases de datos fijas (para entrenamiento)")
    print("5. Generar degradaciones atomizadas para una imagen única")
    print("6. Generar overlays/plantillas por escenas (transparencias)")
    print("7. Combinar fotogramas con overlays en una carpeta destino")
    print("8. Crear video MP4 desde una carpeta de fotogramas")
    print("9. Eliminar archivos metadata.json en subcarpetas")
    print("10. Generar overlays desde una imagen única (RGBA + RRTN + compuesta)")
    print("11. Copiar imagen original y restauraciones RRTN (test_results_30_rec1)")

    modo = input("Elige 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 u 11: ").strip()

    if modo == "1":
        # ... (igual que tu código actual)
        davis_paths = download_davis_dataset()
        annotations_base = davis_paths["Annotations"]
        images_base = davis_paths["JPEGImages"]
        output_root = "video_variaciones"
        os.makedirs(output_root, exist_ok=True)
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
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/train\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/train"
        carpeta_videos = input(
            "Carpeta de salida para videos y CSV:\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/videos/train\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/videos/train"
        carpeta_fotogramas = input(
            "Carpeta de salida para fotogramas degradados (uno por escena):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/train\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/train"
        num_versiones = int(input("¿Cuántas versiones degradadas por escena quieres generar? [1]: ") or "1")
        procesar_carpeta_personalizada(
            carpeta_entrada, carpeta_videos, carpeta_fotogramas, num_versiones=num_versiones
        )

    elif modo == "3":
        carpeta_entrada = input(
            "Ruta a la carpeta raíz de escenas (subcarpetas):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/train\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/train"
        carpeta_videos = input(
            "Carpeta base de salida para videos (igual que opción 2):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/videos/train\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/videos/train"
        carpeta_fotogramas = input(
            "Carpeta base de salida para fotogramas (igual que opción 2):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/train\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/train"

        tipos_txt = input(
            "Lista de tipos separados por coma (ENTER para todos):\n"
            f"Disponibles: {', '.join(DEGRADATION_PRESETS.keys())}\n> "
        ).strip()
        if tipos_txt:
            tipos = [t.strip() for t in tipos_txt.split(",") if t.strip() in DEGRADATION_PRESETS]
            if not tipos:
                print("⚠️ Ningún tipo válido detectado. Usando todos.")
                tipos = list(DEGRADATION_PRESETS.keys())
        else:
            tipos = list(DEGRADATION_PRESETS.keys())

        procesar_carpeta_por_tipos_compatible(
            carpeta_entrada, carpeta_videos, carpeta_fotogramas, tipos=tipos
        )
    elif modo == "4":
        carpeta_entrada = input(
            "Ruta a la carpeta raíz de escenas (subcarpetas, como en opción 2):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/val\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/val"

        carpeta_videos_base = input(
            "Carpeta base de salida para videos (igual que opción 2):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/videos/val\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/videos/val"

        carpeta_fotogramas_base = input(
            "Carpeta base de salida para fotogramas (se adaptará por dataset):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO"

        print("🚀 Generando TODAS las bases de datos fijas...")
        for nombre, token in FIXED_DATASETS.items():
            print(f"\n=== Generando dataset {nombre} ({token}) ===")

            # Carpeta dinámica para este dataset
            carpeta_videos = Path(carpeta_videos_base).parent / f"{Path(carpeta_videos_base).parent.name}_{nombre}" / "videos/train"
            carpeta_fotogramas = Path(carpeta_fotogramas_base + f"_{nombre}") / "val"

            carpeta_videos.mkdir(parents=True, exist_ok=True)
            carpeta_fotogramas.mkdir(parents=True, exist_ok=True)

            # construimos presets_ext local
            presets_ext = dict(DEGRADATION_PRESETS)
            tipos = []
            if token.lower().startswith("custom:"):
                tokens = token.split(":", 1)[1].split("+")
                preset = build_custom_preset(tokens)
                nombre_tipo = nombre  # usamos el nombre del dataset como sufijo
                presets_ext[nombre_tipo] = preset
                tipos = [nombre_tipo]
            else:
                tipos = [token]

            procesar_carpeta_por_tipos_compatible(
                carpeta_entrada,
                carpeta_videos,
                carpeta_fotogramas,
                tipos=tipos,
                presets=presets_ext
            )

        print("✅ Todas las bases de datos fijas han sido generadas.")


    elif modo == "5":
        ruta_imagen = input(
            "Ruta de la imagen de entrada:\n"
            "ENTER para usar la ruta por defecto\n"
            "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train/017/00000096.png\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train/017/00000096.png"

        carpeta_salida = input(
            "Carpeta de salida (ENTER para la predeterminada):\n"
            "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degradacionesimg\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degradacionesimg"

        generar_degradaciones_imagen(
            ruta_imagen,
            output_dir=carpeta_salida,
            presets=ATOMIC_PRESETS,  # usa todos los presets definidos
            scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches")
        )

    elif modo == "6":
        print("Sub-modos disponibles para overlays:")
        print("  1. Generar overlays por preset (modo original, una plantilla por preset).")
        print("  2. Generar plantillas coherentes por escena (vigneta + arañazos + blobs).")
        submodo = input("Elige 1 o 2 [2]: ").strip() or "2"
        export_rrtn = (input("¿Guardar solo la versión estilo RRTN (grises sin alpha)? [s/N]: ").strip().lower() or "n") == "s"

        if submodo == "1":
            carpeta_entrada = input(
                "Ruta a la carpeta raíz de escenas (subcarpetas):\n"
                "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/RTTN/gt/val\n> "
            ).strip() or "/home/laura/CycleGAN/00Databases/REDS/capaTransparente"

            carpeta_salida = input(
                "Carpeta base de salida para overlays:\n"
                "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/MIO/overlays/val\n> "
            ).strip() or "/home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer"

            procesar_carpeta_overlays(
                carpeta_entrada,
                carpeta_salida,
                tipos=list(OVERLAY_PRESETS.keys()),
                presets=OVERLAY_PRESETS,
                export_rrtn_texture=export_rrtn,
                only_rrtn_texture=export_rrtn
            )
        else:
            carpeta_entrada = input(
                "Ruta a la carpeta raíz de escenas (subcarpetas):\n"
                "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train\n> "
            ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train"

            carpeta_salida = input(
                "Carpeta base de salida para las plantillas RGBA:\n"
                "Ej.: /home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer\n> "
            ).strip() or "/home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer"

            inc_vignette = (input("¿Incluir viñeta global? [S/n]: ").strip().lower() or "s") != "n"
            inc_scratches = (input("¿Incluir arañazos? [S/n]: ").strip().lower() or "s") != "n"
            inc_blobs = (input("¿Incluir blobs/gotas? [S/n]: ").strip().lower() or "s") != "n"
            save_meta = (input("¿Guardar metadata JSON por escena? [S/n]: ").strip().lower() or "s") != "n"

            size_prompt = (
                "Tamaño final de las plantillas (px). "
                "Ej.: 512, 512x512, original. ENTER para usar 512x512:\n> "
            )
            size_txt = input(size_prompt).strip().lower()
            target_size = (512, 512)
            if size_txt in {"", "512", "512x512"}:
                target_size = (512, 512)
            elif size_txt in {"original", "orig", "same", "igual"}:
                target_size = None
            else:
                try:
                    if "x" in size_txt:
                        h_str, w_str = size_txt.lower().split("x", 1)
                        target_size = (int(h_str), int(w_str))
                    else:
                        val = int(size_txt)
                        target_size = (val, val)
                except ValueError:
                    print("⚠️ Tamaño inválido, se usará 512x512 por defecto.")
                    target_size = (512, 512)

            if target_size is not None and (target_size[0] < 1 or target_size[1] < 1):
                print("⚠️ Tamaño no válido. Se mantiene el tamaño original de los frames.")
                target_size = None

            procesar_carpeta_overlays_coherentes(
                carpeta_entrada,
                carpeta_salida,
                include_vignette=inc_vignette,
                include_scratches=inc_scratches,
                include_blobs=inc_blobs,
                guardar_metadata=save_meta,
                scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches"),
                target_size=target_size,
                export_rrtn_texture=export_rrtn,
                only_rrtn_texture=export_rrtn
            )

    elif modo == "7":
        carpeta_base = input(
            "Ruta con los fotogramas originales:\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train/000\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train/000"

        carpeta_overlays = input(
            "Ruta con las capas transparentes (mismos nombres):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer/0001\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer/0001"

        overlay_is_rgba = (input(
            "¿Las plantillas tienen transparencia RGBA? [S/n]: "
        ).strip().lower() or "s") != "n"
        overlay_mode = "rgba" if overlay_is_rgba else "grayscale"
        grayscale_strength = 0.6
        if overlay_mode == "grayscale":
            try:
                grayscale_strength = float(input(
                    "Intensidad al aplicar la plantilla gris (0.0–1.0) [0.6]: "
                ).strip() or "0.6")
            except ValueError:
                grayscale_strength = 0.6

        assign_choice = input(
            "¿Cómo emparejar las plantillas? 1=por nombre, 2=por orden [2]: "
        ).strip() or "2"
        assign_mode = "name" if assign_choice == "1" else "sorted"
        reverse_sort = False
        if assign_mode == "sorted":
            reverse_sort = (input(
                "¿Ordenar de mayor a menor antes de emparejar? [s/N]: "
            ).strip().lower() or "n") == "s"

        carpeta_salida = input(
            "Carpeta donde guardar las combinaciones:\n"
            "Ej.: /home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degracacionesCapa\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degracacionesCapa"

        procesar_subcarpetas = (input(
            "¿Procesar automáticamente todas las subcarpetas (escenas)? [s/N]: "
        ).strip().lower() or "n") == "s"

        if procesar_subcarpetas:
            combinar_subcarpetas_base_overlay(
                carpeta_base,
                carpeta_overlays,
                carpeta_salida,
                overlay_mode=overlay_mode,
                grayscale_strength=grayscale_strength,
                assign_mode=assign_mode,
                reverse_sort=reverse_sort
            )
        else:
            combinar_carpeta_base_overlay(
                carpeta_base,
                carpeta_overlays,
                carpeta_salida,
                overlay_mode=overlay_mode,
                grayscale_strength=grayscale_strength,
                assign_mode=assign_mode,
                reverse_sort=reverse_sort
            )

    elif modo == "8":
        carpeta_frames = input(
            "Carpeta con los fotogramas (png/jpg, mismo tamaño):\n"
            "Ej.: /home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degracacionesCapa\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/degracacionesCapa"

        fps = float(input("FPS del video [15]: ").strip() or "15")
        salida = input(
            "Ruta completa del video de salida (ENTER para usar la misma carpeta):\n"
            "Ej.: /ruta/al/video.mp4\n> "
        ).strip()
        salida_video = salida if salida else None

        try:
            crear_video_desde_frames(
                carpeta_frames,
                salida_video=salida_video,
                fps=fps
            )
        except Exception as exc:
            print(f"⚠️ Error al crear el video: {exc}")

    elif modo == "9":
        carpeta_raiz = input(
            "Carpeta raíz donde eliminar metadata.json en subcarpetas:\n"
            "Ej.: /home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/capaTransparenteLayer"
        if not carpeta_raiz:
            print("⚠️ Debes proporcionar una ruta válida.")
        else:
            ruta = Path(carpeta_raiz)
            if not ruta.exists() or not ruta.is_dir():
                print(f"⚠️ Ruta inválida: {ruta}")
            else:
                eliminados = 0
                for metadata_file in ruta.rglob("metadata.json"):
                    try:
                        metadata_file.unlink()
                        eliminados += 1
                    except Exception as exc:
                        print(f"⚠️ No se pudo eliminar {metadata_file}: {exc}")
                print(f"🧹 Archivos metadata.json eliminados: {eliminados}")

    elif modo == "10":
        ruta_imagen = input(
            "Ruta de la imagen base (ENTER para usar la predeterminada):\n"
            "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train/000/00000007.png\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/REDS/COMPARACION/192x1921Channel/RTTN/gt/train/000/00000007.png"

        carpeta_salida = input(
            "Carpeta donde guardar base, overlays y combinaciones (ENTER para usar la predeterminada):\n"
            "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/combinacionpaper\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/combinacionpaper"

        export_rrtn = (input(
            "¿Generar también la textura estilo RRTN (grayscale sin alpha)? [S/n]: "
        ).strip().lower() or "s") != "n"

        include_composed = (input(
            "¿Guardar la imagen base combinada con cada overlay? [S/n]: "
        ).strip().lower() or "s") != "n"

        try:
            generar_overlays_imagen(
                ruta_imagen,
                output_dir=carpeta_salida,
                presets=OVERLAY_PRESETS,
                scratch_dir=Path("/home/laura/CycleGAN/00Databases/plantillasScratches"),
                export_rrtn_texture=export_rrtn,
                compose_and_save=include_composed
            )
            print(f"✅ Overlays generados en: {Path(carpeta_salida).resolve()}")
            print("   Incluye copia de la imagen base, PNG RGBA por preset, texturas RRTN y composiciones.")
        except Exception as exc:
            print(f"⚠️ Error al generar overlays desde imagen única: {exc}")

    elif modo == "11":
        ruta_imagen = input(
            "Ruta de la imagen original (ENTER para usar la predeterminada):\n"
            "/home/laura/CycleGAN/00Databases/escenas_probarRRTN/nosferatu_escena017/00015.jpg\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/escenas_probarRRTN/nosferatu_escena017/00015.jpg"

        carpeta_output = input(
            "Carpeta OUTPUT donde buscar restauraciones (ENTER para usar la predeterminada):\n"
            "/home/laura/CycleGAN/RRTN-old-film-restoration/OUTPUT\n> "
        ).strip() or "/home/laura/CycleGAN/RRTN-old-film-restoration/OUTPUT"

        carpeta_salida = input(
            "Carpeta donde copiar original y restauraciones (ENTER predeterminada):\n"
            "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/combinacionpaper/imgoriginal_restaurada\n> "
        ).strip() or "/home/laura/CycleGAN/00Databases/00Herramientas/imgSinteticasTesteo/combinacionpaper/imgoriginal_restaurada"

        try:
            destino_original, hallazgos = recolectar_restauraciones_imagen(
                ruta_imagen,
                carpeta_output_root=Path(carpeta_output),
                carpeta_salida=Path(carpeta_salida),
                subcarpeta_resultados="test_results_30_rec1"
            )
            print(f"📸 Original copiada en: {destino_original}")
            if not hallazgos:
                print("⚠️ No se encontraron restauraciones en test_results_30_rec1 para esa ruta.")
            else:
                print("✅ Restauraciones copiadas:")
                for info in hallazgos:
                    print(f"   - {info['batch']}/{info['modelo']} → {info['destino'].name}")
                    if info.get("relativo"):
                        print(f"     (subruta encontrada: {info['relativo']})")
        except Exception as exc:
            print(f"⚠️ Error al recopilar restauraciones: {exc}")

    else:
        print("Opción inválida. Ejecuta de nuevo el script.")


if __name__ == "__main__":
    main()
