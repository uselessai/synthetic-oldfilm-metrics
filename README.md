# Synthetic Old-Film Metrics

**Synthetic Old-Film Metrics** is a modular framework designed to generate and evaluate artificial degradations inspired by historical film artefacts.  
The project provides reproducible components to simulate typical visual defects (e.g., scratches, dust, blobs, vignetting, tonal drifts) and assess how restoration models respond to them.

### Overview

Most modern video restoration methods rely on large paired datasets that do not reflect the complex artefacts of archival footage.  
This toolkit aims to fill that gap by offering a flexible, script-based environment to compose *historically inspired* degradations on clean datasets such as REDS or Vimeo.  
It also includes metric utilities to benchmark restoration quality across fidelity, perceptual, and temporal dimensions.

### Features

- **Modular artefact generators:** reusable Python modules for scratches, blobs, vignette, flicker, and burn-in.
- **Procedural composition:** reproducible overlays built with deterministic seeds.
- **Metric suite:** wrappers for fidelity (PSNR, SSIM), perceptual (LPIPS), and no-reference scores (BRISQUE, NIQE, etc.).
- **Extensible evaluation loop:** can integrate with external backbones such as RRTN, BasicVSR, or VRT.

### License & Citation

This repository is provided for academic use and ongoing research on video restoration under synthetic degradation.  
If you use or adapt parts of this code, please cite the corresponding publication once it becomes available.

---

🧩 *Work in progress — additional documentation and scripts will be released after the related publication.*


