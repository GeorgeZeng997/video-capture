# video-capture yolo-live

A desktop project for **real-time inference on HDMI capture card video streams**.

Supported pipelines:

- ONNX inference (`onnxruntime`)
- PT inference (`ultralytics`)
- GUI-based runtime control (model type, device, ROI, capture settings, inference settings)

## Features

- Capture card video input (`any/msmf/dshow`)
- Inference device switch (`auto/cuda/cpu`)
- Inference toggle (`on/off`)
- ROI inference modes:
  - `full`
  - `center_ratio`
  - `manual_rect`
- Startup/runtime logs with system and environment details

## Project Files

- `gui_launcher.py`: GUI launcher and runtime controller
- `run_onnx_valorant.py`: ONNX inference pipeline
- `run_yolo.py`: PT inference pipeline
- `requirements.txt`: dependency list
- `v11s.pt`: PT model file
- `valorant-11.onnx`: ONNX model file

## Quick Start

```bash
python -m pip install -r requirements.txt
python gui_launcher.py
```

## Notes

- If `device=cuda` is selected but CUDA is unavailable in the active interpreter, the launcher applies a fallback policy.
- `msmf` is generally recommended as the preferred capture backend on Windows.

