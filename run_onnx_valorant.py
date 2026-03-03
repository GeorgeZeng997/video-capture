import argparse
import os
import time
from typing import List, Tuple

import cv2
import numpy as np


def preload_cuda_dependencies() -> None:
    """尽量从本机现有环境预加载 CUDA/cuDNN 依赖，优先复用 torch 自带 DLL。"""
    try:
        import torch  # noqa: F401

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            print(f"[CUDA] add_dll_directory: {torch_lib}")
    except Exception as e:
        print(f"[CUDA] torch DLL preload skipped: {e!r}")


preload_cuda_dependencies()
import onnxruntime as ort


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(frame: np.ndarray, imgsz: int):
    img, ratio, dwdh = letterbox(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, ratio, dwdh


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-7
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    idxs = np.argsort(-scores)
    keep: List[int] = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(int(i))
        if idxs.size == 1:
            break
        ious = box_iou_xyxy(boxes[i:i + 1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thres]
    return keep


def postprocess(
    pred: np.ndarray,
    ratio: float,
    dwdh: Tuple[float, float],
    conf_thres: float,
    iou_thres: float,
    orig_shape: Tuple[int, int],
):
    # pred: [1, 25200, 6] -> xywh + obj + cls(1)
    pred = pred[0]
    if pred.shape[-1] < 5:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if pred.shape[-1] == 6:
        scores = pred[:, 4] * pred[:, 5]
    else:
        cls_scores = pred[:, 5:]
        scores = pred[:, 4] * cls_scores.max(axis=1)

    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = pred[mask, :4]
    scores = scores[mask]

    # xywh -> xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(boxes_xyxy, scores, iou_thres)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    # undo letterbox
    dw, dh = dwdh
    boxes_xyxy[:, [0, 2]] -= dw
    boxes_xyxy[:, [1, 3]] -= dh
    boxes_xyxy /= ratio

    h0, w0 = orig_shape
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0 - 1)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0 - 1)
    return boxes_xyxy, scores


def get_roi(frame: np.ndarray, mode: str, ratio: float, rx: int, ry: int, rw: int, rh: int):
    h, w = frame.shape[:2]
    if mode == "full":
        return 0, 0, w, h, frame

    if mode == "center_ratio":
        ratio = max(0.1, min(1.0, ratio))
        rw = max(1, int(w * ratio))
        rh = max(1, int(h * ratio))
        x0 = max(0, (w - rw) // 2)
        y0 = max(0, (h - rh) // 2)
        roi = frame[y0:y0 + rh, x0:x0 + rw]
        return x0, y0, rw, rh, roi

    # manual_rect
    x0 = max(0, min(rx, w - 1))
    y0 = max(0, min(ry, h - 1))
    ww = max(1, min(rw, w - x0))
    hh = max(1, min(rh, h - y0))
    roi = frame[y0:y0 + hh, x0:x0 + ww]
    return x0, y0, ww, hh, roi


def main():
    parser = argparse.ArgumentParser(description="Valorant ONNX + 采集卡实时检测")
    parser.add_argument(
        "--model",
        type=str,
        default=r"D:\download\YOLO-Models-For-Valorant-main\YOLO-Models-For-Valorant-main\Yolov5\YOLOv5s\valorant-11.onnx",
        help="ONNX模型路径",
    )
    parser.add_argument("--source", type=int, default=0, help="视频源索引")
    parser.add_argument("--backend", type=str, default="any", choices=["any", "msmf", "dshow"])
    parser.add_argument("--width", type=int, default=1920, help="采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="采集高度")
    parser.add_argument("--fps", type=int, default=60, help="采集帧率目标")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="推理设备")
    parser.add_argument("--infer", type=str, default="on", choices=["on", "off"], help="是否启用推理")
    parser.add_argument("--roi-mode", type=str, default="full", choices=["full", "center_ratio", "manual_rect"], help="推理区域模式")
    parser.add_argument("--roi-ratio", type=float, default=0.5, help="中心区域比例(0~1], 仅 center_ratio 生效")
    parser.add_argument("--roi-x", type=int, default=0, help="手动ROI左上角x")
    parser.add_argument("--roi-y", type=int, default=0, help="手动ROI左上角y")
    parser.add_argument("--roi-w", type=int, default=960, help="手动ROI宽")
    parser.add_argument("--roi-h", type=int, default=540, help="手动ROI高")
    args = parser.parse_args()

    print(f"[BOOT] model={args.model}")
    print(
        f"[BOOT] source={args.source}, backend={args.backend}, "
        f"capture={args.width}x{args.height}@{args.fps}, "
        f"imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, device={args.device}, infer={args.infer}, "
        f"roi_mode={args.roi_mode}, roi_ratio={args.roi_ratio}, roi_rect=({args.roi_x},{args.roi_y},{args.roi_w},{args.roi_h})"
    )

    print(f"[ONNX] available_providers={ort.get_available_providers()}")
    if args.device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        # auto: 优先CUDA，失败自动回落CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(args.model, providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f"[ONNX] providers={sess.get_providers()}, input={input_name}")

    backend_map = {
        "any": cv2.CAP_ANY,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
    }
    cap = cv2.VideoCapture(args.source, backend_map[args.backend])
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源 source={args.source}, backend={args.backend}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        "[CAPTURE] negotiated "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        f"@{cap.get(cv2.CAP_PROP_FPS):.2f}"
    )

    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] 读取帧失败")
            break

        x0, y0, rw, rh, roi_frame = get_roi(
            frame,
            args.roi_mode,
            args.roi_ratio,
            args.roi_x,
            args.roi_y,
            args.roi_w,
            args.roi_h,
        )

        if args.infer == "on":
            x, ratio, dwdh = preprocess(roi_frame, args.imgsz)
            pred = sess.run(None, {input_name: x})[0]
            boxes, scores = postprocess(pred, ratio, dwdh, args.conf, args.iou, roi_frame.shape[:2])

            if len(boxes):
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0

            for (x1, y1, x2, y2), s in zip(boxes.astype(int), scores):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"target {s:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            boxes = []

        if args.roi_mode in ("center_ratio", "manual_rect"):
            cv2.rectangle(frame, (x0, y0), (x0 + rw, y0 + rh), (255, 180, 0), 2)

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_t)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}  DET: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2)
        cv2.imshow("Valorant ONNX Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

