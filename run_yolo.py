import argparse
import time
import cv2
from ultralytics import YOLO


BACKEND_MAP = {
    "any": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}


def get_roi(frame, mode: str, ratio: float, rx: int, ry: int, rw: int, rh: int):
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


def _try_open(source: int, backend_code: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, backend_code)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_capture(source: int, width: int, height: int, fps: int, backend: str) -> cv2.VideoCapture:
    if backend == "auto":
        # Windows下优先使用CAP_ANY和MSMF，DSHOW作为最后兜底
        candidates = ["any", "msmf", "dshow"]
    else:
        candidates = [backend]

    last_cap = None
    for name in candidates:
        backend_code = BACKEND_MAP[name]
        cap = _try_open(source, backend_code, width, height, fps)
        opened = cap.isOpened()
        print(f"[CAPTURE] try backend={name}, source={source}, opened={opened}")
        if opened:
            return cap
        last_cap = cap
        cap.release()

    if last_cap is not None:
        return last_cap
    return cv2.VideoCapture(source)


def main():
    parser = argparse.ArgumentParser(description="USB采集卡实时YOLO推理")
    parser.add_argument("--source", type=int, default=0, help="摄像头/采集卡索引，如0/1/2")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO模型名或本地模型路径")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "any", "msmf", "dshow"],
        help="视频后端，Windows建议auto/msmf",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="推理设备")
    parser.add_argument("--infer", type=str, default="on", choices=["on", "off"], help="是否启用推理")
    parser.add_argument("--roi-mode", type=str, default="full", choices=["full", "center_ratio", "manual_rect"], help="推理区域模式")
    parser.add_argument("--roi-ratio", type=float, default=0.5, help="中心区域比例(0~1], 仅 center_ratio 生效")
    parser.add_argument("--roi-x", type=int, default=0, help="手动ROI左上角x")
    parser.add_argument("--roi-y", type=int, default=0, help="手动ROI左上角y")
    parser.add_argument("--roi-w", type=int, default=960, help="手动ROI宽")
    parser.add_argument("--roi-h", type=int, default=540, help="手动ROI高")
    args = parser.parse_args()

    print(f"[BOOT] OpenCV={cv2.__version__}")
    print(
        f"[BOOT] source={args.source}, backend={args.backend}, "
        f"size={args.width}x{args.height}@{args.fps}, imgsz={args.imgsz}, model={args.model}, conf={args.conf}, "
        f"device={args.device}, infer={args.infer}, roi_mode={args.roi_mode}, roi_ratio={args.roi_ratio}, "
        f"roi_rect=({args.roi_x},{args.roi_y},{args.roi_w},{args.roi_h})"
    )

    model = YOLO(args.model)
    cap = open_capture(args.source, args.width, args.height, args.fps, args.backend)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频源 source={args.source}。"
            f"可尝试 --source 1/2，或 --backend msmf/any。"
        )

    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取帧失败，检查采集卡输入信号")
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

        det_count = 0
        if args.infer == "on":
            infer_device = None if args.device == "auto" else args.device
            results = model.predict(roi_frame, imgsz=args.imgsz, conf=args.conf, device=infer_device, verbose=False)
            annotated = results[0].plot()
            if hasattr(results[0], "boxes") and results[0].boxes is not None:
                det_count = len(results[0].boxes)
            if args.roi_mode == "center_ratio":
                frame[y0:y0 + rh, x0:x0 + rw] = annotated
                annotated = frame
        else:
            annotated = frame

        if args.roi_mode in ("center_ratio", "manual_rect"):
            cv2.rectangle(annotated, (x0, y0), (x0 + rw, y0 + rh), (255, 180, 0), 2)

        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_t)
        prev_t = now
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}  DET: {det_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (50, 220, 50),
            2,
        )

        cv2.imshow("YOLO Live", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

