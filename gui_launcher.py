import queue
import os
import platform
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Valorant 推理控制台（ONNX / PT）")
        self.root.geometry("860x620")

        self.base_dir = Path(__file__).resolve().parent
        self.python_exec = self._resolve_python_exec()
        self.runner_onnx = self.base_dir / "run_onnx_valorant.py"
        self.runner_pt = self.base_dir / "run_yolo.py"
        self.proc = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.sys_info = {}

        self._build_ui()
        self._detect_and_log_system_info()
        self._poll_logs()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    @staticmethod
    def _probe_python_cuda(py_exec: str):
        code, out = App._run_cmd(
            [
                py_exec,
                "-c",
                "import torch,sys; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())",
            ]
        )
        if code != 0 or not out:
            return {"ok": False, "exec": py_exec, "cuda": False, "ver": "", "count": 0}

        lines = [x.strip() for x in out.splitlines() if x.strip()]
        if len(lines) < 4:
            return {"ok": False, "exec": py_exec, "cuda": False, "ver": "", "count": 0}

        return {
            "ok": True,
            "exec": lines[0],
            "ver": lines[1],
            "cuda": lines[2].lower() == "true",
            "count": int(lines[3]) if lines[3].isdigit() else 0,
        }

    def _resolve_python_exec(self) -> str:
        """优先选 CUDA 可用的 Python 解释器。"""
        candidates = []
        if sys.executable:
            candidates.append(sys.executable)
        candidates.extend(
            [
                r"C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
                r"D:\python\python.exe",
                "python",
            ]
        )

        seen = set()
        ordered = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)

        best = None
        fallback = None
        for c in ordered:
            info = self._probe_python_cuda(c)
            if info["ok"] and fallback is None:
                fallback = info
            if info["ok"] and info["cuda"]:
                best = info
                break

        chosen = best or fallback
        if chosen:
            return chosen["exec"]
        return sys.executable or "python"

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        self.model_var = tk.StringVar(
            value=r"D:\download\YOLO-Models-For-Valorant-main\YOLO-Models-For-Valorant-main\Yolov5\YOLOv5s\valorant-11.onnx"
        )
        self.model_type_var = tk.StringVar(value="onnx")
        self.source_var = tk.StringVar(value="0")
        self.backend_var = tk.StringVar(value="msmf")
        self.width_var = tk.StringVar(value="1920")
        self.height_var = tk.StringVar(value="1080")
        self.fps_var = tk.StringVar(value="60")
        self.imgsz_var = tk.StringVar(value="640")
        self.conf_var = tk.StringVar(value="0.35")
        self.iou_var = tk.StringVar(value="0.45")
        self.infer_var = tk.StringVar(value="on")
        self.device_var = tk.StringVar(value="auto")
        self.roi_mode_var = tk.StringVar(value="full")
        self.roi_ratio_var = tk.StringVar(value="0.5")
        self.roi_x_var = tk.StringVar(value="0")
        self.roi_y_var = tk.StringVar(value="0")
        self.roi_w_var = tk.StringVar(value="960")
        self.roi_h_var = tk.StringVar(value="540")
        self.roi_ratio_entry = None
        self.roi_rect_entries = []

        row = 0
        ttk.Label(frm, text="模型类型:").grid(row=row, column=0, sticky="w", pady=4)
        type_box = ttk.Combobox(
            frm,
            textvariable=self.model_type_var,
            values=["onnx", "pt"],
            state="readonly",
            width=12,
        )
        type_box.grid(row=row, column=1, sticky="w", pady=4)
        type_box.bind("<<ComboboxSelected>>", lambda _e: self._on_model_type_change())

        row += 1
        ttk.Label(frm, text="模型路径:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self.model_var, width=80).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(frm, text="浏览", command=self.pick_model).grid(row=row, column=2, padx=6, pady=4)

        row += 1
        ttk.Label(frm, text="视频源索引:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self.source_var, width=10).grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        ttk.Label(frm, text="后端:").grid(row=row, column=0, sticky="w", pady=4)
        backend_box = ttk.Combobox(
            frm,
            textvariable=self.backend_var,
            values=["any", "msmf", "dshow"],
            state="readonly",
            width=12,
        )
        backend_box.grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        size_row = ttk.Frame(frm)
        size_row.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(frm, text="采集参数:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Label(size_row, text="宽").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.width_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(size_row, text="高").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.height_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(size_row, text="FPS").pack(side=tk.LEFT)
        ttk.Entry(size_row, textvariable=self.fps_var, width=8).pack(side=tk.LEFT, padx=(4, 0))

        row += 1
        infer_row = ttk.Frame(frm)
        infer_row.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(frm, text="推理参数:").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Label(infer_row, text="imgsz").pack(side=tk.LEFT)
        ttk.Entry(infer_row, textvariable=self.imgsz_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(infer_row, text="conf").pack(side=tk.LEFT)
        ttk.Entry(infer_row, textvariable=self.conf_var, width=8).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(infer_row, text="iou").pack(side=tk.LEFT)
        ttk.Entry(infer_row, textvariable=self.iou_var, width=8).pack(side=tk.LEFT, padx=(4, 0))

        row += 1
        roi_row = ttk.Frame(frm)
        roi_row.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(frm, text="推理区域: ").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Label(roi_row, text="模式").pack(side=tk.LEFT)
        roi_mode_box = ttk.Combobox(
            roi_row,
            textvariable=self.roi_mode_var,
            values=["full", "center_ratio", "manual_rect"],
            state="readonly",
            width=14,
        )
        roi_mode_box.pack(side=tk.LEFT, padx=(4, 12))
        roi_mode_box.bind("<<ComboboxSelected>>", lambda _e: self._on_roi_mode_change())
        ttk.Label(roi_row, text="比例").pack(side=tk.LEFT)
        self.roi_ratio_entry = ttk.Entry(roi_row, textvariable=self.roi_ratio_var, width=8)
        self.roi_ratio_entry.pack(side=tk.LEFT, padx=(4, 0))

        row += 1
        rect_row = ttk.Frame(frm)
        rect_row.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(frm, text="手动坐标: ").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Label(rect_row, text="x").pack(side=tk.LEFT)
        e_x = ttk.Entry(rect_row, textvariable=self.roi_x_var, width=7)
        e_x.pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(rect_row, text="y").pack(side=tk.LEFT)
        e_y = ttk.Entry(rect_row, textvariable=self.roi_y_var, width=7)
        e_y.pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(rect_row, text="w").pack(side=tk.LEFT)
        e_w = ttk.Entry(rect_row, textvariable=self.roi_w_var, width=7)
        e_w.pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(rect_row, text="h").pack(side=tk.LEFT)
        e_h = ttk.Entry(rect_row, textvariable=self.roi_h_var, width=7)
        e_h.pack(side=tk.LEFT, padx=(4, 0))
        self.roi_rect_entries = [e_x, e_y, e_w, e_h]

        row += 1
        policy_row = ttk.Frame(frm)
        policy_row.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(frm, text="推理策略:").grid(row=row, column=0, sticky="w", pady=4)

        ttk.Label(policy_row, text="推理开关").pack(side=tk.LEFT)
        infer_box = ttk.Combobox(
            policy_row,
            textvariable=self.infer_var,
            values=["on", "off"],
            state="readonly",
            width=8,
        )
        infer_box.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(policy_row, text="设备").pack(side=tk.LEFT)
        device_box = ttk.Combobox(
            policy_row,
            textvariable=self.device_var,
            values=["auto", "cuda", "cpu"],
            state="readonly",
            width=10,
        )
        device_box.pack(side=tk.LEFT, padx=(4, 0))

        row += 1
        btn_row = ttk.Frame(frm)
        btn_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 6))
        self.start_btn = ttk.Button(btn_row, text="启动推理", command=self.start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btn_row, text="停止推理", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        self.check_btn = ttk.Button(btn_row, text="检查GPU组件", command=self.check_components)
        self.check_btn.pack(side=tk.LEFT, padx=8)

        row += 1
        ttk.Label(frm, text="运行日志:").grid(row=row, column=0, sticky="w", pady=(8, 4))

        row += 1
        self.log_text = tk.Text(frm, height=22, wrap=tk.WORD)
        self.log_text.grid(row=row, column=0, columnspan=3, sticky="nsew")
        scroll = ttk.Scrollbar(frm, command=self.log_text.yview)
        scroll.grid(row=row, column=3, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)
        self._on_roi_mode_change()

    def _on_roi_mode_change(self):
        mode = self.roi_mode_var.get().strip().lower()
        if self.roi_ratio_entry is None:
            return

        if mode == "center_ratio":
            self.roi_ratio_entry.configure(state="normal")
            for e in self.roi_rect_entries:
                e.configure(state="disabled")
        elif mode == "manual_rect":
            self.roi_ratio_entry.configure(state="disabled")
            for e in self.roi_rect_entries:
                e.configure(state="normal")
        else:
            self.roi_ratio_entry.configure(state="disabled")
            for e in self.roi_rect_entries:
                e.configure(state="disabled")

    def pick_model(self):
        mtype = self.model_type_var.get().strip().lower()
        if mtype == "pt":
            title = "选择 PT 模型"
            ftypes = [("PT model", "*.pt"), ("All files", "*.*")]
        else:
            title = "选择 ONNX 模型"
            ftypes = [("ONNX model", "*.onnx"), ("All files", "*.*")]

        p = filedialog.askopenfilename(
            title=title,
            filetypes=ftypes,
        )
        if p:
            self.model_var.set(p)

    def _on_model_type_change(self):
        mtype = self.model_type_var.get().strip().lower()
        cur = self.model_var.get().strip()
        if mtype == "pt":
            if cur.lower().endswith(".onnx"):
                self.model_var.set("yolov8n.pt")
        else:
            if cur.lower().endswith(".pt"):
                self.model_var.set(
                    r"D:\download\YOLO-Models-For-Valorant-main\YOLO-Models-For-Valorant-main\Yolov5\YOLOv5s\valorant-11.onnx"
                )

    def _append_log(self, msg: str):
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)

    @staticmethod
    def _run_cmd(command):
        try:
            p = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return p.returncode, p.stdout.strip()
        except Exception as e:
            return 1, repr(e)

    @staticmethod
    def _preload_torch_cuda_dll_for_ort() -> str:
        """复用 torch 自带 CUDA/cuDNN DLL，避免 ORT 误判 CUDA 不可用。"""
        try:
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.is_dir():
                os.add_dll_directory(str(torch_lib))
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
                return f"[SYS] CUDA DLL preload: {torch_lib}"
            return "[SYS] CUDA DLL preload: torch/lib not found"
        except Exception as e:
            return f"[SYS] CUDA DLL preload skipped: {e!r}"

    def _detect_and_log_system_info(self):
        cpu_name = platform.processor() or "Unknown CPU"
        cpu_count = os.cpu_count() if hasattr(os, "cpu_count") else None
        os_desc = f"{platform.system()} {platform.release()}"

        code, gpu_out = self._run_cmd(["cmd", "/c", "wmic path win32_VideoController get Name"])
        gpu_list = []
        if code == 0 and gpu_out:
            lines = [x.strip() for x in gpu_out.splitlines() if x.strip() and x.strip().lower() != "name"]
            gpu_list = lines

        code, nvsmi_out = self._run_cmd(["cmd", "/c", "where nvidia-smi"])
        has_nvidia_smi = code == 0 and bool(nvsmi_out)

        ort_available = []
        ort_session_providers = []
        onnx_cuda_ok = False
        onnx_cuda_err = ""
        preload_msg = self._preload_torch_cuda_dll_for_ort()
        try:
            import onnxruntime as ort

            ort_available = ort.get_available_providers()
            if self.runner_onnx.exists():
                # 仅做 provider 初始化检查，不做推理
                default_model = self.model_var.get().strip()
                if default_model.lower().endswith(".onnx") and Path(default_model).exists():
                    sess = ort.InferenceSession(default_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    ort_session_providers = sess.get_providers()
                    onnx_cuda_ok = "CUDAExecutionProvider" in ort_session_providers
        except Exception as e:
            onnx_cuda_err = repr(e)

        self.sys_info = {
            "os": os_desc,
            "cpu": cpu_name,
            "cpu_count": cpu_count,
            "gpus": gpu_list,
            "has_nvidia_smi": has_nvidia_smi,
            "ort_available": ort_available,
            "ort_session": ort_session_providers,
            "onnx_cuda_ok": onnx_cuda_ok,
            "onnx_cuda_err": onnx_cuda_err,
        }

        self._append_log("[SYS] ===== 系统信息 =====\n")
        self._append_log(f"[SYS] OS: {os_desc}\n")
        self._append_log(f"[SYS] CPU: {cpu_name}, cores={cpu_count}\n")
        self._append_log(f"[SYS] GPU: {gpu_list if gpu_list else 'Not Found'}\n")
        self._append_log(f"[SYS] nvidia-smi: {'YES' if has_nvidia_smi else 'NO'}\n")
        self._append_log(f"[SYS] ORT providers: {ort_available}\n")
        self._append_log(preload_msg + "\n")
        self._append_log(f"[SYS] Python selected: {self.python_exec}\n")
        if ort_session_providers:
            self._append_log(f"[SYS] ORT session providers: {ort_session_providers}\n")
        if onnx_cuda_err:
            self._append_log(f"[SYS] ORT cuda probe err: {onnx_cuda_err}\n")
        self._append_log("[SYS] ====================\n")

    def _check_selected_python_runtime(self):
        """检查当前选中的 Python 解释器是否具备 CUDA 能力。"""
        info = self._probe_python_cuda(self.python_exec)
        if not info.get("ok"):
            self._append_log(f"[RUNTIME] Python probe failed: {self.python_exec}\n")
            return {"ok": False, "cuda": False, "count": 0, "ver": ""}

        self._append_log(
            f"[RUNTIME] Python={info.get('exec')} torch={info.get('ver')} "
            f"cuda={info.get('cuda')} count={info.get('count')}\n"
        )
        return info

    def _check_cuda_components(self):
        mtype = self.model_type_var.get().strip().lower()
        device = self.device_var.get().strip().lower()

        has_gpu = bool(self.sys_info.get("gpus"))
        if not has_gpu:
            self._append_log("[CHECK] 未检测到独立显卡，当前将使用 CPU 推理。\n")
            return

        # 仅在 ONNX 路线且用户选择 auto/cuda 时检查 CUDA 组件
        if mtype != "onnx" or device == "cpu":
            self._append_log("[CHECK] 当前不是 ONNX+GPU 路线，无需检查 CUDA 组件。\n")
            return

        onnx_cuda_ok = bool(self.sys_info.get("onnx_cuda_ok"))
        ort_available = self.sys_info.get("ort_available") or []
        has_cuda_provider = "CUDAExecutionProvider" in ort_available
        has_nvidia_smi = bool(self.sys_info.get("has_nvidia_smi"))

        # 兜底：如果系统层面已具备 CUDA provider，且 nvidia-smi 可用，则不弹“缺组件”误报
        if onnx_cuda_ok or (has_cuda_provider and has_nvidia_smi):
            if not onnx_cuda_ok and has_cuda_provider and has_nvidia_smi:
                self._append_log(
                    "[POLICY] 检测到 CUDA provider 可用，跳过缺组件弹窗（避免误报）。\n"
                )
            self._append_log("[CHECK] GPU 组件检查通过，可优先使用 GPU 推理。\n")
            return

        self._append_log("[CHECK] 检测到显卡，但 ONNX CUDA 组件可能不完整或未生效。\n")
        self._append_log("[CHECK] 建议安装/检查：NVIDIA 驱动、CUDA 12.x、cuDNN 9.x、VC 运行库。\n")
        self._append_log("[CHECK] 参考文档: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements\n")

        go_install = messagebox.askyesno(
            "GPU 组件检查",
            "检测到可能缺少 GPU 组件。\n\n是否现在开始“下载并安装/修复”？\n"
            "- 是：自动执行可自动化安装，并打开官方页面\n"
            "- 否：取消，不做安装",
        )
        if go_install:
            self._install_missing_components()
        else:
            self._append_log("[CHECK] 用户取消安装。\n")

    def _install_missing_components(self):
        self._append_log("[INSTALL] 开始安装/修复可自动化组件...\n")

        commands = [
            [
                "winget",
                "install",
                "--id",
                "Microsoft.VCRedist.2015+.x64",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
                "--disable-interactivity",
            ],
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "onnxruntime-gpu==1.19.2",
            ],
        ]

        for cmd in commands:
            code, out = self._run_cmd(cmd)
            self._append_log(f"[INSTALL] cmd={' '.join(cmd)}\n")
            self._append_log(f"[INSTALL] exit={code}\n")
            if out:
                self._append_log(out[:1200] + ("\n...[truncated]\n" if len(out) > 1200 else "\n"))

        # CUDA/cuDNN 仍需用户在官网安装（需登录和选择版本）
        webbrowser.open("https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements")
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
        webbrowser.open("https://developer.nvidia.com/cudnn")

        self._append_log("[INSTALL] 已打开 CUDA/cuDNN 官方下载页，请按页面指引完成安装。\n")
        self._append_log("[INSTALL] 安装后请重启本程序，再点“检查GPU组件”复测。\n")

    def check_components(self):
        self._append_log("[CHECK] 手动触发组件检查...\n")
        self._detect_and_log_system_info()
        self._check_cuda_components()

    def _poll_logs(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self.root.after(100, self._poll_logs)

    def _reader_thread(self):
        assert self.proc is not None
        for line in self.proc.stdout:
            self.log_queue.put(line)
        code = self.proc.wait()
        self.log_queue.put(f"\n[PROCESS] exited with code {code}\n")
        self.root.after(0, self._on_process_exit)

    def _on_process_exit(self):
        self.proc = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _build_cmd(self):
        mtype = self.model_type_var.get().strip().lower()
        runner = self.runner_pt if mtype == "pt" else self.runner_onnx

        cmd = [
            self.python_exec,
            str(runner),
            "--model",
            self.model_var.get().strip(),
            "--source",
            self.source_var.get().strip(),
            "--backend",
            self.backend_var.get().strip(),
            "--width",
            self.width_var.get().strip(),
            "--height",
            self.height_var.get().strip(),
            "--fps",
            self.fps_var.get().strip(),
            "--imgsz",
            self.imgsz_var.get().strip(),
            "--conf",
            self.conf_var.get().strip(),
            "--device",
            self.device_var.get().strip(),
            "--infer",
            self.infer_var.get().strip(),
            "--roi-mode",
            self.roi_mode_var.get().strip(),
            "--roi-ratio",
            self.roi_ratio_var.get().strip(),
            "--roi-x",
            self.roi_x_var.get().strip(),
            "--roi-y",
            self.roi_y_var.get().strip(),
            "--roi-w",
            self.roi_w_var.get().strip(),
            "--roi-h",
            self.roi_h_var.get().strip(),
        ]

        if mtype == "onnx":
            cmd.extend(["--iou", self.iou_var.get().strip()])

        return cmd

    def start(self):
        if self.proc is not None:
            messagebox.showinfo("提示", "推理已在运行")
            return

        mtype = self.model_type_var.get().strip().lower()
        runner = self.runner_pt if mtype == "pt" else self.runner_onnx

        rt = self._check_selected_python_runtime()

        dev = self.device_var.get().strip().lower()
        if dev == "cuda" and not rt.get("cuda"):
            go_cpu = messagebox.askyesno(
                "CUDA 不可用",
                "当前解释器未检测到 CUDA（torch.cuda.is_available=False）。\n\n"
                "是否自动切换到 CPU 推理继续运行？\n"
                "- 是：切换 CPU 并继续\n"
                "- 否：取消启动",
            )
            if go_cpu:
                self.device_var.set("cpu")
                self._append_log("[POLICY] device=cuda 但运行时无 CUDA，已切换到 CPU。\n")
            else:
                self._append_log("[POLICY] 取消启动（CUDA 不可用且用户未同意切 CPU）。\n")
                return
        elif dev == "auto" and not rt.get("cuda"):
            self._append_log("[POLICY] device=auto 且 CUDA 不可用，将自动走 CPU。\n")

        # 没检测到显卡时，自动回退到 CPU
        if not self.sys_info.get("gpus") and self.device_var.get().strip().lower() != "cpu":
            self.device_var.set("cpu")
            self._append_log("[POLICY] 未检测到显卡，已自动切换到 CPU 推理。\n")

        if not runner.exists():
            messagebox.showerror("错误", f"未找到脚本: {runner}")
            return

        model_path = Path(self.model_var.get().strip())
        if not model_path.exists():
            messagebox.showerror("错误", f"模型不存在: {model_path}")
            return

        cmd = self._build_cmd()
        self._append_log(f"\n[PROCESS] start: {' '.join(cmd)}\n")

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            self.proc = None
            messagebox.showerror("启动失败", repr(e))
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        t = threading.Thread(target=self._reader_thread, daemon=True)
        t.start()

    def stop(self):
        if self.proc is None:
            return
        self._append_log("\n[PROCESS] stopping...\n")
        self.proc.terminate()

    def on_close(self):
        if self.proc is not None:
            self.proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

