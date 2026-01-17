import torch
import cv2
import os
import sys
from yolov5.utils.general import non_max_suppression, scale_coords

class YOLODetector:
    """YOLO 物體檢測器（羽球、球員）- 使用 torch.load 直接載入"""

    def __init__(self, config=None):
        """
        初始化 YOLO 檢測器

        Args:
            config: 配置字典，包含以下參數：
                - model_path: 模型權重路徑
                - source_path: YOLOv5 原始碼路徑
                - confidence: 信心門檻值 (default: 0.2)
                - detection_size: 檢測尺寸 (default: 640)
                - use_gpu: 是否使用 GPU (default: True)
        """
        if config is None:
            config = {}

        # 取得配置參數
        model_path = config.get('model_path', './models/yolov5/weights/best.pt')
        source_path = config.get('source_path', './yolov5')
        self.conf_threshold = config.get('confidence', 0.2)
        self.detection_size = config.get('detection_size', 640)
        use_gpu = config.get('use_gpu', True)

        # 檢查模型檔案是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"YOLOv5 source path not found: {source_path}")

        # 載入模型
        try:
            print(f"Loading model...")
            print(f"Model path: {model_path}")

            # 禁用警告訊息
            import warnings
            warnings.filterwarnings('ignore')

            # 將 YOLOv5 目錄加入到 Python 路徑（必須在 torch.load 之前）
            abs_source_path = os.path.abspath(source_path)
            if abs_source_path not in sys.path:
                sys.path.insert(0, abs_source_path)
                print(f"Added {abs_source_path} to Python path")

            # 確定設備
            self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
            print(f"Using device: {self.device}")

            # 直接使用 torch.load 載入模型
            # 注意: weights_only=False 是因為我們信任這個模型檔案
            print("Loading model weights...")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # 從 checkpoint 中提取模型
            if isinstance(checkpoint, dict):
                # 如果是字典格式，提取 'model' 或 'ema'
                if 'ema' in checkpoint and checkpoint['ema'] is not None:
                    self.model = checkpoint['ema'].float()
                elif 'model' in checkpoint:
                    self.model = checkpoint['model'].float()
                else:
                    raise RuntimeError("Cannot find model in checkpoint")
            else:
                # 直接就是模型
                self.model = checkpoint.float()

            # Fuse 並設置為評估模式
            if hasattr(self.model, 'fuse'):
                self.model = self.model.fuse()
            self.model.eval()

            # 移動到正確的設備
            self.model = self.model.to(self.device)

            # 重置 Detect 層的 grid 緩存（解決不同輸入尺寸的問題）
            for m in self.model.modules():
                t = type(m).__name__
                if t == 'Detect':
                    m.inplace = True
                    if hasattr(m, 'anchor_grid') and not isinstance(m.anchor_grid, list):
                        delattr(m, 'anchor_grid')
                        setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
                    if hasattr(m, 'grid') and not isinstance(m.grid, list):
                        delattr(m, 'grid')
                        setattr(m, 'grid', [torch.zeros(1)] * m.nl)

            print("Model loaded! Configuring parameters...")

            # 設定信心門檻
            if hasattr(self.model, 'conf'):
                self.model.conf = self.conf_threshold

            # 取得類別名稱
            if hasattr(self.model, 'names'):
                self.classes = self.model.names
            elif hasattr(self.model, 'module') and hasattr(self.model.module, 'names'):
                self.classes = self.model.module.names
            else:
                # 預設類別
                self.classes = {0: 'player', 1: 'shuttle'}
                print("Warning: Cannot read class names from model, using defaults")

            # 檢查圖片尺寸（使用 stride）
            if hasattr(self.model, 'stride'):
                stride = int(self.model.stride.max() if hasattr(self.model.stride, 'max') else self.model.stride)
                # 確保 detection_size 是 stride 的倍數
                self.detection_size = ((self.detection_size // stride) * stride)
            else:
                print("Warning: Cannot get model.stride, using default detection size")

            print(f"YOLO model loaded successfully!")
            print(f"Detection classes: {self.classes}")
            print(f"Detection size: {self.detection_size}")

        except Exception as e:
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def detect(self, frame, size=None):
        """
        檢測物體（優化版 - GPU 加速）

        Args:
            frame: 輸入影像（BGR 格式）
            size: 檢測尺寸（若不指定則使用配置值）

        Returns:
            tuple: (labels, bbox, result)
        """
        if size is None:
            size = self.detection_size

        h0, w0 = frame.shape[:2]

        # ---------- Preprocess ----------
        img = frame[:, :, ::-1] # BGR -> RGB
        # resize
        img = cv2.resize(img, (size, size))
        # HWC -> CHW
        img = img.transpose(2, 0, 1) # HWC -> CHW
        # numpy -> tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)  # (1, 3, H, W)

        # ---------- Inference ----------
        # YOLO 需要 RGB 格式
        # 使用 torch.no_grad() 減少記憶體使用
        with torch.no_grad():
            #result = self.model(frame[:, :, ::-1], size=size)
            preds = self.model(img)[0]   # ← 沒有 size 參數

        # ---------- NMS ----------
        preds = non_max_suppression(
            preds,
            conf_thres=self.conf_threshold,
            iou_thres=0.45
        )

        if preds[0] is None:
            return [], [], preds

        det = preds[0]  # (N, 6) -> xyxy, conf, cls

        # ---------- Convert to xyxyn ----------
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], frame.shape
        )

        labels = det[:, 5]
        bbox = det[:, :4]

        # normalize
        bbox[:, [0, 2]] /= w0
        bbox[:, [1, 3]] /= h0

        return labels, bbox, preds


    def draw_detections(self, frame, labels, bbox):
        """
        在影像上繪製檢測結果

        Args:
            frame: 輸入影像
            labels: 標籤列表
            bbox: 邊界框列表

        Returns:
            繪製後的影像
        """
        output = frame.copy()
        height, width, _ = output.shape

        if len(labels) > 0:
            for i in range(len(labels)):
                x1, y1 = bbox[i][0].item() * width, bbox[i][1].item() * height
                x2, y2 = bbox[i][2].item() * width, bbox[i][3].item() * height
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = self.classes[int(labels[i])]

                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(output, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return output

    def get_player_bbox(self, labels, bbox, frame_shape):
        """
        取得第一個檢測到的球員邊界框

        Args:
            labels: 標籤列表
            bbox: 邊界框列表
            frame_shape: 影像尺寸 (height, width, channels)

        Returns:
            tuple: (x1, y1, x2, y2) 或 None
        """
        if len(labels) > 0:
            height, width = frame_shape[0], frame_shape[1]
            x1, y1 = bbox[0][0].item() * width, bbox[0][1].item() * height
            x2, y2 = bbox[0][2].item() * width, bbox[0][3].item() * height
            return int(x1), int(y1), int(x2), int(y2)
        return None

    def get_shuttle_positions(self, labels, bbox, frame_shape):
        """
        取得所有羽球的位置

        Args:
            labels: 標籤列表
            bbox: 邊界框列表
            frame_shape: 影像尺寸 (height, width, channels)

        Returns:
            list: 羽球中心點列表 [(x, y), ...]
        """
        shuttle_positions = []
        height, width = frame_shape[0], frame_shape[1]

        for i in range(len(labels)):
            if self.classes[int(labels[i])] == 'shuttle':
                x1, y1 = bbox[i][0].item() * width, bbox[i][1].item() * height
                x2, y2 = bbox[i][2].item() * width, bbox[i][3].item() * height
                center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                shuttle_positions.append(center)

        return shuttle_positions
