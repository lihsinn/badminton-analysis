import cv2
import time
import numpy as np
from PIL import Image
from collections import deque

from ..detection.court_detector import CourtDetector
from ..detection.yolo_detector import YOLODetector
from ..detection.skeleton_tracker import SkeletonTracker
from ..detection.ball_tracker import BallTracker
from ..ocr.score_reader import ScoreReader
from ..utils.video_utils import VideoProcessor


class BadmintonVideoPlayer:
    """羽球影片播放器，整合所有檢測功能"""

    def __init__(self, video_source, config=None):
        """
        初始化影片播放器

        Args:
            video_source: 影片檔案路徑
            config: 配置字典，包含各模組的配置參數
                如果為 None，將使用預設配置
        """
        # 如果沒有提供配置，嘗試從配置檔案匯入
        if config is None:
            try:
                from config import CONFIG
                config = CONFIG
            except ImportError:
                print("Warning: Cannot import config file, using default settings")
                config = {}

        # 影片處理器
        self.video_processor = VideoProcessor(video_source)
        self.width = self.video_processor.width
        self.height = self.video_processor.height
        self.frames = self.video_processor.frames
        self.fps = self.video_processor.fps

        # 顯示尺寸配置
        display_config = config.get('display', {})
        self.display_width = display_config.get('width', 1024)
        self.display_height = display_config.get('height', 576)

        # 性能配置
        perf_config = config.get('performance', {})
        self.frame_skip = perf_config.get('frame_skip', 1)
        self.use_cache = perf_config.get('cache_size', 0) > 0
        self.cache_size = perf_config.get('cache_size', 100)

        # 檢測器初始化（使用配置參數）
        print("Initializing court detector...")
        self.court_detector = CourtDetector(config.get('court', {}))

        print("Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(config.get('yolo', {}))

        print("Initializing skeleton tracker...")
        self.skeleton_tracker = SkeletonTracker(config.get('pose', {}))

        print("Initializing ball tracker...")
        tracking_config = config.get('tracking', {})
        buffer_size = tracking_config.get('buffer_size', 32)
        self.ball_tracker = BallTracker(buffer_size=buffer_size)

        print("Initializing score reader...")
        self.score_reader = ScoreReader(config.get('ocr', {}))

        # 軌跡顏色配置
        self.trajectory_color_main = tracking_config.get('trajectory_color_main', (0, 0, 255))
        self.trajectory_color_small = tracking_config.get('trajectory_color_small', (255, 0, 255))

        # 狀態變數
        self.scores = "0\n0"
        self.player_names = "Player1\nPlayer2"
        self.speed_info = "sec/frame"

        # 性能優化變數
        self.frame_count = 0
        self.last_court_status = False
        self.court_check_interval = perf_config.get('court_check_interval', 30)
        self.ocr_interval = perf_config.get('ocr_interval', 30)

        # 快取變數
        self.cached_court_status = None
        self.cached_court_corners = None

        print("Video player initialized")

    def process_frame(self):
        """
        處理單幀影像，執行所有檢測（優化版）

        Process single frame through complete pipeline with performance optimizations:
        - Conditional detection (court check interval)
        - Reduced OCR frequency
        - Cached results

        Pipeline:
        1. Court detection (cached)
        2. OCR for score (interval-based)
        3. YOLO for players and shuttle
        4. Pose estimation
        5. Ball tracking

        Returns:
            tuple: (ret, main_frame, ball_trajectory_frame, scores, player_names, speed_info)
        """
        ret, frame = self.video_processor.read_frame()
        if not ret:
            return False, None, None, self.scores, self.player_names, self.speed_info

        # 計時開始
        t0 = time.time()
        self.frame_count += 1

        # 保存原始影像
        orgimg = frame.copy()

        # 調整大小（使用配置的顯示尺寸）
        ww, hh = self.display_width, self.display_height
        frame = cv2.resize(frame, (ww, hh))

        # 創建球軌跡畫面
        smallframe = np.zeros((hh, ww, 3), dtype="uint8")
        smallframe.fill(0)

        # 1. 場地檢測（條件檢測 - 每N幀檢查一次）
        court_t0 = time.time()
        if self.frame_count % self.court_check_interval == 0:
            is_court_ok, court_corners, frame, area_percent = self.court_detector.detect(frame)
            self.cached_court_status = is_court_ok
            self.cached_court_corners = court_corners
        else:
            # 使用快取的場地狀態
            is_court_ok = self.cached_court_status if self.cached_court_status is not None else True
            court_corners = self.cached_court_corners

        frame = self.court_detector.draw_court_status(frame, is_court_ok)
        court_time = time.time() - court_t0

        # 如果場地檢測成功，執行其他檢測
        if is_court_ok:
            # 在小畫面上繪製場地輪廓
            if court_corners is not None:
                cv2.drawContours(smallframe, [court_corners], -1, (41, 227, 20), 2)

            # 2. OCR 識別比分（降低頻率 - 每N幀讀取一次）
            ocr_t0 = time.time()
            if self.frame_count % self.ocr_interval == 0:
                self.player_names, self.scores = self.score_reader.read_score_and_names(orgimg)
            ocr_time = time.time() - ocr_t0

            # 3. YOLO 物體檢測
            yolo_t0 = time.time()
            labels, bbox, result = self.yolo_detector.detect(frame)

            # 繪製檢測框（跳過第一個，因為是球員）
            if len(labels) > 1:
                height, width, _ = frame.shape
                for i in range(1, len(labels)):
                    x1, y1 = bbox[i][0].item() * width, bbox[i][1].item() * height
                    x2, y2 = bbox[i][2].item() * width, bbox[i][3].item() * height
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    label = self.yolo_detector.classes[int(labels[i])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            yolo_time = time.time() - yolo_t0

            # 4. 骨架追踪
            skeleton_t0 = time.time()
            if len(labels) > 0:
                player_bbox = self.yolo_detector.get_player_bbox(labels, bbox, frame.shape)
                if player_bbox:
                    x1, y1, x2, y2 = player_bbox
                    player_image = frame[y1:y2, x1:x2]
                    pos_list, results = self.skeleton_tracker.process_player(player_image)

                    # 在主畫面和小畫面上繪製骨架
                    frame = self.skeleton_tracker.draw_skeleton(frame, results, bbox=player_bbox)
                    smallframe = self.skeleton_tracker.draw_skeleton(smallframe, results, bbox=player_bbox)
            skeleton_time = time.time() - skeleton_t0

            # 5. 球軌跡追踪
            tracking_t0 = time.time()
            shuttle_positions = self.yolo_detector.get_shuttle_positions(labels, bbox, frame.shape)

            for center in shuttle_positions:
                # 在主畫面和小畫面上繪製球（使用配置的顏色）
                cv2.circle(frame, center, 5, self.trajectory_color_main, -1)
                cv2.circle(smallframe, center, 5, self.trajectory_color_small, -1)
                self.ball_tracker.add_point(center)

            # 繪製軌跡（使用配置的顏色）
            frame = self.ball_tracker.draw_trajectory(frame, color=self.trajectory_color_main)
            smallframe = self.ball_tracker.draw_trajectory(smallframe, color=self.trajectory_color_small)
            tracking_time = time.time() - tracking_t0

            # 計算總時間
            total_time = time.time() - t0

            # 更新速度資訊
            self.speed_info = (
                f"AllTime {round(total_time, 2)}\n"
                f"courtTime {round(court_time, 2)}\n"
                f"OCRTime {round(ocr_time, 2)}\n"
                f"yoloTime {round(yolo_time, 2)}\n"
                f"skeletonTime {round(skeleton_time, 2)}\n"
                f"trackingBalTime {round(tracking_time, 2)}"
            )
        else:
            # 場地檢測失敗
            total_time = time.time() - t0
            self.speed_info = f"AllTime {round(total_time, 2)}"

        # 轉換為 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return True, frame_rgb, smallframe, self.scores, self.player_names, self.speed_info

    def goto_frame(self, frame_no):
        """
        跳轉到指定幀

        Args:
            frame_no: 幀號

        Returns:
            tuple: (ret, frame)
        """
        return self.video_processor.goto_frame(frame_no)

    def release(self):
        """釋放資源"""
        self.video_processor.release()

    def __del__(self):
        """析構函數"""
        self.release()
