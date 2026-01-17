"""
羽球分析系統配置檔案

包含所有模組的配置參數，便於統一管理和調整
"""
import os
import numpy as np


class Config:
    """全域配置類"""

    # ========== 路徑配置 ==========
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolov5', 'weights', 'best.pt')
    YOLO_SOURCE_PATH = os.path.join(PROJECT_ROOT, 'yolov5')
    TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'data', 'templates', 'BWF.jpg')
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # ========== 顯示配置 ==========
    DISPLAY_WIDTH = 1024
    DISPLAY_HEIGHT = 576

    # ========== 場地檢測配置 ==========
    class Court:
        """場地檢測配置"""
        # HSV 顏色範圍（綠色場地）
        COLOR_LOWER = np.array([0, 100, 0])
        COLOR_UPPER = np.array([160, 255, 154])

        # 場地面積閾值（佔畫面百分比）
        AREA_MIN_PERCENT = 18
        AREA_MAX_PERCENT = 50

        # 中值模糊核大小
        MEDIAN_BLUR_KSIZE_1 = 7
        MEDIAN_BLUR_KSIZE_2 = 17

        # 形態學操作核大小
        MORPH_KERNEL_SIZE = (9, 9)
        MORPH_ITERATIONS = 2

        # 閾值參數
        THRESHOLD_VALUE = 75

    # ========== YOLO 檢測配置 ==========
    class YOLO:
        """YOLO 物體檢測配置"""
        CONFIDENCE_THRESHOLD = 0.2
        DETECTION_SIZE = 640

        # 類別名稱（根據您的訓練模型）
        # 通常包括: 'player', 'shuttle'
        EXPECTED_CLASSES = ['player', 'shuttle']

    # ========== OCR 配置 ==========
    class OCR:
        """比分識別配置"""
        # 比分板區域（針對 1920x1080 解析度）
        SCORE_REGION = {
            'y1': 50,
            'y2': 145,
            'x1': 280,
            'x2': 700
        }

        # 球員名稱區域（相對於裁切區域）
        PLAYER_NAME_REGION = {
            'y1': 3,
            'y2': 62,
            'x1': 123,
            'x2': 250
        }

        # 比分區域（相對於裁切區域）
        SCORE_NUMBER_REGION = {
            'y1': 3,
            'y2': 62,
            'x1': 318,
            'x2': 445
        }

        # OCR 配置
        TESSERACT_CONFIG = "--psm 6"
        TESSERACT_LANG = "eng"

        # 圖像處理參數
        THRESHOLD_VALUE = 30
        SCORE_THRESHOLD_VALUE = 215
        RESIZE_SCALE = 3.0

        # 模板匹配
        TEMPLATE_SIZE = (195, 27)

    # ========== 骨架追蹤配置 ==========
    class Pose:
        """姿態估計配置"""
        # MediaPipe Pose 參數
        MIN_DETECTION_CONFIDENCE = 0.5
        MIN_TRACKING_CONFIDENCE = 0.5

        # 骨架繪製參數
        SKELETON_COLOR = (0, 255, 0)
        SKELETON_THICKNESS = 2
        JOINT_RADIUS = 5

    # ========== 球軌跡追蹤配置 ==========
    class Tracking:
        """球軌跡追蹤配置"""
        BUFFER_SIZE = 32

        # 軌跡繪製顏色
        TRAJECTORY_COLOR_MAIN = (0, 0, 255)      # 主畫面 (紅色)
        TRAJECTORY_COLOR_SMALL = (255, 0, 255)   # 小畫面 (紫色)

        # 球繪製參數
        BALL_RADIUS = 5
        BALL_COLOR_MAIN = (0, 0, 255)
        BALL_COLOR_SMALL = (255, 0, 255)

        # 軌跡線粗細
        TRAJECTORY_THICKNESS = 2

    # ========== 性能配置 ==========
    class Performance:
        """性能優化配置"""
        # 是否啟用 GPU
        USE_GPU = True

        # 幀處理跳幀數（每 N 幀處理一次）
        FRAME_SKIP = 1

        # 是否啟用懶載入
        LAZY_LOADING = False

        # 快取大小
        CACHE_SIZE = 100

        # 場地檢測間隔（每 N 幀檢查一次，建議 30）
        COURT_CHECK_INTERVAL = 30

        # OCR 檢測間隔（每 N 幀讀取一次，建議 90）
        OCR_INTERVAL = 90

        # UI 更新間隔（每 N 次循環更新一次 UI 文字）
        UI_UPDATE_INTERVAL = 3

        # 是否啟用性能分析
        ENABLE_PROFILING = False

    # ========== UI 配置 ==========
    class UI:
        """使用者介面配置"""
        # 視窗標題
        WINDOW_TITLE = 'Badminton Analysis System'

        # 字型
        FONT_FAMILY = 'Arial'
        FONT_SIZE = 11
        FONT_SIZE_LARGE = 16

        # 顏色主題
        COLOR_PRIMARY = '#389fff'
        COLOR_BACKGROUND = '#ffffff'
        COLOR_TEXT = '#2C3539'
        COLOR_CANVAS_BG = '#cfcfcf'
        COLOR_INFO_BG = '#e0e0e0'

        # 畫布尺寸
        MAIN_CANVAS_SIZE = (800, 550)
        BALL_CANVAS_SIZE = (436, 300)

        # 滑桿配置
        SLIDER_RANGE = (0, 1000)
        SLIDER_RESOLUTION = 30

    # ========== 影片處理配置 ==========
    class Video:
        """影片處理配置"""
        # 預設 FPS
        DEFAULT_FPS = 30

        # 影片編解碼器
        FOURCC = 'mp4v'

        # 是否儲存處理後的影片
        SAVE_OUTPUT = False
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'output')

# 為了向後相容，提供字典格式的配置
CONFIG = {
    'court': {
        'color_lower': Config.Court.COLOR_LOWER.tolist(),
        'color_upper': Config.Court.COLOR_UPPER.tolist(),
        'area_threshold': (Config.Court.AREA_MIN_PERCENT, Config.Court.AREA_MAX_PERCENT),
        'median_blur_ksize': (Config.Court.MEDIAN_BLUR_KSIZE_1, Config.Court.MEDIAN_BLUR_KSIZE_2),
        'morph_kernel_size': Config.Court.MORPH_KERNEL_SIZE,
        'threshold_value': Config.Court.THRESHOLD_VALUE
    },
    'yolo': {
        'confidence': Config.YOLO.CONFIDENCE_THRESHOLD,
        'model_path': Config.YOLO_MODEL_PATH,
        'source_path': Config.YOLO_SOURCE_PATH,
        'detection_size': Config.YOLO.DETECTION_SIZE
    },
    'ocr': {
        'tesseract_path': Config.TESSERACT_PATH,
        'template_path': Config.TEMPLATE_PATH,
        'score_region': Config.OCR.SCORE_REGION,
        'tesseract_config': Config.OCR.TESSERACT_CONFIG,
        'tesseract_lang': Config.OCR.TESSERACT_LANG
    },
    'pose': {
        'min_detection_confidence': Config.Pose.MIN_DETECTION_CONFIDENCE,
        'min_tracking_confidence': Config.Pose.MIN_TRACKING_CONFIDENCE
    },
    'tracking': {
        'buffer_size': Config.Tracking.BUFFER_SIZE,
        'trajectory_color_main': Config.Tracking.TRAJECTORY_COLOR_MAIN,
        'trajectory_color_small': Config.Tracking.TRAJECTORY_COLOR_SMALL
    },
    'display': {
        'width': Config.DISPLAY_WIDTH,
        'height': Config.DISPLAY_HEIGHT
    },
    'performance': {
        'use_gpu': Config.Performance.USE_GPU,
        'frame_skip': Config.Performance.FRAME_SKIP,
        'lazy_loading': Config.Performance.LAZY_LOADING,
        'cache_size': Config.Performance.CACHE_SIZE,
        'court_check_interval': Config.Performance.COURT_CHECK_INTERVAL,
        'ocr_interval': Config.Performance.OCR_INTERVAL,
        'ui_update_interval': Config.Performance.UI_UPDATE_INTERVAL,
        'enable_profiling': Config.Performance.ENABLE_PROFILING
    }
}


# 匯出配置
__all__ = ['Config', 'CONFIG']
