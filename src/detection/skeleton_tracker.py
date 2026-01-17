import cv2
import mediapipe as mp


class SkeletonTracker:
    """使用 MediaPipe 的骨架追踪器（優化版）"""

    def __init__(self, config=None):
        """
        初始化骨架追踪器

        Args:
            config: 配置字典，包含以下參數：
                - model_complexity: 模型複雜度 (0, 1, 2，default: 0)
                - min_detection_confidence: 最小檢測信心 (default: 0.5)
                - min_tracking_confidence: 最小追踪信心 (default: 0.5)
        """
        if config is None:
            config = {}

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.model_complexity = config.get('model_complexity', 0)
        self.min_detection_confidence = config.get('min_detection_confidence', 0.5)
        self.min_tracking_confidence = config.get('min_tracking_confidence', 0.5)

        # 重用 Pose 對象以提升性能
        self.pose = self.mp_pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def get_position(self, img, results):
        """
        取得骨架關鍵點位置

        Args:
            img: 輸入影像
            results: MediaPipe 處理結果

        Returns:
            list: 關鍵點位置列表 [(x, y), ...]
        """
        lm_list = []
        if results.pose_landmarks:
            h, w, c = img.shape
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))
        return lm_list

    def process_player(self, player_image, draw_on_original=False):
        """
        處理球員影像並追踪骨架（優化版 - 重用 Pose 對象）

        Args:
            player_image: 球員影像
            draw_on_original: 是否在原始影像上繪製

        Returns:
            tuple: (position_list, results)
        """
        # 轉換為 RGB
        player_image.flags.writeable = False
        image_rgb = cv2.cvtColor(player_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # 取得關鍵點位置
        pos_list = self.get_position(player_image, results)

        # 轉回 BGR
        player_image.flags.writeable = True

        return pos_list, results

    def draw_skeleton(self, frame, results, bbox=None):
        """
        在影像上繪製骨架

        Args:
            frame: 輸入影像
            results: MediaPipe 處理結果
            bbox: 可選的邊界框 (x1, y1, x2, y2)，如果提供則只在該區域繪製

        Returns:
            繪製後的影像
        """
        output = frame.copy()

        if results.pose_landmarks:
            if bbox:
                x1, y1, x2, y2 = bbox
                self.mp_drawing.draw_landmarks(
                    output[y1:y2, x1:x2],
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            else:
                self.mp_drawing.draw_landmarks(
                    output,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

        return output

    def __del__(self):
        """析構函數 - 釋放 Pose 資源"""
        if hasattr(self, 'pose'):
            self.pose.close()
