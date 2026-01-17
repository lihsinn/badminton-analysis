import cv2
import numpy as np
from collections import deque


class BallTracker:
    """羽球軌跡追踪器"""

    def __init__(self, buffer_size=32):
        """
        初始化球軌跡追踪器

        Args:
            buffer_size: 軌跡緩衝區大小
        """
        self.pts = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add_point(self, center):
        """
        添加新的軌跡點

        Args:
            center: 球的中心點 (x, y)
        """
        self.pts.appendleft(center)

    def clear(self):
        """清空軌跡"""
        self.pts.clear()

    def draw_trajectory(self, frame, color=(0, 0, 255), thickness_multiplier=2.5):
        """
        在影像上繪製球的軌跡

        Args:
            frame: 輸入影像
            color: 軌跡顏色 (B, G, R)
            thickness_multiplier: 線條粗細倍數

        Returns:
            繪製後的影像
        """
        output = frame.copy()

        for i in range(1, len(self.pts)):
            thickness = int(np.sqrt(self.buffer_size / float(i + 1)) * thickness_multiplier)
            cv2.line(output, self.pts[i - 1], self.pts[i], color, thickness)

        return output

    def draw_current_position(self, frame, center, color=(0, 0, 255), radius=5):
        """
        繪製球的當前位置

        Args:
            frame: 輸入影像
            center: 球的中心點 (x, y)
            color: 顏色 (B, G, R)
            radius: 圓點半徑

        Returns:
            繪製後的影像
        """
        output = frame.copy()
        cv2.circle(output, center, radius, color, -1)
        return output

    def draw_ball_tracking(self, frame, shuttle_positions, draw_trajectory=True,
                          ball_color=(0, 0, 255), trajectory_color=(0, 0, 255)):
        """
        完整的球追踪繪製（位置 + 軌跡）

        Args:
            frame: 輸入影像
            shuttle_positions: 羽球位置列表
            draw_trajectory: 是否繪製軌跡
            ball_color: 球的顏色
            trajectory_color: 軌跡顏色

        Returns:
            繪製後的影像
        """
        output = frame.copy()

        for center in shuttle_positions:
            # 繪製當前位置
            cv2.circle(output, center, 5, ball_color, -1)

            # 添加到軌跡
            self.add_point(center)

        # 繪製軌跡
        if draw_trajectory:
            output = self.draw_trajectory(output, color=trajectory_color)

        return output

    def get_trajectory_points(self):
        """
        取得軌跡點列表

        Returns:
            list: 軌跡點列表
        """
        return list(self.pts)

    def get_latest_position(self):
        """
        取得最新的球位置

        Returns:
            tuple: (x, y) 或 None
        """
        if len(self.pts) > 0:
            return self.pts[0]
        return None
