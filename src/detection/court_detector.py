import cv2
import numpy as np
from PIL import Image


class CourtDetector:
    """羽球場地檢測器"""

    def __init__(self, config=None):
        """
        初始化場地檢測器

        Args:
            config: 配置字典，包含以下參數：
                - color_lower: HSV 顏色下界 (default: [0, 100, 0])
                - color_upper: HSV 顏色上界 (default: [160, 255, 154])
                - area_threshold: 場地面積閾值範圍 (min%, max%)
                - median_blur_ksize: 中值模糊核大小
                - morph_kernel_size: 形態學操作核大小
                - threshold_value: 二值化閾值
        """
        if config is None:
            config = {}

        # 綠色場地檢測範圍 (HSV)
        self.lower_color_bounds = np.array(config.get('color_lower', [0, 100, 0]))
        self.upper_color_bounds = np.array(config.get('color_upper', [160, 255, 154]))

        # 面積閾值
        area_threshold = config.get('area_threshold', (18, 36))
        self.area_min_percent = area_threshold[0]
        self.area_max_percent = area_threshold[1]

        # 圖像處理參數
        self.median_blur_ksize = config.get('median_blur_ksize', (7, 17))
        morph_kernel = config.get('morph_kernel_size', (9, 9))
        self.morph_kernel = np.ones(morph_kernel, np.uint8)
        self.threshold_value = config.get('threshold_value', 75)

    def detect(self, frame):
        """
        檢測羽球場地

        Args:
            frame: 輸入影像

        Returns:
            tuple: (is_court_ok, court_corners, output_frame, area_percent)
        """
        ww, hh = frame.shape[1], frame.shape[0]
        output = frame.copy()

        # 顏色遮罩 - 提取綠色區域
        mask = cv2.inRange(frame, self.lower_color_bounds, self.upper_color_bounds)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cvimg_green = frame & mask_rgb

        # 影像處理
        img_medianblur = cv2.medianBlur(cvimg_green, ksize=self.median_blur_ksize[0])
        img_medianblur2 = cv2.medianBlur(img_medianblur, ksize=self.median_blur_ksize[1])
        img_gray = cv2.cvtColor(img_medianblur2, cv2.COLOR_BGR2GRAY)
        closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)
        ret, thresh = cv2.threshold(closing, self.threshold_value, 255, 1)
        thresh_white = 255 - thresh
        contours, hierarchy = cv2.findContours(thresh_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 排序輪廓
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(sorted_contours) > 0:
            contours = sorted_contours[0]
            area = cv2.contourArea(contours)
        else:
            contours = sorted_contours
            area = 0

        # 尋找四邊形
        approx4 = []
        if len(contours) != 0:
            for eps in np.linspace(0.001, 0.99, 1000):
                peri = cv2.arcLength(contours, True)
                approx = cv2.approxPolyDP(contours, eps * peri, True)

                if len(approx) == 4:
                    approx4 = approx
                    break

        # 計算面積百分比
        if len(approx4) == 4:
            area = cv2.contourArea(approx4)
            area_percent = (area / (ww * hh)) * 100
            n = list(approx4.ravel())
        else:
            n = [0, 0, 0, 0, 0, 0, 0, 0]
            area_percent = 0

        # 給四個座標排序
        n_x_axis_list = [n[0], n[2], n[4], n[6]]
        n_x_axis_list_sort = sorted(n_x_axis_list)

        nn = []
        for i in range(0, 4):
            index = n.index(n_x_axis_list_sort[i])
            nn.append(n[index])
            nn.append(n[index + 1])

        # 顯示檢測面積
        cv2.rectangle(output, (490, 30), (640, 65), (255, 255, 255), -1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, "detect area:" + str(int(area_percent)) + "%",
                    (500, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 判斷場地是否檢測正確
        is_court_ok = (int(area_percent) > self.area_min_percent and
                       int(area_percent) < self.area_max_percent and
                       nn[6] > nn[4] and nn[7] > nn[5] and
                       nn[2] > nn[0] and nn[3] < nn[1])

        return is_court_ok, approx4 if len(approx4) == 4 else None, output, area_percent

    def draw_court_status(self, frame, is_court_ok):
        """
        在畫面上繪製場地檢測狀態

        Args:
            frame: 輸入影像
            is_court_ok: 場地是否檢測正確

        Returns:
            繪製後的影像
        """
        output = frame.copy()
        output2 = output.copy()
        start_point = (30, 70)
        end_point = (150, 20)

        if is_court_ok:
            color = (54, 231, 131)
            text = 'OK'
        else:
            color = (0, 0, 246)
            text = 'NG'

        output2 = cv2.rectangle(output2, start_point, end_point, color, -1, cv2.LINE_AA)
        alpha = 0.3
        beta = 1 - alpha
        gamma = 0
        output = cv2.addWeighted(output, alpha, output2, beta, gamma)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output, text, (55, 65), font, 1.8, (255, 255, 255), 3, cv2.LINE_AA)

        return output
