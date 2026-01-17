import cv2
import numpy as np


def create_blank_image(width, height, color=(0, 0, 0)):
    """
    創建空白影像

    Args:
        width: 寬度
        height: 高度
        color: 背景顏色 (B, G, R)

    Returns:
        空白影像
    """
    image = np.zeros((height, width, 3), dtype="uint8")
    image[:] = color
    return image


def overlay_transparent(background, overlay, alpha=0.3):
    """
    將兩個影像混合

    Args:
        background: 背景影像
        overlay: 覆蓋影像
        alpha: 透明度 (0-1)

    Returns:
        混合後的影像
    """
    beta = 1 - alpha
    gamma = 0
    return cv2.addWeighted(background, alpha, overlay, beta, gamma)


def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.5, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), thickness=1, padding=5):
    """
    在影像上繪製帶背景的文字

    Args:
        frame: 輸入影像
        text: 文字內容
        position: 文字位置 (x, y)
        font: 字體
        font_scale: 字體大小
        text_color: 文字顏色
        bg_color: 背景顏色
        thickness: 文字粗細
        padding: 背景內距

    Returns:
        繪製後的影像
    """
    output = frame.copy()
    x, y = position

    # 計算文字大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 繪製背景矩形
    cv2.rectangle(output,
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + baseline + padding),
                 bg_color, -1)

    # 繪製文字
    cv2.putText(output, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return output


def extract_green_region(frame, lower_bounds=(0, 100, 0), upper_bounds=(160, 255, 154)):
    """
    提取影像中的綠色區域

    Args:
        frame: 輸入影像
        lower_bounds: 下界
        upper_bounds: 上界

    Returns:
        遮罩後的影像
    """
    lower = np.array(lower_bounds)
    upper = np.array(upper_bounds)

    mask = cv2.inRange(frame, lower, upper)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return frame & mask_rgb


def apply_morphology(image, operation=cv2.MORPH_CLOSE, kernel_size=(9, 9), iterations=2):
    """
    應用形態學操作

    Args:
        image: 輸入影像
        operation: 操作類型 (MORPH_CLOSE, MORPH_OPEN, etc.)
        kernel_size: 核大小
        iterations: 迭代次數

    Returns:
        處理後的影像
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, operation, kernel, iterations=iterations)
