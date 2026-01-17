import cv2
import numpy as np
import pytesseract
import os


class ScoreReader:
    """使用 Tesseract OCR 識別比分和球員名稱"""

    def __init__(self, config=None):
        """
        初始化 OCR 讀取器

        Args:
            config: 配置字典，包含以下參數：
                - tesseract_path: Tesseract 執行檔路徑
                - tesseract_lang: OCR 語言
        """
        if config is None:
            config = {}

        # Tesseract 設定
        tesseract_path = config.get('tesseract_path', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            print(f"Warning: Tesseract path not found: {tesseract_path}")

        # OCR 參數
        self.tesseract_lang = config.get('tesseract_lang', 'eng')

    def read_score_and_names(self, frame):
        """
        讀取比分和球員名稱

        Args:
            frame: 原始影像

        Returns:
            tuple: (player_names, scores)
        """
        try:
            import re

            # 名字區域 (針對此影片格式)
            name_region = frame[55:140, 355:540]
            gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            enlarged = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # OCR 識別名字
            text = pytesseract.image_to_string(
                enlarged,
                lang=self.tesseract_lang,
                config='--psm 6'
            )

            lines = [line for line in text.split('\n') if line.strip()]

            player1 = "Player1"
            player2 = "Player2"

            # 處理第一行
            if len(lines) > 0:
                match = re.search(r'([A-Z][A-Z\s\.]+)', lines[0])
                if match:
                    player1 = match.group(1).strip()

            # 處理第二行
            if len(lines) > 1:
                match = re.search(r'([A-Z]{3,})', lines[1])
                if match:
                    player2 = match.group(1)

            # 比分區域 - 使用 HSV 提取白色數字
            score_region = frame[55:140, 595:640]
            hsv = cv2.cvtColor(score_region, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # 分開處理上下兩個比分
            h = mask.shape[0]
            score1_img = cv2.resize(mask[0:h//2, :], None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
            score2_img = cv2.resize(mask[h//2:, :], None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)

            t1 = pytesseract.image_to_string(score1_img, lang='eng', config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()
            t2 = pytesseract.image_to_string(score2_img, lang='eng', config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()

            # 只取第一個數字
            score1 = t1[0] if t1 and t1[0].isdigit() else "0"
            score2 = t2[0] if t2 and t2[0].isdigit() else "0"

            return f"{player1}\n{player2}", f"{score1}\n{score2}"

        except Exception as e:
            print(f"OCR Error: {e}")
            return "Player1\nPlayer2", "0\n0"
