import cv2


class VideoProcessor:
    """影片處理工具類"""

    def __init__(self, video_source):
        """
        初始化影片處理器

        Args:
            video_source: 影片檔案路徑
        """
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # 取得影片屬性
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

    def read_frame(self):
        """
        讀取下一幀

        Returns:
            tuple: (ret, frame)
        """
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            return ret, frame
        return False, None

    def goto_frame(self, frame_no):
        """
        跳轉到指定幀

        Args:
            frame_no: 幀號

        Returns:
            tuple: (ret, frame)
        """
        if self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return ret, None
        return False, None

    def get_current_frame_number(self):
        """
        取得當前幀號

        Returns:
            int: 當前幀號
        """
        return int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))

    def release(self):
        """釋放影片資源"""
        if self.vid.isOpened():
            self.vid.release()

    def __del__(self):
        """析構函數"""
        self.release()


def resize_frame(frame, target_width=800):
    """
    調整影像大小，保持長寬比

    Args:
        frame: 輸入影像
        target_width: 目標寬度

    Returns:
        調整後的影像
    """
    height, width = frame.shape[:2]
    target_height = int(target_width * height / width)
    return cv2.resize(frame, (target_width, target_height))


def bgr_to_rgb(frame):
    """
    將 BGR 轉換為 RGB

    Args:
        frame: BGR 影像

    Returns:
        RGB 影像
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame):
    """
    將 RGB 轉換為 BGR

    Args:
        frame: RGB 影像

    Returns:
        BGR 影像
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
