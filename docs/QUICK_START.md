# 快速開始指南

## 0. 系統需求

### Python 版本要求
**本專案需要 Python 3.9 或以上版本**

檢查你目前的 Python 版本：
```bash
# Windows 用戶（推薦）
py --version

# 或如果已設定為預設
python --version
```

如果版本低於 3.9，請參考下方升級指南。

### 升級 Python（如果需要）

**Windows 用戶：**
1. 前往 Python 官網下載：https://www.python.org/downloads/
2. 下載 Python 3.9 或更新版本（推薦 3.9.x 或 3.10.x）
3. 安裝時勾選 "Add Python to PATH"
4. 安裝完成後，重新開啟終端機
5. 驗證版本：
   ```bash
   # 使用 Python Launcher（推薦，可指定版本）
   py -3.9 --version

   # 或使用預設 Python（如果已設定 PATH）
   python --version
   ```

**macOS 用戶：**
```bash
# 使用 Homebrew
brew install python@3.9
```

**Linux 用戶：**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv

# 設定為預設版本
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9
```

## 1. 環境設置

### 關於 Windows 多版本 Python

如果您的系統上安裝了多個 Python 版本（例如 3.7、3.9、3.11），建議：

**選項 1: 使用 Python Launcher（推薦）**
- 不需要修改環境變數
- 使用 `py -3.9` 明確指定版本
- 可以保留其他版本共存

**選項 2: 只保留 Python 3.9**
1. 透過「設定」→「應用程式」卸載其他版本
2. 確保 Python 3.9 在環境變數 PATH 中
3. 之後可直接使用 `python` 命令

本指南使用 `py -3.9` 命令，適用於多版本環境。

### 建立虛擬環境

**為什麼要用虛擬環境？**
- 隔離專案依賴，避免版本衝突
- 保持系統 Python 環境乾淨
- 便於管理不同專案的套件

**步驟 1: 進入專案目錄**
```bash
cd C:\Users\user\Desktop\CV_Project
```

**步驟 2: 建立虛擬環境**

使用 Python 3.9 內建的 venv：
```bash
# Windows 用戶（推薦，確保使用 Python 3.9）
py -3.9 -m venv computer_vision

# 或如果 Python 3.9 已是預設版本
python -m venv computer_vision
```

**步驟 3: 啟動虛擬環境**

在 Windows (Command Prompt)：
```bash
computer_vision\Scripts\activate
```

在 Windows (PowerShell)：
```bash
computer_vision\Scripts\Activate.ps1
```

在 Windows (Git Bash / MINGW)：
```bash
source computer_vision/Scripts/activate
```

在 macOS/Linux：
```bash
source computer_vision/bin/activate
```

啟動成功後，命令列前面會出現 `(computer_vision)` 字樣。

附註: 如何取消

(取消虛擬環境)
Step 1: Deactivate the current virtual environment
```bash
deactivate
```
Step 2: Remove the existing venv folder
```bash
rm -rf computer_vision
```
Step 3: Create a fresh virtual environment
```bash
# Windows 用戶（確保使用 Python 3.9）
py -3.9 -m venv computer_vision

# 或如果 Python 3.9 已是預設版本
python -m venv computer_vision
```
Step 4: Activate the new virtual environment
```bash
source computer_vision/Scripts/activate
```


**步驟 4: 升級 pip**
```bash
python -m pip install --upgrade pip
```

### 安裝 Python 依賴

**安裝所有依賴套件**
```bash
pip install -r requirements.txt
```

這會安裝所有必要的套件，包括：
- CustomTkinter（免費 GUI 框架）
- OpenCV、MediaPipe（電腦視覺）
- PyTorch、YOLOv5（深度學習）
- Tesseract OCR（文字識別）

**驗證安裝：**
```bash
pip list | grep -E "customtkinter|opencv|torch"
```

**關閉虛擬環境（當不再使用時）：**
```bash
deactivate
```

### 安裝 Tesseract OCR

下載並安裝: https://github.com/UB-Mannheim/tesseract/wiki

確保安裝到: `C:\Program Files\Tesseract-OCR\`

## 2. 準備資料

### 模型權重
模型已經包含在專案中:
```
models/yolov5/weights/best.pt
```

### 測試影片
測試影片位於:
```
data/videos/demo.mp4
```

### BWF 模板（如果需要 OCR）
如果你的影片有 BWF 比分板，需要準備模板圖片並放置於:
```
data/templates/BWF.jpg
```

## 3. 運行應用程式

**重要：請確保已啟動虛擬環境**
```bash
source computer_vision/Scripts/activate  # Git Bash / MINGW
# 或
computer_vision\Scripts\activate  # Windows CMD
```

### 推薦方式：從專案根目錄運行

確保在專案根目錄 `C:\Users\user\Desktop\CV_Project`，然後執行：
```bash
python -m src.main
```

**為什麼要這樣運行？**
- 正確處理 Python 模組的相對導入
- 避免路徑和導入錯誤
- 符合 Python 專案最佳實踐

## 4. 使用介面

1. **載入影片**
   - 點擊 "Browse" 按鈕
   - 選擇影片檔案
   - 影片會自動開始分析

2. **控制播放**
   - `Pause/Play`: 暫停或播放
   - `Next frame`: 跳到下一幀
   - `Slider`: 拖曳到指定位置

3. **查看分析結果**
   - 主畫面: 顯示場地檢測、物體檢測、骨架追踪
   - 右上方: 球軌跡圖
   - 右中方: 比賽比分和球員名稱
   - 右下方: 處理時間資訊

## 5. 常見問題排除

### 問題 1: 無法載入 YOLO 模型

**錯誤**: `FileNotFoundError: models/yolov5/weights/best.pt`

**解決方法**:
```bash
# 確認檔案存在
ls models/yolov5/weights/best.pt

# 如果不存在，檢查 yolov5/runs/train/exp/weights/ 目錄
cp yolov5/runs/train/exp/weights/best.pt models/yolov5/weights/
```

### 問題 2: Tesseract OCR 錯誤

**錯誤**: `TesseractNotFoundError`

**解決方法**:
修改 `src/ocr/score_reader.py` 中的路徑:
```python
pytesseract.pytesseract.tesseract_cmd = r'你的Tesseract安裝路徑'
```

### 問題 3: CUDA 錯誤

**錯誤**: `CUDA out of memory`

**解決方法**:
在 `src/detection/yolo_detector.py` 中，註解掉 GPU 使用:
```python
# self.model.cuda()  # 改用 CPU
```

### 問題 4: 相對路徑錯誤

**錯誤**: 無法找到 `./yolov5` 或其他檔案

**解決方法**:
確保從正確的目錄運行程式:
```bash
# 應該在專案根目錄
cd C:\Users\user\Desktop\CV_Project
cd src
python main.py
```

## 6. 測試單一模組

### 測試場地檢測

創建 `notebooks/test_court_detection.ipynb`:

```python
import sys
sys.path.append('../src')

from detection.court_detector import CourtDetector
import cv2

# 載入測試影像
frame = cv2.imread('../data/videos/test_frame.jpg')

# 檢測場地
detector = CourtDetector()
is_ok, corners, output, area = detector.detect(frame)

print(f"場地檢測: {is_ok}")
print(f"面積: {area}%")

# 顯示結果
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.show()
```

### 測試 YOLO 檢測

```python
from detection.yolo_detector import YOLODetector
import cv2

frame = cv2.imread('../data/videos/test_frame.jpg')

detector = YOLODetector()
labels, bbox, result = detector.detect(frame)

print(f"檢測到 {len(labels)} 個物體")
```

## 7. 效能優化建議

### 使用 GPU (如果可用)

確認 CUDA 已安裝:
```python
import torch
print(torch.cuda.is_available())  # 應該返回 True
```

### 降低影片解析度

在 `video_player.py` 中調整:
```python
ww, hh = 640, 360  # 原本是 800, 450
```

### 跳過幀數

在 `app.py` 中調整:
```python
self.frame += 60  # 原本是 30，現在每次跳 2 倍幀
```

## 8. 開發建議

### 添加新功能

1. 在 Jupyter Notebook 中原型開發
2. 測試功能
3. 轉換為 .py 模組
4. 整合到主程式

### 程式碼格式化

```bash
# 安裝工具
pip install black flake8

# 格式化程式碼
black src/

# 檢查程式碼品質
flake8 src/
```

### 執行測試

```bash
# 安裝 pytest
pip install pytest

# 執行測試
pytest tests/
```

## 9. 進階使用

### 批次處理影片

創建 `scripts/batch_process.py`:

```python
import os
from src.gui.video_player import BadmintonVideoPlayer

video_dir = "data/videos/"
output_dir = "data/output/"

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        print(f"Processing {video_file}...")
        player = BadmintonVideoPlayer(os.path.join(video_dir, video_file))
        # 處理影片...
```

### 匯出分析結果

在 `video_player.py` 中添加:

```python
def export_analysis(self, output_path):
    """匯出分析結果到 CSV"""
    import pandas as pd

    data = {
        'frame': self.frame_numbers,
        'ball_x': self.ball_x_positions,
        'ball_y': self.ball_y_positions,
        'score': self.scores
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
```

## 10. 獲取幫助

- 查看 README.md
- 閱讀 docs/PROJECT_STRUCTURE.md
- 檢查範例 notebooks/
- 提交 Issue 到 GitHub

---

祝你使用愉快！
