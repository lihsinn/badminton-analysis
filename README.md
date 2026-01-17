# 羽球分析系統 (Badminton Analysis System)

這是一個基於深度學習和電腦視覺的羽球比賽影片分析系統，
能夠自動檢測場地、追踪球員動作、識別羽球軌跡以及讀取比分。

## Demo 展示

https://github.com/user-attachments/assets/c76b885a-6d19-4940-b00e-97c0101eed77

## 功能特色
1. ✅ **場地偵測**（Court Detection）- 自動識別羽球場地邊界(綠色區域檢測 + 多邊形逼近)
2. ✅ **比分辨識**（Tesseract OCR）- 讀取比分和選手名稱
3. ✅ **物件偵測**（YOLO v5）- 偵測選手和羽球
4. ✅ **骨架追蹤**（MediaPipe Pose）- 追蹤選手姿態
5. ✅ **軌跡追蹤**（Ball Tracking）- 追蹤羽球運動軌跡
6. ✅ **GUI 介面**（PySimpleGUI + Tkinter）- 影片播放、控制、結果顯示


## 專案結構

```
badminton-analysis/
│
├── src/                      # 主程式碼
│   ├── main.py              # 主程式入口
│   ├── gui/                 # GUI 相關模組
│   │   ├── app.py          # 主應用程式
│   │   └── video_player.py  # 影片播放器
│   │
│   ├── detection/           # 檢測相關模組
│   │   ├── court_detector.py    # 場地檢測
│   │   ├── yolo_detector.py     # YOLO 物體檢測
│   │   ├── skeleton_tracker.py  # 骨架追踪
│   │   └── ball_tracker.py      # 球軌跡追踪
│   │
│   ├── ocr/                 # OCR 相關
│   │   └── score_reader.py      # 比分識別
│   │
│   └── utils/               # 工具函數
│       ├── video_utils.py       # 影片處理工具
│       └── image_processing.py  # 影像處理工具
│
├── models/                  # 模型檔案
│   └── yolov5/             # YOLOv5 模型
│       └── weights/
│           └── best.pt
│
├── data/                    # 資料檔案
│   ├── videos/             # 測試影片
│   ├── templates/          # 模板圖片
│   └── output/             # 輸出結果
│
├── notebooks/              # Jupyter Notebooks（可選）
├── tests/                  # 測試檔案
├── docs/                   # 文檔
├── requirements.txt        # 相依套件
└── README.md              # 專案說明
```

## 環境需求

### 必要軟體

- Python 3.8 或以上
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (需要安裝在 `C:\Program Files\Tesseract-OCR\`)
- CUDA (可選，用於 GPU 加速)

### Python 套件

請參考 `requirements.txt`

## 安裝步驟

1. **Clone 專案**
   ```bash
   git clone <repository-url>
   cd badminton-analysis
   ```

2. **安裝 Python 依賴**
   ```bash
   pip install -r requirements.txt
   ```

3. **安裝 Tesseract OCR**
   - Windows: 下載並安裝 [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

4. **下載 YOLOv5 模型權重**
   - 模型權重已包含在 `models/yolov5/weights/best.pt`
   - 或訓練自己的模型並放置於該目錄

## 使用方法

### 啟動應用程式

```bash
source computer_vision/bin/activate
cd CV_Project
computer_vision/Scripts/python.exe -m src.main
```

### 使用介面

1. 點擊 "Browse" 按鈕選擇影片檔案
2. 影片會自動開始播放並進行分析
3. 使用控制按鈕：
   - **Pause/Play**: 暫停/播放影片
   - **Next frame**: 跳到下一幀
   - **Slider**: 拖曳滑桿跳轉到指定位置
4. 右側面板顯示：
   - 球軌跡圖
   - 比賽比分
   - 球員名稱
   - 處理時間資訊

## 技術架構

### 核心技術

- **YOLOv5**: 物體檢測（羽球、球員）
- **MediaPipe**: 人體姿態估計
- **OpenCV**: 影像處理和場地檢測
- **Tesseract OCR**: 文字識別（比分、球員名稱）
- **PySimpleGUI**: 圖形使用者介面
- **PyTorch**: 深度學習框架

### 處理流程

1. **影像輸入**: 讀取影片幀
2. **場地檢測**: 使用顏色遮罩和輪廓檢測識別場地
3. **物體檢測**: YOLOv5 檢測羽球和球員
4. **骨架追踪**: MediaPipe 追踪球員姿態
5. **軌跡追踪**: 記錄並繪製羽球運動軌跡
6. **比分識別**: OCR 讀取螢幕上的比分資訊
7. **結果顯示**: 在 GUI 中顯示分析結果

## 開發指南

### 添加新功能

1. 在對應的模組中創建新類別
2. 在 `video_player.py` 中整合新功能
3. 更新 GUI 介面（如需要）

### 訓練自己的 YOLOv5 模型

```bash
cd models/yolov5
python train.py --data your_data.yaml --weights yolov5s.pt --epochs 100
```

### 運行測試

```bash
pytest tests/
```

## 性能優化

本專案已經過優化，可以實現更流暢的影片分析體驗：

### 已實現的優化

1. **條件檢測** - 降低不必要的計算
   - 場地檢測每 30 幀執行一次（可在 `config.py` 調整）
   - OCR 比分識別每 90 幀執行一次
   - 其他幀使用快取結果

2. **GPU 加速**
   - YOLO 模型自動使用 CUDA（如果可用）
   - 啟用 `torch.backends.cudnn.benchmark` 優化
   - 使用 `torch.no_grad()` 減少記憶體使用

3. **資源重用**
   - MediaPipe Pose 對象只初始化一次，避免重複創建
   - 減少不必要的物件複製

4. **UI 優化**
   - 降低 UI 文字更新頻率（每 3 次循環更新一次）
   - 使用 `timeout` 參數避免阻塞

### 性能調整

在 `config.py` 中修改以下參數來調整性能：

```python
class Performance:
    # GPU 設定
    USE_GPU = True  # 是否使用 GPU

    # 檢測間隔（數值越大，性能越好但更新越慢）
    COURT_CHECK_INTERVAL = 30  # 場地檢測間隔
    OCR_INTERVAL = 90  # OCR 檢測間隔

    # UI 更新頻率
    UI_UPDATE_INTERVAL = 3  # UI 文字更新間隔
```

### 效能提升

優化後預期效能提升：
- 處理速度提升 40-60%（取決於硬體配置）
- CPU 使用率降低 30-50%
- 記憶體使用更穩定
- UI 響應更流暢

## 常見問題

### Q: Tesseract OCR 無法識別文字？
A: 確認 Tesseract 已正確安裝，並在 `config.py` 中設定正確的路徑。

### Q: YOLO 檢測速度太慢？
A:
1. 安裝 CUDA 並確保 PyTorch 使用 GPU
2. 在 `config.py` 中設定 `USE_GPU = True`
3. 調整檢測間隔參數以降低計算頻率

### Q: 無法找到模型檔案？
A: 確認模型權重檔案 `best.pt` 位於 `models/yolov5/weights/` 目錄。

### Q: 系統記憶體不足？
A:
1. 降低檢測頻率（增加 `COURT_CHECK_INTERVAL` 和 `OCR_INTERVAL`）
2. 減少軌跡緩衝區大小（在 `config.py` 中調整 `BUFFER_SIZE`）
3. 使用較小的 YOLO 模型

## 授權

此專案僅供學習和研究使用。

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 聯絡

如有問題請聯繫我。

