# 專案結構說明

## 目錄結構詳解

### `src/` - 主程式碼目錄

#### `src/main.py`
- **用途**: 應用程式主入口點
- **功能**: 啟動 GUI 應用程式
- **類型**: `.py` (生產環境)

#### `src/gui/` - GUI 模組
- `app.py`: 主應用程式類別，處理 GUI 事件和使用者互動
- `video_player.py`: 影片播放器類別，整合所有檢測功能

**為什麼用 .py?**
- GUI 需要穩定的類別定義
- 更好的錯誤處理
- 容易打包成可執行檔

#### `src/detection/` - 檢測模組
- `court_detector.py`: 場地檢測器
  - 使用顏色遮罩提取綠色區域
  - 輪廓檢測找到場地邊界
  - 判斷場地是否符合標準

- `yolo_detector.py`: YOLO 物體檢測器
  - 載入預訓練的 YOLOv5 模型
  - 檢測羽球和球員
  - 提供邊界框和標籤

- `skeleton_tracker.py`: 骨架追踪器
  - 使用 MediaPipe Pose
  - 追踪球員關鍵點
  - 繪製骨架

- `ball_tracker.py`: 球軌跡追踪器
  - 維護軌跡點隊列
  - 繪製運動軌跡
  - 提供軌跡查詢功能

**為什麼用 .py?**
- 這些是核心功能模組
- 需要被其他程式碼重用
- 方便單元測試

#### `src/ocr/` - OCR 模組
- `score_reader.py`: 比分和球員名稱識別
  - 模板匹配找到比分板
  - 影像預處理
  - Tesseract OCR 識別

**為什麼用 .py?**
- OCR 邏輯複雜，需要模組化
- 可能需要在其他地方調用

#### `src/utils/` - 工具函數
- `video_utils.py`: 影片處理工具
- `image_processing.py`: 影像處理工具

**為什麼用 .py?**
- 通用工具函數
- 多處重用
- 方便測試

---

### `models/` - 模型檔案目錄

```
models/
└── yolov5/
    ├── (yolov5 原始碼)
    └── weights/
        └── best.pt  # 訓練好的模型權重
```

**存儲內容**:
- 預訓練模型權重
- 模型配置檔案
- YOLOv5 原始碼

---

### `data/` - 資料目錄

```
data/
├── videos/        # 測試和演示影片
├── templates/     # OCR 模板圖片 (如 BWF.jpg)
└── output/        # 處理結果輸出
```

**用途**:
- 儲存測試資料
- 模板匹配資源
- 輸出分析結果

---

### `notebooks/` - Jupyter Notebooks (可選)

**建議用途**:
1. `demo.ipynb` - 功能演示
   - 展示各個模組的功能
   - 視覺化結果
   - 互動式測試

2. `model_training.ipynb` - 模型訓練
   - 資料預處理
   - YOLOv5 訓練流程
   - 訓練結果視覺化

3. `analysis.ipynb` - 資料分析
   - 比賽統計分析
   - 球員動作分析
   - 視覺化報表

**為什麼用 .ipynb?**
- 互動式開發和測試
- 即時視覺化
- 適合實驗和探索
- 方便製作教學材料

---

### `tests/` - 測試目錄

**建議內容**:
```
tests/
├── test_detection.py    # 檢測模組測試
├── test_ocr.py          # OCR 模組測試
└── test_utils.py        # 工具函數測試
```

**為什麼用 .py?**
- 自動化測試需要穩定的腳本
- CI/CD 整合
- 使用 pytest 執行

---

### `docs/` - 文檔目錄

**建議內容**:
- `PROJECT_STRUCTURE.md` (本檔案)
- `API.md` - API 文檔
- `USER_GUIDE.md` - 使用者指南
- `DEVELOPMENT.md` - 開發指南

---

## .py vs .ipynb 使用指南

### 使用 .py 檔案當：

1. **生產環境程式碼**
   - 主程式、GUI、API

2. **可重用模組**
   - 類別定義
   - 工具函數
   - 檢測器

3. **自動化腳本**
   - 批次處理
   - 部署腳本

4. **測試程式碼**
   - 單元測試
   - 整合測試

### 使用 .ipynb 檔案當：

1. **探索性分析**
   - 資料探索
   - 視覺化
   - 實驗新方法

2. **教學和演示**
   - 步驟說明
   - 互動式教學
   - 結果展示

3. **模型訓練**
   - 訓練過程記錄
   - 超參數調整
   - 視覺化訓練曲線

4. **報告生成**
   - 分析報告
   - 統計圖表

---

## 模組依賴關係

```
main.py
  └── gui/app.py
        └── gui/video_player.py
              ├── detection/court_detector.py
              ├── detection/yolo_detector.py
              ├── detection/skeleton_tracker.py
              ├── detection/ball_tracker.py
              ├── ocr/score_reader.py
              └── utils/
                    ├── video_utils.py
                    └── image_processing.py
```

---

## 最佳實踐

### 1. 模組化
- 每個模組負責單一功能
- 使用類別封裝相關功能
- 避免循環依賴

### 2. 程式碼風格
- 遵循 PEP 8
- 使用有意義的變數名稱
- 添加適當的註解

### 3. 錯誤處理
- 使用 try-except 處理例外
- 提供有用的錯誤訊息
- 記錄錯誤日誌

### 4. 測試
- 為核心功能編寫測試
- 使用 pytest 自動化測試
- 保持測試覆蓋率

---

## 擴展指南

### 添加新的檢測功能

1. 在 `src/detection/` 創建新模組
2. 實現檢測類別
3. 在 `video_player.py` 中整合
4. 更新 GUI 顯示（如需要）

### 添加新的分析功能

1. 在 `notebooks/` 中創建實驗 notebook
2. 測試和驗證功能
3. 轉換為 .py 模組
4. 整合到主程式

---

## 常見問題

**Q: 為什麼不全部用 Jupyter Notebook?**
A: Notebook 適合探索和演示，但不適合生產環境。.py 檔案更容易版本控制、測試和部署。

**Q: 可以混合使用嗎?**
A: 可以！核心功能用 .py，實驗和演示用 .ipynb。

**Q: 如何在 Notebook 中使用這些模組?**
A: 在 notebook 中 import：
```python
import sys
sys.path.append('../src')
from detection.court_detector import CourtDetector
```
