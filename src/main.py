"""
羽球分析系統 - 主程式入口

此程式提供羽球影片分析功能，包括：
- 場地檢測
- 球員骨架追踪
- 羽球軌跡追踪
- 比分識別（OCR）
"""

from .gui.app import BadmintonAnalysisApp


def main():
    """主函數"""
    print("Starting Badminton Analysis System...")
    app = BadmintonAnalysisApp()
    app.run()


if __name__ == '__main__':
    main()
