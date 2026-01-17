import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, Canvas
import customtkinter as ctk
from PIL import Image, ImageTk

from .video_player import BadmintonVideoPlayer


class BadmintonAnalysisApp:
    """羽球分析系統主應用程式 - CustomTkinter 版本"""

    def __init__(self):
        # 設定 CustomTkinter 外觀
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # 應用程式狀態
        self.play = True
        self.delay = 0.023
        self.frame = 1
        self.frames = None

        # 其他變數
        self.vid_player = None
        self.photo = None
        self.photo2 = None
        self.scores = "0\n0"
        self.speed = "sec/frame"
        self.player_names = "Player1\nPlayer2"
        self.vid_width = 1024
        self.vid_height = 576

        # 性能優化 - UI 更新計數器
        self.ui_update_counter = 0
        self.slider_updating = False  # 防止滑桿更新循環
        try:
            from config import Config
            self.ui_update_interval = Config.Performance.UI_UPDATE_INTERVAL
        except:
            self.ui_update_interval = 3  # 預設每3幀更新一次 UI 文字

        # 創建主視窗
        self.root = ctk.CTk()
        self.root.title("Badminton Analysis System")
        self.root.geometry("1600x900")

        # 建立 UI
        self.create_widgets()

        # 啟動影片更新線程
        self.load_video()

    def create_widgets(self):
        """創建所有 GUI 元件"""
        # 主框架 - 使用 grid 佈局
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 左側主欄位
        main_frame = ctk.CTkFrame(self.root, fg_color="#ffffff")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 右側資訊欄位
        info_frame = ctk.CTkFrame(self.root, fg_color="#cfcfcf")
        info_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # === 左側主欄位內容 ===
        # 檔案選擇區
        file_label = ctk.CTkLabel(
            main_frame,
            text="Open video",
            text_color="#2C3539",
            font=("Arial", 11)
        )
        file_label.pack(anchor="w", padx=10, pady=(10, 5))

        file_frame = ctk.CTkFrame(main_frame, fg_color="#ffffff")
        file_frame.pack(fill="x", padx=10, pady=5)

        self.file_entry = ctk.CTkEntry(
            file_frame,
            placeholder_text="Select video file...",
            width=800,
            height=30
        )
        self.file_entry.pack(side="left", padx=(0, 5))

        browse_btn = ctk.CTkButton(
            file_frame,
            text="Browse",
            command=self.browse_file,
            fg_color="#389fff",
            text_color="#ffffff",
            width=100,
            height=30
        )
        browse_btn.pack(side="left")

        # 主影片畫布
        self.main_canvas = Canvas(
            main_frame,
            width=1024,
            height=576,
            bg="#cfcfcf",
            highlightthickness=2,
            highlightbackground="#cccccc"
        )
        self.main_canvas.pack(padx=10, pady=10)

        # 控制區
        control_frame = ctk.CTkFrame(main_frame, fg_color="#ffffff")
        control_frame.pack(fill="x", padx=10, pady=5)

        # 進度滑桿和計數器
        self.slider = ctk.CTkSlider(
            control_frame,
            from_=0,
            to=1000,
            number_of_steps=1000,
            command=self.slider_changed,
            width=900,
            fg_color="#389fff",
            progress_color="#389fff"
        )
        self.slider.pack(side="left", padx=(0, 10))

        self.counter_label = ctk.CTkLabel(
            control_frame,
            text="0/0",
            text_color="#2C3539",
            font=("Arial", 11),
            width=100
        )
        self.counter_label.pack(side="left")

        # 按鈕區
        button_frame = ctk.CTkFrame(main_frame, fg_color="#ffffff")
        button_frame.pack(pady=10)

        self.play_button = ctk.CTkButton(
            button_frame,
            text="Pause",
            command=self.toggle_play,
            fg_color="#389fff",
            text_color="#ffffff",
            width=100,
            height=30
        )
        self.play_button.pack(side="left", padx=5)

        next_btn = ctk.CTkButton(
            button_frame,
            text="Next frame",
            command=self.next_frame,
            fg_color="#389fff",
            text_color="#ffffff",
            width=100,
            height=30
        )
        next_btn.pack(side="left", padx=5)

        exit_btn = ctk.CTkButton(
            button_frame,
            text="Exit",
            command=self.exit_app,
            fg_color="#389fff",
            text_color="#ffffff",
            width=100,
            height=30
        )
        exit_btn.pack(side="left", padx=5)

        # === 右側資訊欄位內容 ===
        # 球軌跡畫布
        self.ball_canvas = Canvas(
            info_frame,
            width=500,
            height=350,
            bg="#ffffff",
            highlightthickness=0
        )
        self.ball_canvas.pack(padx=10, pady=(20, 30))

        # Match score 標題
        score_title = ctk.CTkLabel(
            info_frame,
            text="Match score",
            text_color="#2C3539",
            font=("Arial", 16),
            fg_color="#ffffff",
            width=400
        )
        score_title.pack(pady=(0, 10))

        # 球員名稱和比分區
        player_score_frame = ctk.CTkFrame(info_frame, fg_color="#cfcfcf")
        player_score_frame.pack(pady=(0, 30))

        self.player_name_label = ctk.CTkLabel(
            player_score_frame,
            text="Player1\nPlayer2",
            text_color="#2C3539",
            font=("Arial", 16),
            fg_color="#cfcfcf",
            width=180,
            height=80,
            justify="center"
        )
        self.player_name_label.pack(side="left", padx=5)

        self.score_label = ctk.CTkLabel(
            player_score_frame,
            text="0\n0",
            text_color="#2C3539",
            font=("Arial", 16),
            fg_color="#cfcfcf",
            width=180,
            height=80,
            justify="center"
        )
        self.score_label.pack(side="left", padx=5)

        # 性能資訊
        self.speed_label = ctk.CTkLabel(
            info_frame,
            text="sec/frame",
            text_color="#389fff",
            font=("Arial", 16),
            fg_color="#e0e0e0",
            width=400,
            height=120,
            justify="right",
            anchor="e"
        )
        self.speed_label.pack(pady=(0, 30), padx=10)

    def browse_file(self):
        """瀏覽並載入影片檔案"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            print(f"Loading video: {filename}")
            try:
                # 初始化影片播放器
                self.vid_player = BadmintonVideoPlayer(filename)

                # 計算影片尺寸
                self.vid_width = 1024
                self.vid_height = int(self.vid_width * self.vid_player.height / self.vid_player.width)
                self.frames = int(self.vid_player.frames)

                # 更新滑桿
                self.slider.configure(from_=0, to=self.frames)
                self.slider.set(0)
                self.counter_label.configure(text=f"0/{self.frames}")

                # 調整 canvas 大小
                self.main_canvas.configure(width=self.vid_width, height=self.vid_height)
                self.ball_canvas.configure(width=500, height=350)

                # 重置幀計數
                self.frame = 0
                self.delay = 1 / self.vid_player.fps

                # 更新檔案路徑
                self.file_entry.delete(0, tk.END)
                self.file_entry.insert(0, filename)

                print(f"Video loaded successfully: {self.frames} frames at {self.vid_player.fps} fps")
            except Exception as e:
                print(f"Error loading video: {e}")
                import traceback
                traceback.print_exc()

    def toggle_play(self):
        """切換播放/暫停狀態"""
        self.play = not self.play
        if self.play:
            self.play_button.configure(text="Pause")
        else:
            self.play_button.configure(text="Play")

    def next_frame(self):
        """跳至下一幀"""
        self.set_frame(self.frame + 1)

    def slider_changed(self, value):
        """滑桿值改變時跳轉幀"""
        if self.slider_updating:
            return
        self.play = False  # 拖拉時暫停播放
        self.play_button.configure(text="Play")
        self.set_frame(int(value))

    def set_frame(self, frame_no):
        """跳轉到指定幀"""
        if self.vid_player:
            ret, frame = self.vid_player.goto_frame(frame_no)
            self.frame = frame_no
            self.update_counter(self.frame)

            if ret:
                self.photo = ImageTk.PhotoImage(
                    image=Image.fromarray(frame).resize(
                        (self.vid_width, self.vid_height),
                        Image.NEAREST
                    )
                )
                self.main_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_counter(self, frame):
        """更新滑桿和幀計數器"""
        self.slider_updating = True
        self.slider.set(frame)
        self.slider_updating = False
        self.counter_label.configure(text=f"{frame}/{self.frames}")

    def load_video(self):
        """啟動影片更新線程"""
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = 1
        thread.start()

    def update(self):
        """更新畫布，顯示下一幀"""
        start_time = time.time()

        try:
            if self.vid_player:
                if self.play:
                    # 處理影像幀
                    ret, frame, smallframe, scores, player_names, speed = self.vid_player.process_frame()

                    if ret:
                        self.scores = scores
                        self.player_names = player_names
                        self.speed = speed

                        # 更新主畫面
                        self.photo = ImageTk.PhotoImage(
                            image=Image.fromarray(frame).resize(
                                (self.vid_width, self.vid_height),
                                Image.NEAREST
                            )
                        )
                        self.main_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                        # 更新球軌跡畫面
                        self.photo2 = ImageTk.PhotoImage(
                            image=Image.fromarray(smallframe).resize((500, 350), Image.NEAREST)
                        )
                        self.ball_canvas.create_image(0, 0, image=self.photo2, anchor=tk.NW)

                        # 更新幀計數
                        self.frame = self.vid_player.video_processor.get_current_frame_number()
                        self.update_counter(self.frame)

                        # 降低 UI 文字更新頻率以提升性能
                        self.ui_update_counter += 1
                        if self.ui_update_counter >= self.ui_update_interval:
                            self.score_label.configure(text=self.scores)
                            self.player_name_label.configure(text=self.player_names)
                            self.speed_label.configure(text=self.speed)
                            self.ui_update_counter = 0
        except Exception as e:
            print(f"Error processing frame: {e}")

        # 遞迴更新
        delay_ms = abs(int((self.delay - (time.time() - start_time)) * 1000))
        self.root.after(delay_ms, self.update)

    def exit_app(self):
        """退出應用程式"""
        print("Exiting application...")
        self.root.quit()
        self.root.destroy()
        sys.exit()

    def run(self):
        """運行主事件循環"""
        self.root.mainloop()
