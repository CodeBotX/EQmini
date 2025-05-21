#Tạo giao diện đơn giản với PyQt5   

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from equalizer_processor import EqualizerProcessor
import os
import sounddevice as sd
import numpy as np
import threading
import queue
import traceback
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   filename='equalizer.log')
logger = logging.getLogger(__name__)

class EqualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Equalizer")
        
        # Khởi tạo audio processor
        self.audio_processor = EqualizerProcessor()
        
        # Queue để truyền dữ liệu âm thanh giữa các thread
        self.audio_queue = queue.Queue()
        
        # Tần số các dải
        self.bands = {
            'Bass': (20, 250),
            'Mid': (250, 4000),
            'Treble': (4000, 20000)
        }
        
        # Các tần số cụ thể
        self.specific_freqs = [32, 64, 125, 250, 500, 1000, 2000, 4000, 6000]
        
        # Tạo frame chính
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame cho phần điều khiển file
        self.file_frame = ttk.LabelFrame(self.main_frame, text="Điều khiển file", padding="5")
        self.file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Nút chọn file
        self.select_button = ttk.Button(self.file_frame, text="Chọn file", command=self.select_file)
        self.select_button.grid(row=0, column=0, padx=5)
        
        # Label hiển thị tên file
        self.file_label = ttk.Label(self.file_frame, text="Chưa chọn file")
        self.file_label.grid(row=0, column=1, padx=5)
        
        # Label trạng thái phát
        self.status_label = ttk.Label(self.file_frame, text="")
        self.status_label.grid(row=0, column=2, padx=5)
        
        # Frame cho phần điều khiển phát
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Nút Play/Pause
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=5)
        
        # Thanh thời gian
        self.time_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=300)
        self.time_slider.grid(row=0, column=1, padx=5)
        self.time_slider.bind("<ButtonRelease-1>", self.seek_audio)
        
        # Label thời gian
        self.time_label = ttk.Label(self.control_frame, text="00:00 / 00:00")
        self.time_label.grid(row=0, column=2, padx=5)
        
        # Tạo các thanh trượt cho dải tần
        self.band_sliders = {}
        row = 2
        for band, (low, high) in self.bands.items():
            ttk.Label(self.main_frame, text=f"{band} ({low}-{high} Hz)").grid(row=row, column=0, pady=5)
            slider = ttk.Scale(self.main_frame, from_=-12, to=12, orient=tk.HORIZONTAL, length=200)
            slider.set(0)
            slider.grid(row=row, column=1, pady=5)
            slider.bind("<ButtonRelease-1>", self.on_slider_change)
            self.band_sliders[band] = slider
            row += 1
        
        # Tạo các thanh trượt cho tần số cụ thể
        ttk.Label(self.main_frame, text="Tần số cụ thể").grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        self.freq_sliders = {}
        for freq in self.specific_freqs:
            ttk.Label(self.main_frame, text=f"{freq} Hz").grid(row=row, column=0, pady=2)
            slider = ttk.Scale(self.main_frame, from_=-12, to=12, orient=tk.HORIZONTAL, length=200)
            slider.set(0)
            slider.grid(row=row, column=1, pady=2)
            slider.bind("<ButtonRelease-1>", self.on_slider_change)
            self.freq_sliders[freq] = slider
            row += 1
        
        # Điều khiển âm lượng
        ttk.Label(self.main_frame, text="Âm lượng chính").grid(row=row, column=0, pady=10)
        self.volume_slider = ttk.Scale(self.main_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
        self.volume_slider.set(70)
        self.volume_slider.grid(row=row, column=1, pady=10)
        self.volume_slider.bind("<ButtonRelease-1>", self.on_slider_change)
        
        # Biến để theo dõi trạng thái phát
        self.is_playing = False
        self.current_file = None
        self.stream = None
        self.processing_thread = None
        self.stop_processing = False
        self.audio_data = None
        self.lock = threading.Lock()  # Thêm lock để đồng bộ hóa
        
        # Bắt đầu cập nhật thời gian
        self.update_time_slider()
        
    def select_file(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.flac")]
            )
            if file_path:
                self.current_file = file_path
                self.file_label.config(text=os.path.basename(file_path))
                if self.audio_processor.load_audio_file(file_path):
                    self.audio_data = self.audio_processor.audio_data.copy()
                    self.time_slider.config(to=self.audio_processor.get_total_time())
                    self.play_button.config(text="Play")
                    self.is_playing = False
                    self.stop_processing = True
                    self.cleanup_stream()
                    logger.info(f"Đã tải file: {file_path}")
                else:
                    messagebox.showerror("Lỗi", "Không thể tải file âm thanh")
        except Exception as e:
            logger.error(f"Lỗi khi tải file: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Không thể tải file: {str(e)}")
            
    def cleanup_stream(self):
        """Dọn dẹp stream an toàn"""
        with self.lock:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                finally:
                    self.stream = None
            
    def toggle_play(self):
        if not self.current_file:
            return
            
        try:
            if self.is_playing:
                self.stop_processing = True
                self.cleanup_stream()
                self.play_button.config(text="Play")
                self.status_label.config(text="Đã tạm dừng")
            else:
                self.stop_processing = False
                # Bắt đầu thread xử lý âm thanh
                self.processing_thread = threading.Thread(target=self.process_and_play_audio)
                self.processing_thread.daemon = True
                self.processing_thread.start()
                
                self.play_button.config(text="Pause")
                self.status_label.config(text="Đang phát")
            self.is_playing = not self.is_playing
        except Exception as e:
            logger.error(f"Lỗi khi phát/tạm dừng: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Không thể phát/tạm dừng: {str(e)}")
            self.is_playing = False
            self.play_button.config(text="Play")
        
    def process_and_play_audio(self):
        """Xử lý và phát âm thanh trong thread riêng"""
        try:
            if self.audio_data is None:
                logger.error("Không có dữ liệu âm thanh để xử lý")
                return
                
            # Lấy dữ liệu âm thanh đã xử lý
            band_values = {band: slider.get() for band, slider in self.band_sliders.items()}
            freq_values = {freq: slider.get() for freq, slider in self.freq_sliders.items()}
            volume = self.volume_slider.get() / 100.0
            
            # Xử lý âm thanh
            processed_audio = self.audio_processor.process_audio(band_values, freq_values, volume)
            if processed_audio is None:
                logger.error("Không thể xử lý âm thanh")
                return
                
            # Phát âm thanh
            with self.lock:
                self.stream = sd.OutputStream(samplerate=self.audio_processor.sample_rate, channels=1)
                self.stream.start()
            
            # Chia nhỏ dữ liệu để phát
            chunk_size = 1024
            for i in range(0, len(processed_audio), chunk_size):
                if self.stop_processing:
                    break
                chunk = processed_audio[i:i + chunk_size]
                if len(chunk) > 0:
                    with self.lock:
                        if self.stream is not None:
                            self.stream.write(chunk)
                            self.audio_processor.current_time = i / self.audio_processor.sample_rate
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý âm thanh: {str(e)}")
            logger.error(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi xử lý âm thanh: {str(e)}"))
        finally:
            self.cleanup_stream()
            self.is_playing = False
            self.root.after(0, lambda: self.play_button.config(text="Play"))
            self.root.after(0, lambda: self.status_label.config(text="Đã dừng"))
        
    def seek_audio(self, event=None):
        try:
            if self.current_file:
                position = self.time_slider.get()
                self.audio_processor.current_time = position
                self.audio_processor.current_position = int(position * self.audio_processor.sample_rate)
        except Exception as e:
            logger.error(f"Lỗi khi seek: {str(e)}")
            logger.error(traceback.format_exc())
            
    def update_time_slider(self):
        try:
            if self.current_file and self.is_playing:
                current_time = self.audio_processor.get_current_time()
                total_time = self.audio_processor.get_total_time()
                self.time_slider.set(current_time)
                self.time_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật thời gian: {str(e)}")
        finally:
            self.root.after(100, self.update_time_slider)
        
    def format_time(self, seconds):
        try:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
        except:
            return "00:00"
        
    def on_slider_change(self, event=None):
        try:
            if not self.is_playing:
                return
                
            # Dừng phát hiện tại
            self.stop_processing = True
            self.cleanup_stream()
            
            # Bắt đầu phát lại với các giá trị mới
            self.processing_thread = threading.Thread(target=self.process_and_play_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
        except Exception as e:
            logger.error(f"Lỗi khi thay đổi slider: {str(e)}")
            logger.error(traceback.format_exc())
            messagebox.showerror("Lỗi", f"Không thể áp dụng thay đổi: {str(e)}")
