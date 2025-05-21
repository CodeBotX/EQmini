import numpy as np
import soundfile as sf
from base_processor import BaseAudioProcessor
from filters import BandPassFilter, PeakingFilter
import logging
import traceback

logger = logging.getLogger(__name__)

class EqualizerProcessor(BaseAudioProcessor):
    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.current_position = 0
        self.current_time = 0
        self.total_time = 0
        
        # Hệ số khuếch đại cơ bản
        self.base_gain = 2.0  # Tăng âm lượng gấp đôi
        
        try:
            # Khởi tạo các bộ lọc dải tần
            self.band_filters = {
                'Bass': BandPassFilter(self.sample_rate, 20, 250),
                'Mid': BandPassFilter(self.sample_rate, 250, 4000),
                'Treble': BandPassFilter(self.sample_rate, 4000, 20000)
            }
            
            # Khởi tạo các bộ lọc peaking cho tần số cụ thể
            self.freq_filters = {
                32: PeakingFilter(self.sample_rate, 32, 0, 1),
                64: PeakingFilter(self.sample_rate, 64, 0, 1),
                125: PeakingFilter(self.sample_rate, 125, 0, 1),
                250: PeakingFilter(self.sample_rate, 250, 0, 1),
                500: PeakingFilter(self.sample_rate, 500, 0, 1),
                1000: PeakingFilter(self.sample_rate, 1000, 0, 1),
                2000: PeakingFilter(self.sample_rate, 2000, 0, 1),
                4000: PeakingFilter(self.sample_rate, 4000, 0, 1),
                6000: PeakingFilter(self.sample_rate, 6000, 0, 1)
            }
            logger.info("Đã khởi tạo các bộ lọc thành công")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo bộ lọc: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def load_audio_file(self, file_path):
        """Tải file âm thanh (sử dụng soundfile chỉ cho việc đọc file)"""
        try:
            logger.info(f"Đang tải file: {file_path}")
            self.audio_data, self.sample_rate = sf.read(file_path)
            
            if self.audio_data is None or len(self.audio_data) == 0:
                logger.error("Không thể đọc dữ liệu âm thanh")
                return False
                
            if len(self.audio_data.shape) > 1:  # Nếu là stereo, chuyển sang mono
                self.audio_data = np.mean(self.audio_data, axis=1)
                
            # Áp dụng hệ số khuếch đại cơ bản
            self.audio_data *= self.base_gain
                
            self.total_time = len(self.audio_data) / self.sample_rate
            self.current_position = 0
            self.current_time = 0
            
            logger.info(f"Đã tải file thành công: {len(self.audio_data)} samples, {self.sample_rate} Hz")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải file: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def process_audio(self, band_values, freq_values, volume):
        """Xử lý âm thanh với các bộ lọc equalizer"""
        try:
            if self.audio_data is None:
                logger.error("Không có dữ liệu âm thanh để xử lý")
                return None
                
            # Tạo bản sao của dữ liệu âm thanh
            processed_audio = self.audio_data.copy()
            
            # Áp dụng các bộ lọc dải tần
            for band, gain in band_values.items():
                if band in self.band_filters:
                    try:
                        # Thiết kế và áp dụng bộ lọc
                        filter_coeffs = self.band_filters[band].design()
                        processed_audio = self.apply_filter(processed_audio, filter_coeffs)
                        # Áp dụng gain
                        processed_audio *= (10 ** (gain / 20))  # Chuyển dB thành tỷ lệ
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý dải {band}: {str(e)}")
                        continue
            
            # Áp dụng các bộ lọc tần số cụ thể
            for freq, gain in freq_values.items():
                if freq in self.freq_filters:
                    try:
                        self.freq_filters[freq].gain_db = gain
                        processed_audio = self.freq_filters[freq].process(processed_audio)
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý tần số {freq}: {str(e)}")
                        continue
            
            # Áp dụng âm lượng
            processed_audio *= volume
            
            # Kiểm tra dữ liệu sau khi xử lý
            if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
                logger.error("Dữ liệu âm thanh không hợp lệ sau khi xử lý")
                return None
                
            # Chuẩn hóa tín hiệu
            processed_audio = self.normalize_signal(processed_audio)
            
            # Kiểm tra kích thước dữ liệu
            if len(processed_audio) != len(self.audio_data):
                logger.error(f"Kích thước dữ liệu không khớp: {len(processed_audio)} vs {len(self.audio_data)}")
                return None
            
            # Chuyển đổi sang float32 cho sounddevice
            return processed_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý âm thanh: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def get_current_time(self):
        """Lấy thời gian hiện tại của bài hát"""
        return self.current_time
        
    def get_total_time(self):
        """Lấy tổng thời gian của bài hát"""
        return self.total_time 