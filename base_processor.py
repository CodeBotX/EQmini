#Xử lý âm thanh cơ bản
import numpy as np
from transforms import DiscreteFourierTransform, InverseDiscreteFourierTransform
import logging
import traceback

logger = logging.getLogger(__name__)

class BaseAudioProcessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.frame_size = 2048
        self.hop_size = 512
        
        # Tạo window function (Hanning window)
        self.window = np.hanning(self.frame_size)
        
        try:
            # Kiểm tra kích thước frame và hop size
            if self.frame_size <= 0 or self.hop_size <= 0:
                raise ValueError("Frame size và hop size phải lớn hơn 0")
            if self.hop_size > self.frame_size:
                raise ValueError("Hop size không được lớn hơn frame size")
                
            logger.info(f"Khởi tạo BaseAudioProcessor: sample_rate={sample_rate}, frame_size={self.frame_size}, hop_size={self.hop_size}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo BaseAudioProcessor: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    def frame_signal(self, signal):
        """Chia tín hiệu thành các frame chồng lấp"""
        try:
            if signal is None or len(signal) == 0:
                logger.error("Tín hiệu đầu vào không hợp lệ")
                return None
                
            frames = []
            for i in range(0, len(signal) - self.frame_size + 1, self.hop_size):
                frame = signal[i:i + self.frame_size]
                # Áp dụng window function
                frame = frame * self.window
                frames.append(frame)
                
            if not frames:
                logger.error("Không thể chia tín hiệu thành frame")
                return None
                
            return np.array(frames)
            
        except Exception as e:
            logger.error(f"Lỗi khi chia frame: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def overlap_add(self, frames):
        """Kết hợp các frame chồng lấp thành tín hiệu liên tục"""
        try:
            if frames is None or len(frames) == 0:
                logger.error("Không có frame để kết hợp")
                return None
                
            signal_length = (len(frames) - 1) * self.hop_size + self.frame_size
            signal = np.zeros(signal_length)
            
            for i, frame in enumerate(frames):
                start = i * self.hop_size
                # Áp dụng window function ngược
                frame = frame * self.window
                signal[start:start + self.frame_size] += frame
                
            return signal
            
        except Exception as e:
            logger.error(f"Lỗi khi kết hợp frame: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def process(self, signal):
        """Xử lý tín hiệu cơ bản sử dụng DFT và IDFT"""
        try:
            if signal is None or len(signal) == 0:
                logger.error("Tín hiệu đầu vào không hợp lệ")
                return None
                
            # Chia thành các frame
            frames = self.frame_signal(signal)
            if frames is None:
                return None
                
            # Xử lý từng frame
            processed_frames = []
            for i, frame in enumerate(frames):
                try:
                    # Chuyển sang miền tần số
                    dft = DiscreteFourierTransform(frame.tolist())
                    spectrum = dft.transform()
                    
                    # Xử lý phổ tần số ở đây
                    # Có thể thêm các xử lý phổ tần số tùy chỉnh
                    # Ví dụ: lọc nhiễu, điều chỉnh biên độ, v.v.
                    
                    # Chuyển về miền thời gian
                    idft = InverseDiscreteFourierTransform(spectrum)
                    processed_frame = np.array(idft.transform())
                    processed_frames.append(processed_frame)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý frame {i}: {str(e)}")
                    continue
                    
            if not processed_frames:
                logger.error("Không có frame nào được xử lý thành công")
                return None
                
            # Kết hợp các frame
            processed_signal = self.overlap_add(processed_frames)
            if processed_signal is None:
                return None
                
            # Kiểm tra kết quả
            if np.isnan(processed_signal).any() or np.isinf(processed_signal).any():
                logger.error("Kết quả xử lý chứa giá trị không hợp lệ")
                return None
                
            return processed_signal
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý tín hiệu: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def apply_filter(self, signal, filter_coeffs):
        """Áp dụng bộ lọc FIR cho tín hiệu"""
        try:
            if signal is None or len(signal) == 0:
                logger.error("Tín hiệu đầu vào không hợp lệ")
                return None
                
            if filter_coeffs is None or len(filter_coeffs) == 0:
                logger.error("Hệ số bộ lọc không hợp lệ")
                return None
                
            filtered_signal = np.convolve(signal, filter_coeffs, mode='same')
            
            # Kiểm tra kết quả
            if np.isnan(filtered_signal).any() or np.isinf(filtered_signal).any():
                logger.error("Kết quả lọc chứa giá trị không hợp lệ")
                return None
                
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng bộ lọc: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
    def normalize_signal(self, signal):
        """Chuẩn hóa tín hiệu về biên độ [-1, 1]"""
        try:
            if signal is None or len(signal) == 0:
                logger.error("Tín hiệu đầu vào không hợp lệ")
                return None
                
            max_amplitude = np.max(np.abs(signal))
            if max_amplitude > 0:
                normalized = signal / max_amplitude
                
                # Kiểm tra kết quả
                if np.isnan(normalized).any() or np.isinf(normalized).any():
                    logger.error("Kết quả chuẩn hóa chứa giá trị không hợp lệ")
                    return None
                    
                return normalized
            return signal
            
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn hóa tín hiệu: {str(e)}")
            logger.error(traceback.format_exc())
            return None


