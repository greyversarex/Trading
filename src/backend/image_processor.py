import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from typing import Tuple, Optional, List
import io
from PIL import Image


class ImageProcessor:
    """Extracts price line from chart screenshots using computer vision."""
    
    def __init__(self):
        self.min_line_length = 50
        self.max_line_gap = 10
    
    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return blurred
    
    def detect_chart_area(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        top_strip = gray[:int(h * 0.15), :]
        top_text_density = np.sum(cv2.Canny(top_strip, 50, 150) > 0) / top_strip.size
        
        right_strip = gray[:, int(w * 0.85):]
        right_text_density = np.sum(cv2.Canny(right_strip, 50, 150) > 0) / right_strip.size
        
        bottom_strip = gray[int(h * 0.85):, :]
        bottom_text_density = np.sum(cv2.Canny(bottom_strip, 50, 150) > 0) / bottom_strip.size
        
        y_start = int(h * 0.15) if top_text_density > 0.05 else 0
        x_end = int(w * 0.85) if right_text_density > 0.05 else w
        y_end = int(h * 0.85) if bottom_text_density > 0.05 else h
        x_start = 0
        
        margin = 5
        x_start = max(0, x_start + margin)
        y_start = max(0, y_start + margin)
        x_end = min(w, x_end - margin)
        y_end = min(h, y_end - margin)
        
        return x_start, y_start, x_end - x_start, y_end - y_start
    
    def detect_volume_area(self, img: np.ndarray) -> int:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        for y in range(int(h * 0.5), h):
            row = gray[y, :]
            dark_ratio = np.sum(row < 40) / len(row)
            if dark_ratio > 0.7:
                return y
        
        return h
    
    def extract_candlestick_prices(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        
        green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        candle_mask = cv2.bitwise_or(green_mask, red_mask)
        
        total_candle_pixels = cv2.countNonZero(candle_mask)
        if total_candle_pixels < (h * w * 0.005):
            return np.array([])
        
        kernel_v = np.ones((3, 1), np.uint8)
        candle_mask = cv2.morphologyEx(candle_mask, cv2.MORPH_CLOSE, kernel_v)
        
        col_density = np.sum(candle_mask > 0, axis=0).astype(float)
        
        min_candle_height = h * 0.01
        candle_cols = np.where(col_density > min_candle_height)[0]
        
        if len(candle_cols) < 10:
            return np.array([])
        
        line = np.full(w, np.nan)
        
        green_weight = 0.7
        red_weight = 0.3
        
        for x in range(w):
            g_col = green_mask[:, x]
            r_col = red_mask[:, x]
            g_positions = np.where(g_col > 0)[0]
            r_positions = np.where(r_col > 0)[0]
            
            if len(g_positions) > 2:
                close_y = np.min(g_positions)
                line[x] = h - close_y
            elif len(r_positions) > 2:
                close_y = np.max(r_positions)
                line[x] = h - close_y
        
        valid_mask = ~np.isnan(line)
        valid_count = np.sum(valid_mask)
        
        if valid_count < w * 0.1:
            return np.array([])
        
        indices = np.arange(len(line))
        line[~valid_mask] = np.interp(indices[~valid_mask], indices[valid_mask], line[valid_mask])
        
        kernel_size = max(3, w // 100)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, len(line) - 1)
        if kernel_size >= 3:
            line = ndimage.median_filter(line, size=kernel_size)
        
        return line
    
    def extract_price_line_by_color(self, img: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        
        color_ranges = [
            ("blue", (100, 100, 100), (130, 255, 255)),
            ("cyan", (85, 100, 100), (100, 255, 255)),
            ("yellow", (20, 100, 100), (35, 255, 255)),
            ("orange", (10, 100, 100), (20, 255, 255)),
            ("white", (0, 0, 200), (180, 30, 255)),
            ("magenta", (140, 100, 100), (170, 255, 255)),
        ]
        
        best_mask = None
        best_score = 0
        
        for name, lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            points = cv2.countNonZero(mask)
            area = h * w
            
            if points < area * 0.001 or points > area * 0.3:
                continue
            
            col_coverage = np.sum(np.any(mask > 0, axis=0))
            coverage_ratio = col_coverage / w
            
            if coverage_ratio < 0.3:
                continue
            
            col_heights = np.sum(mask > 0, axis=0).astype(float)
            active_cols = col_heights[col_heights > 0]
            if len(active_cols) > 0:
                avg_thickness = np.mean(active_cols)
                if avg_thickness > h * 0.3:
                    continue
            
            avg_t = avg_thickness if len(active_cols) > 0 else 0
            score = coverage_ratio * 100 - (avg_t / h * 50 if avg_t > 0 else 50)
            
            if score > best_score:
                best_score = score
                best_mask = mask
        
        return best_mask
    
    def extract_price_line_by_edges(self, img: np.ndarray) -> np.ndarray:
        gray = self.preprocess_image(img)
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges
    
    def mask_to_line(self, mask: np.ndarray, preserve_sharp: bool = False) -> np.ndarray:
        height, width = mask.shape
        line = np.zeros(width, dtype=np.float64)
        valid_cols = 0
        
        for x in range(width):
            col = mask[:, x]
            y_positions = np.where(col > 0)[0]
            
            if len(y_positions) > 0:
                line[x] = height - np.mean(y_positions)
                valid_cols += 1
            else:
                line[x] = np.nan
        
        if valid_cols < width * 0.2:
            return np.array([])
        
        mask_valid = ~np.isnan(line)
        if np.sum(mask_valid) < 10:
            return np.array([])
        
        indices = np.arange(len(line))
        line[~mask_valid] = np.interp(indices[~mask_valid], indices[mask_valid], line[mask_valid])
        
        if not preserve_sharp:
            window_length = min(11, len(line) - 1)
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                line = savgol_filter(line, window_length, 2)
        else:
            kernel_size = 3
            line = ndimage.median_filter(line, size=kernel_size)
        
        return line
    
    def extract_price_line(self, image_bytes: bytes) -> np.ndarray:
        img = self.load_image_from_bytes(image_bytes)
        
        if img is None:
            raise ValueError("Failed to load image")
        
        x, y, w, h = self.detect_chart_area(img)
        chart_region = img[y:y+h, x:x+w]
        
        vol_y = self.detect_volume_area(chart_region)
        if vol_y < chart_region.shape[0] * 0.95:
            chart_region = chart_region[:vol_y, :]
        
        candle_line = self.extract_candlestick_prices(chart_region)
        if len(candle_line) > 20:
            return candle_line
        
        color_mask = self.extract_price_line_by_color(chart_region)
        if color_mask is not None:
            has_sharp = self._detect_sharp_movements(color_mask)
            line = self.mask_to_line(color_mask, preserve_sharp=has_sharp)
            if len(line) > 20:
                return line
        
        edge_mask = self.extract_price_line_by_edges(chart_region)
        line = self.mask_to_line(edge_mask)
        
        if len(line) < 20:
            gray = cv2.cvtColor(chart_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            line = self.mask_to_line(binary)
        
        return line
    
    def _detect_sharp_movements(self, mask: np.ndarray) -> bool:
        height, width = mask.shape
        col_means = np.zeros(width)
        
        for x in range(width):
            col = mask[:, x]
            positions = np.where(col > 0)[0]
            if len(positions) > 0:
                col_means[x] = np.mean(positions)
            else:
                col_means[x] = np.nan
        
        valid = ~np.isnan(col_means)
        if np.sum(valid) < 10:
            return False
        
        valid_means = col_means[valid]
        diffs = np.abs(np.diff(valid_means))
        
        if len(diffs) == 0:
            return False
        
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        return max_diff > mean_diff * 5
    
    def resample_line(self, line: np.ndarray, target_points: int = 100) -> np.ndarray:
        if len(line) == 0:
            return np.array([])
        
        x_original = np.linspace(0, 1, len(line))
        x_target = np.linspace(0, 1, target_points)
        return np.interp(x_target, x_original, line)
