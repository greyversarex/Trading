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
        """Load image from bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Remove noise and prepare image for line detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return blurred
    
    def detect_chart_area(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect the main chart area, excluding UI elements."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            return x, y, w, h
        
        return 0, 0, img.shape[1], img.shape[0]
    
    def extract_price_line_by_color(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract price line by detecting the most prominent non-background color."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        color_ranges = [
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255)),
            ((35, 100, 100), (85, 255, 255)),
            ((100, 100, 100), (130, 255, 255)),
            ((0, 0, 200), (180, 30, 255)),
        ]
        
        best_mask = None
        max_points = 0
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            points = cv2.countNonZero(mask)
            if points > max_points and points < (img.shape[0] * img.shape[1] * 0.5):
                max_points = points
                best_mask = mask
        
        return best_mask
    
    def extract_price_line_by_edges(self, img: np.ndarray) -> np.ndarray:
        """Extract price line using edge detection."""
        gray = self.preprocess_image(img)
        
        edges = cv2.Canny(gray, 30, 100)
        
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        return edges
    
    def mask_to_line(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to single continuous price line."""
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
        
        if valid_cols < width * 0.3:
            return np.array([])
        
        mask_valid = ~np.isnan(line)
        if np.sum(mask_valid) < 10:
            return np.array([])
        
        indices = np.arange(len(line))
        line[~mask_valid] = np.interp(indices[~mask_valid], indices[mask_valid], line[mask_valid])
        
        window_length = min(21, len(line) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            line = savgol_filter(line, window_length, 3)
        
        return line
    
    def extract_price_line(self, image_bytes: bytes) -> np.ndarray:
        """Main method to extract price line from screenshot."""
        img = self.load_image_from_bytes(image_bytes)
        
        if img is None:
            raise ValueError("Failed to load image")
        
        x, y, w, h = self.detect_chart_area(img)
        chart_region = img[y:y+h, x:x+w]
        
        color_mask = self.extract_price_line_by_color(chart_region)
        if color_mask is not None:
            line = self.mask_to_line(color_mask)
            if len(line) > 20:
                return line
        
        edge_mask = self.extract_price_line_by_edges(chart_region)
        line = self.mask_to_line(edge_mask)
        
        if len(line) < 20:
            gray = cv2.cvtColor(chart_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            line = self.mask_to_line(binary)
        
        return line
    
    def resample_line(self, line: np.ndarray, target_points: int = 100) -> np.ndarray:
        """Resample line to fixed number of points for comparison."""
        if len(line) == 0:
            return np.array([])
        
        x_original = np.linspace(0, 1, len(line))
        x_target = np.linspace(0, 1, target_points)
        return np.interp(x_target, x_original, line)
