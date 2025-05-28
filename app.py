# app.py - Updated with Advanced False Positive Reduction
# Giao diện người dùng đồ họa hiện đại với tích hợp FP Reduction System

import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import json
from datetime import datetime

# Fix TensorFlow compatibility first
try:
    from tensorflow_compatibility_fix import fix_tensorflow_compatibility
    fix_tensorflow_compatibility()
except ImportError:
    print("⚠️ TensorFlow compatibility fix not found, continuing...")

# Import State-of-the-Art system
try:
    from sota import StateOfTheArtFractureDetector
except ImportError as e:
    print(f"⚠️ Could not import StateOfTheArtFractureDetector: {e}")
    StateOfTheArtFractureDetector = None

# Import ensemble system
try:
    from ensemble_prediction import MultiRegionEnsemble
except ImportError as e:
    print(f"⚠️ Could not import MultiRegionEnsemble: {e}")
    MultiRegionEnsemble = None

# Import Advanced FP Reduction System
try:
    from advanced_false_positive_reduction import (
        UncertaintyQuantification,
        ConfidenceCalibration,
        HardNegativeMining,
        AdaptiveThresholding,
        EnsembleUncertainty,
        AdvancedFalsePositiveReducer
    )
except ImportError as e:
    print(f"⚠️ Could not import FP Reduction components: {e}")
    # Create dummy classes
    class UncertaintyQuantification: pass
    class ConfidenceCalibration: pass
    class HardNegativeMining: pass
    class AdaptiveThresholding: pass
    class EnsembleUncertainty: pass
    class AdvancedFalsePositiveReducer: pass

class SOTAHoughTransform:
    """
    State-of-the-Art Hough Transform for Fracture Detection
    Based on latest research: Multi-parameter, YOLO-inspired, Dynamic thresholding
    """
    def __init__(self):
        self.line_orientations = np.arange(0, 180, 15)  # 12 orientations
        self.debug_mode = True
        
    def advanced_edge_detection(self, image):
        """Advanced multi-scale edge detection"""
        edges_list = []
        
        # 1. Multi-scale Canny edge detection
        canny_params = [
            (15, 45),   # Very sensitive
            (25, 75),   # High sensitivity  
            (35, 105),  # Medium sensitivity
            (50, 150),  # Standard
            (70, 200),  # Conservative
        ]
        
        for low, high in canny_params:
            edges = cv2.Canny(image, low, high, apertureSize=3)
            edges_list.append(edges.astype(np.float32) / 255.0)
        
        # 2. Sobel edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        sobel_norm = sobel_combined / (np.max(sobel_combined) + 1e-8)
        edges_list.append(sobel_norm)
        
        # 3. Laplacian edge detection
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_norm = np.abs(laplacian) / (np.max(np.abs(laplacian)) + 1e-8)
        edges_list.append(laplacian_norm)
        
        # Weighted combination
        weights = [0.2, 0.2, 0.15, 0.1, 0.05, 0.15, 0.15]
        combined = np.zeros_like(edges_list[0])
        for edge_map, weight in zip(edges_list, weights):
            combined += edge_map * weight
        
        # Non-maximum suppression
        combined = self._non_maximum_suppression(combined, image)
        
        return (combined * 255).astype(np.uint8)
    
    def _non_maximum_suppression(self, edge_map, original_image):
        """Non-maximum suppression for edge thinning"""
        # Compute gradient direction
        Ix = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)
        
        direction = np.arctan2(Iy, Ix) * 180 / np.pi
        direction[direction < 0] += 180
        
        suppressed = np.zeros_like(edge_map)
        
        for i in range(1, edge_map.shape[0] - 1):
            for j in range(1, edge_map.shape[1] - 1):
                angle = direction[i, j]
                
                # Determine neighbors based on gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [edge_map[i, j-1], edge_map[i, j+1]]
                elif 22.5 <= angle < 67.5:
                    neighbors = [edge_map[i-1, j-1], edge_map[i+1, j+1]]
                elif 67.5 <= angle < 112.5:
                    neighbors = [edge_map[i-1, j], edge_map[i+1, j]]
                else:
                    neighbors = [edge_map[i-1, j+1], edge_map[i+1, j-1]]
                
                if edge_map[i, j] >= max(neighbors):
                    suppressed[i, j] = edge_map[i, j]
        
        return suppressed
    
    def sota_hough_line_detection(self, edges):
        """SOTA Hough Transform with multiple strategies"""
        all_lines = []
        
        # 1. Multi-parameter Standard Hough Transform
        hough_configs = [
            {'rho': 1, 'theta': np.pi/180, 'threshold': 25, 'min_len': 8, 'max_gap': 3},
            {'rho': 1, 'theta': np.pi/180, 'threshold': 35, 'min_len': 15, 'max_gap': 5},
            {'rho': 1, 'theta': np.pi/180, 'threshold': 45, 'min_len': 20, 'max_gap': 8},
            {'rho': 2, 'theta': np.pi/180, 'threshold': 30, 'min_len': 12, 'max_gap': 4},
            {'rho': 0.5, 'theta': np.pi/360, 'threshold': 20, 'min_len': 10, 'max_gap': 6},
        ]
        
        for config in hough_configs:
            lines = cv2.HoughLinesP(edges, **config)
            if lines is not None:
                all_lines.extend(lines)
        
        # 2. Directional Hough for specific fracture angles
        fracture_angles = [30, 45, 60, 90, 120, 135, 150]
        for angle_deg in fracture_angles:
            filtered_edges = self._apply_directional_filter(edges, angle_deg)
            dir_lines = cv2.HoughLinesP(filtered_edges, 1, np.pi/180, 20, 
                                      minLineLength=8, maxLineGap=4)
            if dir_lines is not None:
                all_lines.extend(dir_lines)
        
        # 3. Adaptive threshold Hough
        adaptive_lines = self._adaptive_threshold_hough(edges)
        if adaptive_lines is not None:
            all_lines.extend(adaptive_lines)
        
        # 4. Multi-scale Hough
        multiscale_lines = self._multiscale_hough(edges)
        if multiscale_lines is not None:
            all_lines.extend(multiscale_lines)
        
        return np.array(all_lines) if all_lines else None
    
    def _apply_directional_filter(self, edges, angle_deg):
        """Apply directional filter for specific angles"""
        angle_rad = np.deg2rad(angle_deg)
        
        # Create directional kernel
        size = 15
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = j - center, i - center
                dist_to_line = abs(x * np.sin(angle_rad) - y * np.cos(angle_rad))
                if dist_to_line <= 2.0:
                    kernel[i, j] = 1.0
        
        if np.sum(kernel) > 0:
            kernel /= np.sum(kernel)
            filtered = cv2.filter2D(edges, -1, kernel)
            return filtered
        
        return edges
    
    def _adaptive_threshold_hough(self, edges):
        """Adaptive threshold based on image content"""
        # Calculate optimal threshold based on edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 0.1:  # High edge density
            threshold = 40
            min_len = 15
        elif edge_density > 0.05:  # Medium edge density
            threshold = 30
            min_len = 12
        else:  # Low edge density
            threshold = 20
            min_len = 8
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 
                              minLineLength=min_len, maxLineGap=5)
        return lines
    
    def _multiscale_hough(self, edges):
        """Multi-scale Hough Transform"""
        all_lines = []
        
        # Different scales
        scales = [0.5, 1.0, 1.5, 2.0]
        
        for scale in scales:
            if scale != 1.0:
                h, w = edges.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_edges = cv2.resize(edges, (new_w, new_h))
            else:
                scaled_edges = edges
            
            # Scale-adapted parameters
            threshold = max(15, int(25 / scale))
            min_len = max(5, int(10 / scale))
            max_gap = max(2, int(4 / scale))
            
            lines = cv2.HoughLinesP(scaled_edges, 1, np.pi/180, threshold,
                                  minLineLength=min_len, maxLineGap=max_gap)
            
            if lines is not None:
                # Scale lines back to original size
                if scale != 1.0:
                    lines = lines / scale
                all_lines.extend(lines)
        
        return all_lines if all_lines else None
    
    def analyze_fracture_patterns(self, lines, image_shape):
        """Advanced fracture pattern analysis"""
        if lines is None or len(lines) == 0:
            return {
                'fracture_score': 0.0,
                'confidence': 0.0,
                'pattern_type': 'none',
                'line_count': 0,
                'analysis_details': {}
            }
        
        h, w = image_shape[:2]
        
        # Flatten lines if needed
        if len(lines.shape) == 3:
            lines = lines.reshape(-1, 4)
        
        # Extract line features
        line_features = self._extract_line_features(lines, (h, w))
        
        # Multiple analysis modules
        geometric_analysis = self._geometric_analysis(line_features)
        medical_analysis = self._medical_pattern_analysis(line_features, (h, w))
        spatial_analysis = self._spatial_distribution_analysis(line_features, (h, w))
        
        # Combine analyses
        final_score = self._weighted_score_combination([
            geometric_analysis['score'],
            medical_analysis['score'],
            spatial_analysis['score']
        ], weights=[0.3, 0.5, 0.2])
        
        # Calculate confidence based on pattern consistency
        confidence = np.mean([
            geometric_analysis.get('consistency', 0.5),
            medical_analysis.get('consistency', 0.5),
            spatial_analysis.get('consistency', 0.5)
        ])
        
        # Determine pattern type
        pattern_type = self._determine_pattern_type(line_features)
        
        return {
            'fracture_score': min(final_score, 1.0),
            'confidence': confidence,
            'pattern_type': pattern_type,
            'line_count': len(lines),
            'analysis_details': {
                'geometric': geometric_analysis,
                'medical': medical_analysis,
                'spatial': spatial_analysis
            }
        }
    
    def _extract_line_features(self, lines, image_shape):
        """Extract comprehensive line features"""
        h, w = image_shape
        features = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Basic geometric features
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle < 0:
                angle += 180
            
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Advanced features
            center_x, center_y = w // 2, h // 2
            dist_to_center = np.sqrt((mid_x - center_x)**2 + (mid_y - center_y)**2)
            
            # Orientation analysis
            angle_to_horizontal = min(abs(angle), abs(angle - 180))
            angle_to_vertical = min(abs(angle - 90), abs(angle - 270))
            
            # Medical relevance
            is_transverse = 80 <= angle_to_vertical <= 100
            is_oblique = 30 <= angle_to_horizontal <= 60
            is_longitudinal = angle_to_horizontal <= 20 or angle_to_horizontal >= 160
            
            features.append({
                'coords': (x1, y1, x2, y2),
                'length': length,
                'angle': angle,
                'midpoint': (mid_x, mid_y),
                'dist_to_center': dist_to_center,
                'angle_to_horizontal': angle_to_horizontal,
                'angle_to_vertical': angle_to_vertical,
                'is_transverse': is_transverse,
                'is_oblique': is_oblique,
                'is_longitudinal': is_longitudinal
            })
        
        return features
    
    def _geometric_analysis(self, line_features):
        """Analyze geometric properties"""
        if not line_features:
            return {'score': 0.0, 'consistency': 0.0}
        
        scores = []
        
        # Length distribution analysis
        lengths = [f['length'] for f in line_features]
        length_std = np.std(lengths)
        length_mean = np.mean(lengths)
        
        # Short lines indicate fragmentation
        short_lines = [f for f in line_features if f['length'] < 25]
        if len(short_lines) > 3:
            scores.append(min(len(short_lines) / 10, 1.0))
        
        # Length variability
        if length_mean > 0:
            length_cv = length_std / length_mean
            if length_cv > 0.5:
                scores.append(min(length_cv, 1.0))
        
        # Angle diversity
        angles = [f['angle'] for f in line_features]
        angle_std = np.std(angles)
        if 15 < angle_std < 60:  # Moderate diversity
            scores.append(0.7)
        elif angle_std >= 60:  # High diversity
            scores.append(0.9)
        
        final_score = np.mean(scores) if scores else 0.0
        consistency = 1 - np.std(scores) / (np.mean(scores) + 1e-8) if len(scores) > 1 else 1.0
        
        return {
            'score': final_score,
            'consistency': max(0, consistency)
        }
    
    def _medical_pattern_analysis(self, line_features, image_shape):
        """Medical knowledge-based analysis"""
        h, w = image_shape
        scores = []
        
        # Transverse fracture detection (most common)
        transverse_lines = [f for f in line_features if f['is_transverse']]
        if transverse_lines:
            scores.append(min(len(transverse_lines) / 3, 1.0) * 0.9)
        
        # Oblique fracture detection
        oblique_lines = [f for f in line_features if f['is_oblique']]
        if oblique_lines:
            scores.append(min(len(oblique_lines) / 4, 1.0) * 0.8)
        
        # Longitudinal fracture detection (less common)
        longitudinal_lines = [f for f in line_features if f['is_longitudinal']]
        if longitudinal_lines:
            scores.append(min(len(longitudinal_lines) / 2, 1.0) * 0.6)
        
        # Comminuted fracture (multiple fragments)
        if len(line_features) > 8:
            small_fragments = [f for f in line_features if f['length'] < 20]
            if len(small_fragments) > 5:
                scores.append(min(len(small_fragments) / 12, 1.0) * 0.95)
        
        # Central location preference
        central_lines = [f for f in line_features 
                        if f['dist_to_center'] < min(w, h) * 0.3]
        if central_lines:
            scores.append(min(len(central_lines) / len(line_features), 1.0) * 0.7)
        
        final_score = np.mean(scores) if scores else 0.0
        consistency = 1.0 if scores else 0.0
        
        return {
            'score': final_score,
            'consistency': consistency
        }
    
    def _spatial_distribution_analysis(self, line_features, image_shape):
        """Analyze spatial distribution"""
        if not line_features:
            return {'score': 0.0, 'consistency': 0.0}
        
        h, w = image_shape
        scores = []
        
        # Clustering analysis
        positions = [f['midpoint'] for f in line_features]
        if len(positions) >= 3:
            cluster_score = self._analyze_spatial_clustering(positions)
            if cluster_score > 0.4:
                scores.append(cluster_score)
        
        # Density analysis
        grid_size = 8
        density_grid = np.zeros((grid_size, grid_size))
        
        for x, y in positions:
            grid_x = min(int(x / (w / grid_size)), grid_size - 1)
            grid_y = min(int(y / (h / grid_size)), grid_size - 1)
            density_grid[grid_y, grid_x] += 1
        
        max_density = np.max(density_grid)
        if max_density > 2:
            high_density_cells = np.sum(density_grid >= max_density * 0.7)
            if high_density_cells <= 3:
                scores.append(min(max_density / 6, 1.0))
        
        final_score = np.mean(scores) if scores else 0.0
        consistency = 1.0 if scores else 0.0
        
        return {
            'score': final_score,
            'consistency': consistency
        }
    
    def _analyze_spatial_clustering(self, positions):
        """Analyze spatial clustering of positions"""
        if len(positions) < 3:
            return 0.0
        
        positions = np.array(positions)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(positions)
        
        # Clustering score based on distance distribution
        median_dist = np.median(distances)
        close_pairs = np.sum(distances < median_dist * 0.5)
        total_pairs = len(distances)
        
        return close_pairs / total_pairs
    
    def _weighted_score_combination(self, scores, weights):
        """Combine scores with weights"""
        if not scores:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        combined = np.sum(np.array(scores) * weights)
        
        # Apply enhancement function
        enhanced = 1 / (1 + np.exp(-4 * (combined - 0.5)))
        
        return min(enhanced, 1.0)
    
    def _determine_pattern_type(self, line_features):
        """Determine fracture pattern type"""
        if not line_features:
            return 'none'
        
        transverse_count = sum(1 for f in line_features if f['is_transverse'])
        oblique_count = sum(1 for f in line_features if f['is_oblique'])
        longitudinal_count = sum(1 for f in line_features if f['is_longitudinal'])
        
        if len(line_features) > 8:
            return 'comminuted'
        elif transverse_count > oblique_count and transverse_count > longitudinal_count:
            return 'transverse'
        elif oblique_count > transverse_count and oblique_count > longitudinal_count:
            return 'oblique'
        elif longitudinal_count > 0:
            return 'longitudinal'
        else:
            return 'complex'
    
    def predict_fracture(self, image):
        """Main prediction method for SOTA Hough"""
        try:
            # Advanced edge detection
            edges = self.advanced_edge_detection(image)
            
            # SOTA Hough line detection
            lines = self.sota_hough_line_detection(edges)
            
            # Pattern analysis
            analysis = self.analyze_fracture_patterns(lines, image.shape)
            
            # Create visualization heatmap
            heatmap = self._create_hough_heatmap(image, lines, analysis)
            
            return analysis['fracture_score'], heatmap, analysis
            
        except Exception as e:
            print(f"SOTA Hough prediction error: {e}")
            return 0.0, np.zeros_like(image, dtype=np.float32), {
                'fracture_score': 0.0,
                'confidence': 0.0,
                'pattern_type': 'error',
                'line_count': 0,
                'analysis_details': {}
            }
    
    def _create_hough_heatmap(self, image, lines, analysis):
        """Create advanced heatmap from Hough analysis"""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        if lines is None or len(lines) == 0:
            return heatmap
        
        # Flatten lines if needed
        if len(lines.shape) == 3:
            lines = lines.reshape(-1, 4)
        
        fracture_score = analysis['fracture_score']
        confidence = analysis['confidence']
        
        # Draw lines with intensity based on medical relevance
        for line in lines:
            x1, y1, x2, y2 = [int(coord) for coord in line]
            x1, y1 = max(0, min(w-1, x1)), max(0, min(h-1, y1))
            x2, y2 = max(0, min(w-1, x2)), max(0, min(h-1, y2))
            
            # Calculate line properties
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle < 0:
                angle += 180
            
            # Medical relevance weighting
            angle_to_vertical = min(abs(angle - 90), abs(angle - 270))
            if 80 <= angle_to_vertical <= 100:  # Transverse
                intensity = 0.9
                thickness = 8
            elif 30 <= angle <= 60 or 120 <= angle <= 150:  # Oblique
                intensity = 0.7
                thickness = 6
            else:  # Other orientations
                intensity = 0.5
                thickness = 4
            
            # Length-based adjustment
            if length < 15:
                intensity *= 1.2  # Small fragments more suspicious
                thickness += 2
            elif length > 50:
                intensity *= 0.8  # Very long lines less likely fractures
            
            # Apply confidence modulation
            intensity *= confidence
            
            cv2.line(heatmap, (x1, y1), (x2, y2), intensity, thickness)
        
        # Apply Gaussian smoothing
        sigma = max(2, min(h, w) // 150)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
        
        # Enhance based on overall fracture score
        heatmap *= (0.3 + 0.7 * fracture_score)
        
        # Final normalization
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return np.clip(heatmap, 0, 1)

class AdvancedFractureDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("🚀 Advanced AI Fracture Detection with FP Reduction")
        self.master.geometry("1600x1000")
        self.master.minsize(1500, 900)
        
        # Enhanced color scheme
        self.colors = {
            'primary': '#1E3A8A',      # Deep Blue
            'secondary': '#7C3AED',    # Purple
            'accent': '#F59E0B',       # Amber
            'success': '#DC2626',      # Red for fracture detection
            'sota': '#10B981',         # Emerald for SOTA features
            'fp_reduction': '#8B5CF6', # Purple for FP Reduction
            'confidence': '#06B6D4',   # Cyan for confidence
            'background': '#F8FAFC',   # Light gray
            'surface': '#FFFFFF',      # White
            'text_primary': '#1F2937', # Dark gray
            'text_secondary': '#6B7280', # Medium gray
            'border': '#E5E7EB',       # Light border
            'shadow': '#D1D5DB'        # Shadow color
        }
        
        self.master.configure(bg=self.colors['background'])
        
        # Setup modern styling
        self.setup_styles()
        
        # Variables
        self.current_image_path = None
        self.prediction_result = None
        self.sota_detector = None
        self.sota_hough = None
        self.ensemble = None
        self.fp_reducer = None
        self.model_loaded = False
        self.ensemble_loaded = False
        self.hough_initialized = False
        self.fp_reduction_enabled = tk.BooleanVar(value=True)
        self.load_model_thread = None
        self.selected_model = tk.StringVar(value="resnet50v2")
        self.progress_var = tk.DoubleVar()
        self.current_mode = tk.StringVar(value="sota_fp")  # "sota", "ensemble", "sota_fp"
        
        # FP Reduction settings
        self.target_specificity = tk.DoubleVar(value=0.95)
        self.confidence_threshold = tk.DoubleVar(value=0.8)
        self.mc_samples = tk.IntVar(value=30)
        
        # Create modern interface
        self.create_modern_interface()
        
        # Load default SOTA model (SOTA Hough will be initialized manually)
        self.load_sota_model_async()

    def setup_styles(self):
        """Setup modern styling with FP Reduction theme"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # FP Reduction Button style
        self.style.configure('FPReduction.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 12),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('FPReduction.TButton',
                      background=[('active', self.colors['fp_reduction']),
                                ('pressed', '#7C3AED'),
                                ('!active', self.colors['fp_reduction'])],
                      foreground=[('active', 'white'),
                                ('pressed', 'white'),
                                ('!active', 'white')])
        
        # Confidence Button style
        self.style.configure('Confidence.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(15, 10),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('Confidence.TButton',
                      background=[('active', self.colors['confidence']),
                                ('pressed', '#0891B2'),
                                ('!active', self.colors['confidence'])],
                      foreground=[('active', 'white'),
                                ('pressed', 'white'),
                                ('!active', 'white')])
        
        # SOTA Button style
        self.style.configure('SOTA.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 12),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('SOTA.TButton',
                      background=[('active', self.colors['sota']),
                                ('pressed', '#059669'),
                                ('!active', self.colors['sota'])],
                      foreground=[('active', 'white'),
                                ('pressed', 'white'),
                                ('!active', 'white')])
        
        # Primary button
        self.style.configure('Primary.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 12),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('Primary.TButton',
                      background=[('active', self.colors['primary']),
                                ('pressed', '#1E40AF'),
                                ('!active', self.colors['primary'])],
                      foreground=[('active', 'white'),
                                ('pressed', 'white'),
                                ('!active', 'white')])
        
        # Accent button
        self.style.configure('Accent.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(15, 10),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('Accent.TButton',
                      background=[('active', self.colors['accent']),
                                ('pressed', '#D97706'),
                                ('!active', self.colors['accent'])],
                      foreground=[('active', 'white'),
                                ('pressed', 'white'),
                                ('!active', 'white')])

    def create_modern_interface(self):
        """Create Advanced interface with FP Reduction"""
        # Main container with padding
        main_container = tk.Frame(self.master, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_container)
        
        # Content area
        content_frame = tk.Frame(main_container, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel (controls and info) - WITH SCROLLBARS
        self.create_scrollable_left_panel(content_frame)
        
        # Right panel (image display)
        right_panel = tk.Frame(content_frame, bg=self.colors['background'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create panels content
        self.create_image_panel(right_panel)
        
        # Footer with status
        self.create_footer(main_container)

    def create_header(self, parent):
        """Create Advanced header with FP Reduction"""
        header_frame = tk.Frame(parent, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title with Advanced styling
        title_frame = tk.Frame(header_frame, bg=self.colors['background'])
        title_frame.pack(fill=tk.X)
        
        # Advanced title
        title_label = tk.Label(title_frame, 
                              text="🚀 Advanced AI Fracture Detection + FP Reduction",
                              font=('Segoe UI', 24, 'bold'),
                              bg=self.colors['background'],
                              fg=self.colors['primary'])
        title_label.pack(side=tk.LEFT)
        
        # Version badges
        badges_frame = tk.Frame(title_frame, bg=self.colors['background'])
        badges_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        # SOTA badge
        sota_badge = tk.Frame(badges_frame, bg=self.colors['sota'], relief='flat')
        sota_badge.pack(side=tk.LEFT, padx=(0, 5))
        
        sota_label = tk.Label(sota_badge, 
                             text=" SOTA v4.0 ",
                             font=('Segoe UI', 9, 'bold'),
                             bg=self.colors['sota'],
                             fg='white')
        sota_label.pack(padx=6, pady=3)
        
        # FP Reduction badge
        fp_badge = tk.Frame(badges_frame, bg=self.colors['fp_reduction'], relief='flat')
        fp_badge.pack(side=tk.LEFT)
        
        fp_label = tk.Label(fp_badge, 
                           text=" FP-REDUCE ",
                           font=('Segoe UI', 9, 'bold'),
                           bg=self.colors['fp_reduction'],
                           fg='white')
        fp_label.pack(padx=6, pady=3)
        
        # Enhanced subtitle
        subtitle_label = tk.Label(header_frame,
                                text="🔬 Uncertainty Quantification • Confidence Calibration • Hard Negative Mining • SOTA Hough Transform",
                                font=('Segoe UI', 11),
                                bg=self.colors['background'],
                                fg=self.colors['text_secondary'])
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))

    def create_scrollable_left_panel(self, parent):
        """Create scrollable left panel with both scrollbars"""
        # Container for the left panel with fixed width
        left_panel_container = tk.Frame(parent, bg=self.colors['background'], width=400)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel_container.pack_propagate(False)
        
        # Create Canvas and Scrollbars (both vertical and horizontal)
        canvas = tk.Canvas(left_panel_container, 
                          bg=self.colors['background'],
                          highlightthickness=0,
                          width=380)
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(left_panel_container, orient="horizontal", command=canvas.xview)
        
        self.scrollable_frame = tk.Frame(canvas, bg=self.colors['background'])
        
        # Configure scrolling
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack canvas and scrollbars
        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        left_panel_container.grid_rowconfigure(0, weight=1)
        left_panel_container.grid_columnconfigure(0, weight=1)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_horizontal_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Shift-MouseWheel>", _on_horizontal_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Shift-MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Create control panel content
        self.create_control_panel(self.scrollable_frame)

    def create_control_panel(self, parent):
        """Create Advanced control panel with FP Reduction"""
        # Mode Selection Card with FP Reduction
        mode_card = self.create_card(parent, "🎯 Advanced Analysis Mode")
        
        mode_frame = tk.Frame(mode_card, bg=self.colors['surface'])
        mode_frame.pack(fill=tk.X, pady=10)
        
        sota_fp_rb = tk.Radiobutton(mode_frame,
                                   text="🚀 SOTA + FP Reduction (Recommended)",
                                   variable=self.current_mode,
                                   value="sota_fp",
                                   font=('Segoe UI', 11, 'bold'),
                                   bg=self.colors['surface'],
                                   fg=self.colors['fp_reduction'],
                                   selectcolor=self.colors['fp_reduction'],
                                   activebackground=self.colors['surface'],
                                   command=self.on_mode_change)
        sota_fp_rb.pack(anchor=tk.W, pady=2)
        
        sota_rb = tk.Radiobutton(mode_frame,
                                text="🔬 SOTA Analysis Only",
                                variable=self.current_mode,
                                value="sota",
                                font=('Segoe UI', 10),
                                bg=self.colors['surface'],
                                fg=self.colors['sota'],
                                selectcolor=self.colors['sota'],
                                activebackground=self.colors['surface'],
                                command=self.on_mode_change)
        sota_rb.pack(anchor=tk.W, pady=2)
        
        ensemble_rb = tk.Radiobutton(mode_frame,
                                   text="🔗 Multi-Region Ensemble",
                                   variable=self.current_mode,
                                   value="ensemble",
                                   font=('Segoe UI', 10),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text_primary'],
                                   selectcolor=self.colors['accent'],
                                   activebackground=self.colors['surface'],
                                   command=self.on_mode_change)
        ensemble_rb.pack(anchor=tk.W, pady=2)
        
        # FP Reduction Configuration Card
        self.fp_config_card = self.create_card(parent, "🛡️ False Positive Reduction Settings")
        
        # FP Reduction toggle
        fp_toggle_frame = tk.Frame(self.fp_config_card, bg=self.colors['surface'])
        fp_toggle_frame.pack(fill=tk.X, pady=5)
        
        fp_check = tk.Checkbutton(fp_toggle_frame,
                                 text="Enable Advanced FP Reduction",
                                 variable=self.fp_reduction_enabled,
                                 font=('Segoe UI', 11, 'bold'),
                                 bg=self.colors['surface'],
                                 fg=self.colors['fp_reduction'],
                                 selectcolor=self.colors['fp_reduction'],
                                 activebackground=self.colors['surface'],
                                 command=self.on_fp_toggle)
        fp_check.pack(anchor=tk.W)
        
        # Target Specificity
        spec_frame = tk.Frame(self.fp_config_card, bg=self.colors['surface'])
        spec_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(spec_frame, text="Target Specificity:",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['surface'],
                fg=self.colors['text_primary']).pack(anchor=tk.W)
        
        spec_scale = tk.Scale(spec_frame, from_=0.8, to=0.99, resolution=0.01,
                             orient=tk.HORIZONTAL, variable=self.target_specificity,
                             bg=self.colors['surface'], fg=self.colors['text_primary'])
        spec_scale.pack(fill=tk.X, padx=10)
        
        # Model Configuration Card
        model_card = self.create_card(parent, "🔬 Model Configuration")
        
        # Model selection buttons
        btn_frame = tk.Frame(model_card, bg=self.colors['surface'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.sota_model_btn = ttk.Button(btn_frame, 
                                        text="🧠 Select SOTA Model",
                                        style='SOTA.TButton',
                                        command=self.show_sota_model_menu)
        self.sota_model_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.ensemble_btn = ttk.Button(btn_frame,
                                     text="🔗 Initialize Ensemble",
                                     style='Accent.TButton',
                                     command=self.initialize_ensemble)
        self.ensemble_btn.pack(fill=tk.X, pady=(5, 0))
        
        self.fp_init_btn = ttk.Button(btn_frame,
                                     text="🛡️ Initialize FP Reducer",
                                     style='FPReduction.TButton',
                                     command=self.initialize_fp_reducer)
        self.fp_init_btn.pack(fill=tk.X, pady=(5, 0))
        
        self.hough_init_btn = ttk.Button(btn_frame,
                                       text="📐 Initialize SOTA Hough",
                                       style='Confidence.TButton',
                                       command=self.initialize_sota_hough)
        self.hough_init_btn.pack(fill=tk.X, pady=(5, 0))
        
        # Model info display
        self.model_info_frame = tk.Frame(model_card, bg=self.colors['surface'])
        self.model_info_frame.pack(fill=tk.X, pady=5)
        
        self.model_status_label = tk.Label(self.model_info_frame,
                                          text="Loading SOTA model...",
                                          font=('Segoe UI', 10),
                                          bg=self.colors['surface'],
                                          fg=self.colors['text_secondary'])
        self.model_status_label.pack(anchor=tk.W)
        
        self.fp_status_label = tk.Label(self.model_info_frame,
                                       text="FP Reducer not initialized",
                                       font=('Segoe UI', 10),
                                       bg=self.colors['surface'],
                                       fg=self.colors['text_secondary'])
        self.fp_status_label.pack(anchor=tk.W)
        
        self.hough_status_label = tk.Label(self.model_info_frame,
                                         text="SOTA Hough not initialized",
                                         font=('Segoe UI', 10),
                                         bg=self.colors['surface'],
                                         fg=self.colors['text_secondary'])
        self.hough_status_label.pack(anchor=tk.W)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(model_card,
                                          mode='indeterminate',
                                          length=360)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Image Selection Card
        image_card = self.create_card(parent, "📁 Image Selection")
        
        upload_btn = ttk.Button(image_card,
                               text="🖼️ Choose X-ray Image",
                               style='Primary.TButton',
                               command=self.browse_image)
        upload_btn.pack(fill=tk.X, pady=10)
        
        # Image info
        self.image_info_label = tk.Label(image_card,
                                       text="No image selected",
                                       font=('Segoe UI', 10),
                                       bg=self.colors['surface'],
                                       fg=self.colors['text_secondary'],
                                       wraplength=340)
        self.image_info_label.pack(anchor=tk.W, pady=5)
        
        # Analysis Methods Card
        self.method_card = self.create_card(parent, "⚙️ Analysis Method")
        
        self.method_var = tk.StringVar(value="sota_combined_fp")
        
        # SOTA methods
        self.sota_methods_frame = tk.Frame(self.method_card, bg=self.colors['surface'])
        
        sota_methods = [
            ("🧠 SOTA CNN + FP Reduction", "sota_cnn_fp"),
            ("📐 SOTA Hough + FP Reduction", "sota_hough_fp"),
            ("🚀 SOTA Combined + FP Reduction", "sota_combined_fp"),
            ("🔬 SOTA Hough Analysis Only", "sota_hough_only")
        ]
        
        for text, value in sota_methods:
            rb = tk.Radiobutton(self.sota_methods_frame,
                              text=text,
                              variable=self.method_var,
                              value=value,
                              font=('Segoe UI', 10),
                              bg=self.colors['surface'],
                              fg=self.colors['text_primary'],
                              selectcolor=self.colors['fp_reduction'],
                              activebackground=self.colors['surface'])
            rb.pack(anchor=tk.W, pady=2)
        
        # Analysis Button
        self.predict_btn = ttk.Button(self.method_card,
                                     text="🚀 Run Advanced Analysis",
                                     style='FPReduction.TButton',
                                     command=self.predict_image,
                                     state=tk.DISABLED)
        self.predict_btn.pack(fill=tk.X, pady=(15, 5))
        
        # Update UI based on initial mode
        self.on_mode_change()
        
        # Advanced Results Card
        self.create_advanced_results_card(parent)

    def create_advanced_results_card(self, parent):
        """Create Advanced results display card with FP Reduction metrics"""
        results_card = self.create_card(parent, "📊 Advanced Analysis Results")
        
        # Result display area
        self.result_display = tk.Frame(results_card, bg=self.colors['surface'])
        self.result_display.pack(fill=tk.X, pady=10)
        
        # Prediction result with confidence
        self.prediction_frame = tk.Frame(self.result_display, bg=self.colors['surface'])
        self.prediction_frame.pack(fill=tk.X, pady=5)
        
        pred_label = tk.Label(self.prediction_frame,
                            text="Advanced Diagnosis:",
                            font=('Segoe UI', 10, 'bold'),
                            bg=self.colors['surface'],
                            fg=self.colors['text_primary'])
        pred_label.pack(anchor=tk.W)
        
        self.prediction_result_label = tk.Label(self.prediction_frame,
                                              text="Awaiting advanced analysis...",
                                              font=('Segoe UI', 12, 'bold'),
                                              bg=self.colors['surface'],
                                              fg=self.colors['text_secondary'])
        self.prediction_result_label.pack(anchor=tk.W, padx=10)
        
        # Confidence metrics
        confidence_frame = tk.Frame(self.result_display, bg=self.colors['surface'])
        confidence_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(confidence_frame,
                text="Confidence Metrics:",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['surface'],
                fg=self.colors['text_primary']).pack(anchor=tk.W)
        
        # Main confidence
        conf_main_frame = tk.Frame(confidence_frame, bg=self.colors['surface'])
        conf_main_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(conf_main_frame, text="Prediction Confidence:",
               font=('Segoe UI', 9),
               bg=self.colors['surface'],
               fg=self.colors['text_primary']).pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(conf_main_frame, text="-",
                                       font=('Segoe UI', 9, 'bold'),
                                       bg=self.colors['surface'],
                                       fg=self.colors['confidence'])
        self.confidence_label.pack(side=tk.RIGHT)
        
        # Uncertainty
        uncertainty_frame = tk.Frame(confidence_frame, bg=self.colors['surface'])
        uncertainty_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(uncertainty_frame, text="Uncertainty:",
               font=('Segoe UI', 9),
               bg=self.colors['surface'],
               fg=self.colors['text_primary']).pack(side=tk.LEFT)
        
        self.uncertainty_label = tk.Label(uncertainty_frame, text="-",
                                        font=('Segoe UI', 9, 'bold'),
                                        bg=self.colors['surface'],
                                        fg=self.colors['fp_reduction'])
        self.uncertainty_label.pack(side=tk.RIGHT)
        
        # Coverage status
        coverage_frame = tk.Frame(confidence_frame, bg=self.colors['surface'])
        coverage_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(coverage_frame, text="Analysis Status:",
               font=('Segoe UI', 9),
               bg=self.colors['surface'],
               fg=self.colors['text_primary']).pack(side=tk.LEFT)
        
        self.coverage_label = tk.Label(coverage_frame, text="-",
                                     font=('Segoe UI', 9, 'bold'),
                                     bg=self.colors['surface'],
                                     fg=self.colors['sota'])
        self.coverage_label.pack(side=tk.RIGHT)
        
        # Action buttons
        action_frame = tk.Frame(results_card, bg=self.colors['surface'])
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        save_btn = ttk.Button(action_frame,
                             text="💾 Save Results",
                             style='Accent.TButton',
                             command=self.save_result)
        save_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        confidence_btn = ttk.Button(action_frame,
                                   text="📊 Confidence Analysis",
                                   style='Confidence.TButton',
                                   command=self.show_confidence_analysis)
        confidence_btn.pack(side=tk.LEFT, padx=(0, 5))

    def create_card(self, parent, title):
        """Create a modern card with Advanced styling"""
        # Card container with shadow effect
        card_container = tk.Frame(parent, bg=self.colors['background'])
        card_container.pack(fill=tk.X, pady=(0, 15))
        
        # Main card
        card = tk.Frame(card_container, bg=self.colors['surface'], relief='flat', bd=1)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Card header color logic
        if "FP Reduction" in title or "False Positive" in title:
            header_color = self.colors['fp_reduction']
        elif "SOTA" in title:
            header_color = self.colors['sota']
        elif "Analysis" in title:
            header_color = self.colors['primary']
        elif "Confidence" in title:
            header_color = self.colors['confidence']
        else:
            header_color = self.colors['secondary']
            
        header = tk.Frame(card, bg=header_color, height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(header,
                             text=title,
                             font=('Segoe UI', 12, 'bold'),
                             bg=header_color,
                             fg='white')
        title_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['surface'])
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        return content

    def create_image_panel(self, parent):
        """Create enhanced image display panel"""
        # Images container
        images_frame = tk.Frame(parent, bg=self.colors['background'])
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image card
        original_card = self.create_image_card(images_frame, "📷 Original X-ray Image")
        self.canvas_original = tk.Canvas(original_card,
                                       bg='#1A202C',
                                       highlightthickness=0,
                                       relief='flat')
        self.canvas_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Advanced analysis heatmap card
        heatmap_card = self.create_image_card(images_frame, "🚀 Advanced Analysis + Confidence Map")
        self.canvas_heatmap = tk.Canvas(heatmap_card,
                                      bg='#1A202C',
                                      highlightthickness=0,
                                      relief='flat')
        self.canvas_heatmap.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_image_card(self, parent, title):
        """Create a card specifically for image display"""
        card_frame = tk.Frame(parent, bg=self.colors['background'])
        card_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Card with Advanced styling
        card = tk.Frame(card_frame, bg=self.colors['surface'], relief='flat', bd=1)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Header với Advanced color for analysis
        header_color = self.colors['fp_reduction'] if "Advanced" in title else self.colors['primary']
        header = tk.Frame(card, bg=header_color, height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(header,
                             text=title,
                             font=('Segoe UI', 11, 'bold'),
                             bg=header_color,
                             fg='white')
        title_label.pack(side=tk.LEFT, padx=12, pady=8)
        
        # Image area
        image_area = tk.Frame(card, bg=self.colors['surface'])
        image_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        return image_area

    def create_footer(self, parent):
        """Create modern footer with Advanced styling"""
        footer = tk.Frame(parent, bg=self.colors['surface'], height=40, relief='flat', bd=1)
        footer.pack(fill=tk.X, pady=(20, 0))
        footer.pack_propagate(False)
        
        # Status with Advanced icon
        status_frame = tk.Frame(footer, bg=self.colors['surface'])
        status_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=15, pady=8)
        
        status_icon = tk.Label(status_frame,
                             text="🚀",
                             font=('Segoe UI', 12),
                             bg=self.colors['surface'])
        status_icon.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Advanced AI system with FP Reduction ready")
        self.status_label = tk.Label(status_frame,
                                   textvariable=self.status_var,
                                   font=('Segoe UI', 10),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text_secondary'])
        self.status_label.pack(side=tk.LEFT, padx=(8, 0))

    # Mode and UI Management Methods
    def on_mode_change(self):
        """Handle mode change between SOTA, ensemble, and SOTA+FP"""
        mode = self.current_mode.get()
        
        if mode == "sota_fp":
            # Show SOTA+FP UI
            self.fp_config_card.pack(fill=tk.X, pady=(0, 15))
            self.sota_methods_frame.pack(fill=tk.X, pady=5)
            self.predict_btn.config(text="🚀 Run Advanced Analysis + FP Reduction")
            
        elif mode == "sota":
            # Show SOTA only UI
            self.fp_config_card.pack_forget()
            self.sota_methods_frame.pack(fill=tk.X, pady=5)
            self.predict_btn.config(text="🔬 Run SOTA Analysis")
            
        else:  # ensemble
            # Show ensemble UI
            self.fp_config_card.pack_forget()
            self.sota_methods_frame.pack_forget()
            self.predict_btn.config(text="🔗 Run Ensemble Analysis")
        
        # Update predict button state
        self.update_predict_button_state()

    def on_fp_toggle(self):
        """Handle FP reduction toggle"""
        if self.fp_reduction_enabled.get():
            self.status_var.set("FP Reduction enabled - Enhanced specificity mode")
        else:
            self.status_var.set("FP Reduction disabled - Standard analysis mode")

    def update_predict_button_state(self):
        """Update predict button state based on current mode"""
        mode = self.current_mode.get()
        method = self.method_var.get()
        
        if mode == "sota_fp":
            # Check requirements based on selected method
            if "hough" in method and not self.hough_initialized:
                self.predict_btn.config(state=tk.DISABLED)
                return
            if "_fp" in method and not self.fp_reducer:
                self.predict_btn.config(state=tk.DISABLED)
                return
            # Enable if basic requirements met
            if self.model_loaded and self.current_image_path:
                self.predict_btn.config(state=tk.NORMAL)
            else:
                self.predict_btn.config(state=tk.DISABLED)
        elif mode == "sota":
            # Enable if SOTA model loaded and image selected
            if self.model_loaded and self.current_image_path:
                self.predict_btn.config(state=tk.NORMAL)
            else:
                self.predict_btn.config(state=tk.DISABLED)
        else:  # ensemble
            # Enable if ensemble loaded and image selected
            if self.ensemble_loaded and self.current_image_path:
                self.predict_btn.config(state=tk.NORMAL)
            else:
                self.predict_btn.config(state=tk.DISABLED)

    # Model Management Methods
    def load_sota_model_async(self, model_type="resnet50v2", region="XR_HAND"):
        """Load SOTA model asynchronously"""
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set("Loading SOTA model...")
        self.model_status_label.config(text="Loading SOTA model...", fg=self.colors['sota'])
        self.progress_bar.start(10)
        
        self.load_model_thread = threading.Thread(target=self.load_sota_model, args=(model_type, region))
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

    def load_sota_model(self, model_type="resnet50v2", region="XR_HAND"):
        """Load SOTA model in background"""
        try:
            # Define SOTA model paths
            base_dir = "C:\\Users\\USER\\Documents\\coze"
            model_paths = [
                os.path.join(base_dir, "models", "res", f"{model_type}_{region}_best.h5"),
                os.path.join(base_dir, "models", "den", f"{model_type}_{region}_best.h5"),
                os.path.join(base_dir, "models", f"{model_type}_{region}_best.h5"),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(f"SOTA model file not found: {model_type}_{region}_best.h5")

            # Initialize SOTA detector
            if StateOfTheArtFractureDetector:
                self.sota_detector = StateOfTheArtFractureDetector(model_path)
                self.model_loaded = True
                self.master.after(0, self.update_sota_model_info, model_type, region, model_path)
            else:
                raise ImportError("StateOfTheArtFractureDetector not available")
            
        except Exception as e:
            self.master.after(0, self.show_error, f"Error loading SOTA model: {str(e)}")

    def update_sota_model_info(self, model_type, region, model_path):
        """Update SOTA model information in UI"""
        self.progress_bar.stop()
        model_display_name = {"densenet121": "DenseNet121", "resnet50v2": "ResNet50V2"}.get(model_type, model_type)
        
        self.model_status_label.config(
            text=f"✅ SOTA {model_display_name} ({region})",
            fg=self.colors['sota']
        )
        self.status_var.set(f"SOTA model loaded: {model_display_name} for {region}")
        
        self.update_predict_button_state()

    def initialize_fp_reducer(self):
        """Initialize FP Reduction system"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a SOTA model first!")
            return
            
        self.fp_init_btn.config(state=tk.DISABLED)
        self.status_var.set("Initializing FP Reduction system...")
        self.fp_status_label.config(text="Initializing FP Reducer...", fg=self.colors['fp_reduction'])
        self.progress_bar.start(10)
        
        fp_thread = threading.Thread(target=self.load_fp_reducer)
        fp_thread.daemon = True
        fp_thread.start()

    def load_fp_reducer(self):
        """Load FP Reduction system in background"""
        try:
            # Apply TensorFlow compatibility fixes
            import tensorflow as tf
            
            # Ensure TensorFlow 2.x compatibility
            if not hasattr(tf, 'reduce_std'):
                def safe_reduce_std(input_tensor, axis=None, keepdims=False):
                    try:
                        return tf.math.reduce_std(input_tensor, axis=axis, keepdims=keepdims)
                    except AttributeError:
                        mean = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
                        squared_diff = tf.square(input_tensor - mean)
                        variance = tf.reduce_mean(squared_diff, axis=axis, keepdims=keepdims)
                        return tf.sqrt(variance)
                
                tf.reduce_std = safe_reduce_std
            
            # Initialize FP Reducer with SOTA model
            if not self.sota_detector:
                raise ValueError("SOTA detector not available")
                
            models = [self.sota_detector.cnn_model] if hasattr(self.sota_detector, 'cnn_model') else []
            
            if not models:
                raise ValueError("No SOTA model available for FP Reduction")
            
            # Check if AdvancedFalsePositiveReducer is available
            if AdvancedFalsePositiveReducer == type:
                raise ValueError("FP Reduction system not available - please check imports")
            
            self.fp_reducer = AdvancedFalsePositiveReducer(
                models, 
                target_specificity=self.target_specificity.get()
            )
            
            # Note: In production, you would train the FP reducer here with validation data
            # For demo, we'll use default settings
            
            self.master.after(0, self.update_fp_reducer_info)
            
        except Exception as e:
            error_msg = f"Error initializing FP Reducer: {str(e)}"
            print(f"🔧 {error_msg}")
            self.master.after(0, self.show_error, error_msg)

    def update_fp_reducer_info(self):
        """Update FP Reducer information in UI"""
        self.progress_bar.stop()
        
        self.fp_status_label.config(
            text="✅ FP Reducer Ready",
            fg=self.colors['fp_reduction']
        )
        self.status_var.set("FP Reduction system initialized - Enhanced specificity enabled")
        
        self.fp_init_btn.config(state=tk.NORMAL)
        self.update_predict_button_state()

    def initialize_sota_hough(self):
        """Initialize SOTA Hough Transform system"""
        self.hough_init_btn.config(state=tk.DISABLED)
        self.status_var.set("Initializing SOTA Hough Transform...")
        self.hough_status_label.config(text="Initializing SOTA Hough...", fg=self.colors['confidence'])
        self.progress_bar.start(10)
        
        hough_thread = threading.Thread(target=self.load_sota_hough)
        hough_thread.daemon = True
        hough_thread.start()

    def load_sota_hough(self):
        """Load SOTA Hough system in background"""
        try:
            # Apply TensorFlow fixes if needed
            import tensorflow as tf
            
            # Check for TensorFlow 2.x compatibility
            if not hasattr(tf, 'reduce_std'):
                # Add compatibility function
                def safe_reduce_std(input_tensor, axis=None, keepdims=False):
                    try:
                        return tf.math.reduce_std(input_tensor, axis=axis, keepdims=keepdims)
                    except AttributeError:
                        mean = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
                        squared_diff = tf.square(input_tensor - mean)
                        variance = tf.reduce_mean(squared_diff, axis=axis, keepdims=keepdims)
                        return tf.sqrt(variance)
                
                tf.reduce_std = safe_reduce_std
            
            # Initialize SOTA Hough Transform
            self.sota_hough = SOTAHoughTransform()
            self.hough_initialized = True
            self.master.after(0, self.update_sota_hough_info)
            
        except Exception as e:
            error_msg = f"Error initializing SOTA Hough: {str(e)}"
            print(f"🔧 {error_msg}")
            self.master.after(0, self.show_error, error_msg)

    def update_sota_hough_info(self):
        """Update SOTA Hough information in UI"""
        self.progress_bar.stop()
        
        self.hough_status_label.config(
            text="✅ SOTA Hough Ready",
            fg=self.colors['confidence']
        )
        self.status_var.set("SOTA Hough Transform initialized - Advanced line detection enabled")
        
        self.hough_init_btn.config(state=tk.NORMAL)
        self.update_predict_button_state()

    def show_sota_model_menu(self):
        """Show SOTA model selection menu"""
        menu_window = tk.Toplevel(self.master)
        menu_window.title("Select SOTA Model")
        menu_window.geometry("450x350")
        menu_window.configure(bg=self.colors['background'])
        menu_window.resizable(False, False)
        
        # Center the window
        menu_window.transient(self.master)
        menu_window.grab_set()
        
        # SOTA Title
        title_label = tk.Label(menu_window,
                             text="🚀 Select SOTA Model",
                             font=('Segoe UI', 18, 'bold'),
                             bg=self.colors['background'],
                             fg=self.colors['sota'])
        title_label.pack(pady=20)
        
        # SOTA Model options
        options_frame = tk.Frame(menu_window, bg=self.colors['background'])
        options_frame.pack(fill=tk.BOTH, expand=True, padx=30)
        
        sota_models = [
            ("SOTA DenseNet121", "densenet121", "🧠 Dynamic Snake Conv + Weighted Channel Attention"),
            ("SOTA ResNet50V2", "resnet50v2", "🔬 Multi-scale Fusion + Advanced Grad-CAM")
        ]
        
        for name, code, desc in sota_models:
            btn = tk.Button(options_frame,
                          text=f"{name}\n{desc}",
                          font=('Segoe UI', 11, 'bold'),
                          bg=self.colors['sota'],
                          fg='white',
                          activebackground=self.colors['primary'],
                          activeforeground='white',
                          relief='flat',
                          pady=15,
                          command=lambda c=code: self.select_sota_model(c, menu_window))
            btn.pack(fill=tk.X, pady=8)

    def select_sota_model(self, model_type, window):
        """Select and load SOTA model"""
        window.destroy()
        self.load_sota_model_async(model_type)

    def initialize_ensemble(self):
        """Initialize ensemble system"""
        if not MultiRegionEnsemble:
            messagebox.showerror("Error", "Ensemble system not available - please check imports")
            return
            
        self.ensemble_btn.config(state=tk.DISABLED)
        self.status_var.set("Initializing ensemble system...")
        self.progress_bar.start(10)
        
        ensemble_thread = threading.Thread(target=self.load_ensemble)
        ensemble_thread.daemon = True
        ensemble_thread.start()

    def load_ensemble(self):
        """Load ensemble system in background"""
        try:
            self.ensemble = MultiRegionEnsemble()
            self.ensemble_loaded = True
            self.master.after(0, self.update_ensemble_info)
        except Exception as e:
            self.master.after(0, self.show_error, f"Error initializing ensemble: {str(e)}")

    def update_ensemble_info(self):
        """Update ensemble information in UI"""
        self.progress_bar.stop()
        self.ensemble_btn.config(state=tk.NORMAL)
        self.update_predict_button_state()

    # Image Management
    def browse_image(self):
        """Browse for X-ray image"""
        supported_formats = ('.png', '.jpg', '.jpeg')
        image_path = filedialog.askopenfilename(
            title="Select X-ray Image for Advanced Analysis",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if image_path:
            self.current_image_path = image_path
            self.load_image(image_path)
            self.reset_prediction_results()
            self.update_predict_button_state()

    def load_image(self, image_path):
        """Load and display image"""
        try:
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image file: {image_path}")

            # Display original image
            self.display_image(image, self.canvas_original)

            # Display placeholder for analysis
            placeholder = np.zeros_like(image)
            self.display_image_with_text(placeholder, self.canvas_heatmap, "Awaiting advanced analysis...")

            # Update image info
            filename = os.path.basename(image_path)
            file_size = os.path.getsize(image_path) / 1024  # KB
            self.image_info_label.config(
                text=f"📁 {filename}\n💾 {file_size:.1f} KB\n📐 {image.shape[1]}x{image.shape[0]} pixels"
            )

            # Update status
            self.status_var.set(f"Image loaded: {filename}")

        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")

    def display_image(self, image, canvas):
        """Display image on canvas"""
        canvas.update()
        canvas_width = max(canvas.winfo_width(), 400)
        canvas_height = max(canvas.winfo_height(), 400)

        if len(image.shape) == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
            
        # Calculate aspect ratio
        aspect_ratio = w / h
        if canvas_width / canvas_height > aspect_ratio:
            new_height = canvas_height - 20
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width - 20
            new_height = int(new_width / aspect_ratio)

        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL for display
        if len(resized_image.shape) == 2:
            pil_image = Image.fromarray(resized_image).convert('RGB')
        else:
            pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        
        photo = ImageTk.PhotoImage(image=pil_image)
        canvas.delete("all")
        
        # Center the image
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=photo)
        canvas.image = photo

    def display_image_with_text(self, image, canvas, text):
        """Display image with overlay text"""
        self.display_image(image, canvas)
        canvas.update()
        
        # Add text overlay
        canvas.create_text(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            text=text,
            fill='white',
            font=('Segoe UI', 14, 'bold'),
            anchor='center'
        )

    # Prediction Methods
    def predict_image(self):
        """Run Advanced prediction with optional FP Reduction"""
        if not self.current_image_path:
            self.show_error("Please select an image first")
            return
        
        mode = self.current_mode.get()
        
        if mode == "sota_fp":
            if not self.model_loaded:
                self.show_error("Please load a SOTA model first")
                return
        elif mode == "sota":
            if not self.model_loaded:
                self.show_error("Please load a SOTA model first")
                return
        else:  # ensemble
            if not self.ensemble_loaded:
                self.show_error("Please initialize ensemble system first")
                return
            
        self.predict_btn.config(state=tk.DISABLED)
        
        if mode == "sota_fp":
            self.status_var.set("Running advanced analysis with FP reduction...")
            analysis_text = "🚀 Running advanced analysis with FP reduction..."
            color = self.colors['fp_reduction']
        elif mode == "sota":
            self.status_var.set("Running SOTA analysis...")
            analysis_text = "🔬 Running SOTA analysis..."
            color = self.colors['sota']
        else:
            self.status_var.set("Running ensemble analysis...")
            analysis_text = "🔗 Running ensemble analysis..."
            color = self.colors['accent']
            
        self.progress_bar.start(10)
        
        # Update UI to show analysis in progress
        self.prediction_result_label.config(text=analysis_text, fg=color)
        
        if mode == "sota_fp":
            method = self.method_var.get()
            prediction_thread = threading.Thread(
                target=self.predict_sota_fp_in_thread,
                args=(self.current_image_path, method)
            )
        elif mode == "sota":
            method = self.method_var.get()
            prediction_thread = threading.Thread(
                target=self.predict_sota_in_thread,
                args=(self.current_image_path, method)
            )
        else:  # ensemble
            voting_method = "weighted_average"  # Default
            prediction_thread = threading.Thread(
                target=self.predict_ensemble_in_thread,
                args=(self.current_image_path, voting_method)
            )
        
        prediction_thread.daemon = True
        prediction_thread.start()

    def predict_sota_fp_in_thread(self, image_path, method):
        """Perform SOTA prediction with FP reduction in separate thread"""
        try:
            # Read image for analysis
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image from {image_path}")
            
            # Determine base method and analysis type
            if "hough_only" in method:
                # Pure SOTA Hough analysis without CNN
                if not self.hough_initialized:
                    raise ValueError("SOTA Hough not initialized")
                
                hough_score, hough_heatmap, hough_analysis = self.sota_hough.predict_fracture(image)
                
                # Create result structure
                result = {
                    'image_path': image_path,
                    'image': image,
                    'method': method,
                    'score': hough_score,
                    'predicted_label': 1 if hough_score > 0.5 else 0,
                    'true_label': 1 if "abnormal" in image_path.lower() else 0,
                    'confidence': hough_analysis['confidence'] * 100,
                    'heatmap': hough_heatmap,
                    'processing_time': 2.5,  # Approximate
                    'hough_analysis': hough_analysis,
                    'analysis_status': f"SOTA Hough - {hough_analysis['pattern_type']} pattern",
                    'final_prediction': 1 if hough_score > 0.5 else 0,
                    'coverage': True
                }
                
            elif "hough" in method:
                # SOTA Hough + CNN combination
                if not self.hough_initialized:
                    raise ValueError("SOTA Hough not initialized")
                
                # Get CNN prediction
                base_method = method.replace('_fp', '').replace('_hough', '_cnn')
                sota_result = self.sota_detector.predict(image_path, method=base_method)
                
                # Get SOTA Hough prediction
                hough_score, hough_heatmap, hough_analysis = self.sota_hough.predict_fracture(image)
                
                # Combine predictions (weighted average)
                cnn_weight = 0.6
                hough_weight = 0.4
                combined_score = cnn_weight * sota_result['score'] + hough_weight * hough_score
                
                # Combine heatmaps
                cnn_heatmap_resized = cv2.resize(sota_result['heatmap'], 
                                               (hough_heatmap.shape[1], hough_heatmap.shape[0]))
                combined_heatmap = cnn_weight * cnn_heatmap_resized + hough_weight * hough_heatmap
                
                result = {
                    **sota_result,
                    'hough_score': hough_score,
                    'hough_analysis': hough_analysis,
                    'combined_score': combined_score,
                    'score': combined_score,
                    'heatmap': combined_heatmap,
                    'predicted_label': 1 if combined_score > 0.5 else 0,
                    'analysis_status': f"SOTA CNN+Hough - {hough_analysis['pattern_type']} pattern"
                }
                
            else:
                # Standard CNN-based prediction
                base_method = method.replace('_fp', '')
                sota_result = self.sota_detector.predict(image_path, method=base_method)
                result = sota_result
            
            # Apply FP reduction if enabled and method includes _fp
            if self.fp_reduction_enabled.get() and "_fp" in method and self.fp_reducer:
                img_tensor, _ = self.sota_detector.preprocess_for_cnn(image)
                
                # Apply FP reduction
                fp_result = self.fp_reducer.predict_with_fp_reduction(img_tensor)
                
                # Update result with FP reduction
                result.update({
                    'fp_prediction': fp_result['prediction'][0],
                    'fp_confidence': fp_result['confidence'][0],
                    'fp_calibrated_score': fp_result['calibrated_score'][0],
                    'fp_high_confidence': fp_result['high_confidence'][0],
                    'uncertainty': 1 - fp_result['confidence'][0],
                })
                
                # Override prediction if FP reduction suggests uncertainty
                if fp_result['prediction'][0] == -1:  # Uncertain
                    result['final_prediction'] = 'uncertain'
                    result['coverage'] = False
                    result['analysis_status'] += " - Flagged for Review"
                else:
                    result['final_prediction'] = fp_result['prediction'][0]
                    result['coverage'] = True
            else:
                # No FP reduction applied
                result.update({
                    'fp_confidence': result['confidence'] / 100,
                    'fp_calibrated_score': result['score'],
                    'uncertainty': 1 - (result['confidence'] / 100),
                    'final_prediction': result['predicted_label'],
                    'coverage': True
                })
                
                if 'analysis_status' not in result:
                    result['analysis_status'] = 'Standard Analysis'
            
            self.prediction_result = result
            self.master.after(0, self.update_sota_fp_results, result)
            
        except Exception as e:
            self.master.after(0, self.show_error, f"Advanced analysis error: {str(e)}")
            self.master.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

    def predict_sota_in_thread(self, image_path, method):
        """Perform SOTA prediction in separate thread"""
        try:
            result = self.sota_detector.predict(image_path, method=method)
            self.prediction_result = result
            self.master.after(0, self.update_sota_prediction_results, result)
        except Exception as e:
            self.master.after(0, self.show_error, f"SOTA prediction error: {str(e)}")
            self.master.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

    def predict_ensemble_in_thread(self, image_path, voting_method):
        """Perform ensemble prediction in separate thread"""
        try:
            result = self.ensemble.predict(image_path, voting_method)
            self.prediction_result = result
            self.master.after(0, self.update_ensemble_results, result)
        except Exception as e:
            self.master.after(0, self.show_error, f"Ensemble prediction error: {str(e)}")
            self.master.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

    def update_sota_fp_results(self, result):
        """Update SOTA + FP Reduction results"""
        self.progress_bar.stop()
        
        # Determine final prediction
        if result.get('final_prediction') == 'uncertain':
            self.prediction_result_label.config(
                text="❓ UNCERTAIN - Requires Expert Review",
                fg=self.colors['accent']
            )
            diagnosis = "uncertain"
        elif result.get('final_prediction', result['predicted_label']) == 1:
            self.prediction_result_label.config(
                text="⚠️ FRACTURE DETECTED (Advanced + FP Reduction)",
                fg=self.colors['success']
            )
            diagnosis = "fracture detected"
        else:
            self.prediction_result_label.config(
                text="✅ NO FRACTURE DETECTED (Advanced + FP Reduction)",
                fg=self.colors['sota']
            )
            diagnosis = "no fracture"
        
        # Update confidence metrics
        fp_confidence = result.get('fp_confidence', result['confidence'] / 100)
        uncertainty = result.get('uncertainty', 1 - fp_confidence)
        
        self.confidence_label.config(text=f"{fp_confidence*100:.1f}%")
        self.uncertainty_label.config(text=f"{uncertainty*100:.1f}%")
        self.coverage_label.config(text=result.get('analysis_status', 'N/A'))
        
        # Display advanced heatmap
        self.display_advanced_heatmap(result['image'], result['heatmap'], fp_confidence)
        
        # Update status
        coverage_text = "with coverage" if result.get('coverage', True) else "flagged for review"
        self.status_var.set(f"Advanced analysis complete: {diagnosis} ({fp_confidence*100:.1f}% confidence) {coverage_text}")
        self.predict_btn.config(state=tk.NORMAL)

    def update_sota_prediction_results(self, result):
        """Update SOTA prediction results"""
        self.progress_bar.stop()
        
        # Update prediction label
        if result['predicted_label'] == 1:
            self.prediction_result_label.config(
                text="⚠️ FRACTURE DETECTED (SOTA)",
                fg=self.colors['success']
            )
        else:
            self.prediction_result_label.config(
                text="✅ NO FRACTURE DETECTED (SOTA)",
                fg=self.colors['sota']
            )
        
        # Update confidence
        confidence = result['confidence'] / 100
        self.confidence_label.config(text=f"{result['confidence']:.1f}%")
        self.uncertainty_label.config(text=f"{(1-confidence)*100:.1f}%")
        self.coverage_label.config(text="Standard Analysis")
        
        # Display heatmap
        self.display_advanced_heatmap(result['image'], result['heatmap'], confidence)
        
        # Update status
        diagnosis = "fracture detected" if result['predicted_label'] == 1 else "no fracture"
        self.status_var.set(f"SOTA analysis complete: {diagnosis} ({result['confidence']:.1f}% confidence)")
        self.predict_btn.config(state=tk.NORMAL)

    def update_ensemble_results(self, result):
        """Update ensemble prediction results"""
        self.progress_bar.stop()
        
        # Update prediction label
        if result['predicted_label'] == 1:
            self.prediction_result_label.config(
                text="⚠️ FRACTURE DETECTED (Ensemble)",
                fg=self.colors['success']
            )
        else:
            self.prediction_result_label.config(
                text="✅ NO FRACTURE DETECTED (Ensemble)",
                fg=self.colors['primary']
            )
        
        # Update confidence
        confidence = result['confidence'] / 100
        self.confidence_label.config(text=f"{result['confidence']:.1f}%")
        self.uncertainty_label.config(text=f"{(1-confidence)*100:.1f}%")
        self.coverage_label.config(text="Ensemble Analysis")
        
        # Display heatmap
        self.display_advanced_heatmap(result['image'], result['heatmap'], confidence)
        
        # Update status
        diagnosis = "fracture detected" if result['predicted_label'] == 1 else "no fracture"
        self.status_var.set(f"Ensemble analysis complete: {diagnosis} ({result['confidence']:.1f}% confidence)")
        self.predict_btn.config(state=tk.NORMAL)

    def display_advanced_heatmap(self, original_image, heatmap, confidence):
        """Display advanced heatmap with confidence visualization"""
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        alpha = 0.6
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        overlaid = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Add confidence visualization border
        h, w = overlaid.shape[:2]
        border_thickness = max(5, min(h, w) // 100)
        
        # Color based on confidence
        if confidence > 0.8:
            border_color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.6:
            border_color = (0, 255, 255)  # Yellow - medium confidence
        elif confidence > 0.4:
            border_color = (0, 165, 255)  # Orange - low confidence
        else:
            border_color = (0, 0, 255)  # Red - very low confidence
        
        # Draw confidence border
        cv2.rectangle(overlaid, (0, 0), (w-1, h-1), border_color, border_thickness)
        
        # Add confidence text overlay
        font_scale = max(0.5, min(h, w) / 800)
        font_thickness = max(1, int(font_scale * 2))
        
        confidence_text = f"Confidence: {confidence*100:.1f}%"
        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # Position text at top-right corner
        text_x = w - text_size[0] - 10
        text_y = 30
        
        # Add text background
        cv2.rectangle(overlaid, (text_x - 5, text_y - 20), 
                     (text_x + text_size[0] + 5, text_y + 5), 
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(overlaid, confidence_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        self.display_image(overlaid, self.canvas_heatmap)

    def reset_prediction_results(self):
        """Reset prediction results"""
        self.prediction_result_label.config(
            text="Awaiting advanced analysis...",
            fg=self.colors['text_secondary']
        )
        self.confidence_label.config(text="-")
        self.uncertainty_label.config(text="-")
        self.coverage_label.config(text="-")
        self.prediction_result = None

    def show_confidence_analysis(self):
        """Show detailed confidence analysis"""
        if self.prediction_result is None:
            messagebox.showinfo("Info", "No analysis results available!")
            return
        
        # Create confidence analysis window
        conf_window = tk.Toplevel(self.master)
        conf_window.title("📊 Confidence Analysis")
        conf_window.geometry("800x700")
        conf_window.configure(bg=self.colors['background'])
        
        # Header
        header_frame = tk.Frame(conf_window, bg=self.colors['confidence'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="📊 Advanced Confidence Analysis",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['confidence'],
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Content frame
        content_frame = tk.Frame(conf_window, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create scrollable content
        canvas = tk.Canvas(content_frame, bg=self.colors['background'])
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['background'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add confidence analysis content
        self.create_confidence_analysis_content(scrollable_frame)

    def create_confidence_analysis_content(self, parent):
        """Create detailed confidence analysis content"""
        result = self.prediction_result
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, (np.ndarray, list)):
                    return float(value[0]) if len(value) > 0 else default
                elif hasattr(value, 'item'):  # numpy scalar
                    return float(value.item())
                else:
                    return float(value)
            except (TypeError, ValueError, IndexError):
                return default
        
        # Main prediction card
        pred_card = self.create_analysis_card(parent, "🎯 Prediction Summary")
        
        # Prediction details
        pred_text = "FRACTURE DETECTED" if result.get('final_prediction', result['predicted_label']) == 1 else "NO FRACTURE"
        if result.get('final_prediction') == 'uncertain':
            pred_text = "UNCERTAIN - REQUIRES REVIEW"
            pred_color = self.colors['accent']
        elif result.get('final_prediction', result['predicted_label']) == 1:
            pred_color = self.colors['success']
        else:
            pred_color = self.colors['sota']
        
        pred_label = tk.Label(pred_card, text=pred_text,
                            font=('Segoe UI', 16, 'bold'),
                            bg=self.colors['surface'],
                            fg=pred_color)
        pred_label.pack(pady=10)
        
        # Confidence metrics
        conf_card = self.create_analysis_card(parent, "📊 Confidence Metrics")
        
        # Safe conversion of confidence values
        fp_conf = safe_float(result.get('fp_confidence', result['confidence']/100))
        uncertainty = safe_float(result.get('uncertainty', 1-result['confidence']/100))
        proc_time = safe_float(result.get('processing_time', 0))
        
        metrics = [
            ("Prediction Confidence", f"{fp_conf*100:.1f}%"),
            ("Uncertainty", f"{uncertainty*100:.1f}%"),
            ("Coverage", "Yes" if result.get('coverage', True) else "Flagged for Review"),
            ("Analysis Method", result.get('method', 'Unknown')),
            ("Processing Time", f"{proc_time:.2f}s")
        ]
        
        for metric, value in metrics:
            metric_frame = tk.Frame(conf_card, bg=self.colors['surface'])
            metric_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(metric_frame, text=f"{metric}:",
                   font=('Segoe UI', 10),
                   bg=self.colors['surface'],
                   fg=self.colors['text_primary']).pack(side=tk.LEFT)
            
            tk.Label(metric_frame, text=value,
                   font=('Segoe UI', 10, 'bold'),
                   bg=self.colors['surface'],
                   fg=self.colors['confidence']).pack(side=tk.RIGHT)
        
        # FP Reduction details (if available)
        if 'fp_prediction' in result:
            fp_card = self.create_analysis_card(parent, "🛡️ False Positive Reduction")
            
            fp_pred = result.get('fp_prediction', 'N/A')
            fp_conf_val = safe_float(result.get('fp_confidence', 0))
            fp_cal_score = safe_float(result.get('fp_calibrated_score', 0))
            
            fp_metrics = [
                ("FP Prediction", str(fp_pred)),
                ("FP Confidence", f"{fp_conf_val*100:.1f}%"),
                ("Calibrated Score", f"{fp_cal_score:.4f}"),
                ("High Confidence", "Yes" if result.get('fp_high_confidence', False) else "No")
            ]
            
            for metric, value in fp_metrics:
                metric_frame = tk.Frame(fp_card, bg=self.colors['surface'])
                metric_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(metric_frame, text=f"{metric}:",
                       font=('Segoe UI', 10),
                       bg=self.colors['surface'],
                       fg=self.colors['text_primary']).pack(side=tk.LEFT)
                
                tk.Label(metric_frame, text=value,
                       font=('Segoe UI', 10, 'bold'),
                       bg=self.colors['surface'],
                       fg=self.colors['fp_reduction']).pack(side=tk.RIGHT)
        
        # Hough analysis details (if available)
        if 'hough_analysis' in result:
            hough_card = self.create_analysis_card(parent, "📐 SOTA Hough Analysis")
            
            hough_analysis = result['hough_analysis']
            
            frac_score = safe_float(hough_analysis.get('fracture_score', 0))
            hough_conf = safe_float(hough_analysis.get('confidence', 0))
            
            hough_metrics = [
                ("Fracture Score", f"{frac_score:.4f}"),
                ("Pattern Type", hough_analysis.get('pattern_type', 'Unknown')),
                ("Line Count", str(hough_analysis.get('line_count', 0))),
                ("Analysis Confidence", f"{hough_conf*100:.1f}%")
            ]
            
            for metric, value in hough_metrics:
                metric_frame = tk.Frame(hough_card, bg=self.colors['surface'])
                metric_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(metric_frame, text=f"{metric}:",
                       font=('Segoe UI', 10),
                       bg=self.colors['surface'],
                       fg=self.colors['text_primary']).pack(side=tk.LEFT)
                
                tk.Label(metric_frame, text=value,
                       font=('Segoe UI', 10, 'bold'),
                       bg=self.colors['surface'],
                       fg=self.colors['confidence']).pack(side=tk.RIGHT)
        
        # Ground truth comparison
        gt_card = self.create_analysis_card(parent, "✅ Ground Truth Comparison")
        
        true_label = "FRACTURE" if result['true_label'] == 1 else "NORMAL"
        predicted_label = result.get('final_prediction', result['predicted_label'])
        
        if predicted_label == 'uncertain':
            accuracy_text = "UNCERTAIN - Cannot determine accuracy"
            accuracy_color = self.colors['accent']
        else:
            is_correct = predicted_label == result['true_label']
            accuracy_text = "CORRECT ✓" if is_correct else "INCORRECT ✗"
            accuracy_color = self.colors['sota'] if is_correct else self.colors['success']
        
        gt_metrics = [
            ("Ground Truth", true_label),
            ("Prediction", pred_text),
            ("Accuracy", accuracy_text)
        ]
        
        for metric, value in gt_metrics:
            metric_frame = tk.Frame(gt_card, bg=self.colors['surface'])
            metric_frame.pack(fill=tk.X, pady=2)
            
            tk.Label(metric_frame, text=f"{metric}:",
                   font=('Segoe UI', 10),
                   bg=self.colors['surface'],
                   fg=self.colors['text_primary']).pack(side=tk.LEFT)
            
            color = accuracy_color if metric == "Accuracy" else self.colors['text_primary']
            tk.Label(metric_frame, text=value,
                   font=('Segoe UI', 10, 'bold'),
                   bg=self.colors['surface'],
                   fg=color).pack(side=tk.RIGHT)

    def create_analysis_card(self, parent, title):
        """Create analysis card for confidence window"""
        card_container = tk.Frame(parent, bg=self.colors['background'])
        card_container.pack(fill=tk.X, pady=(0, 15))
        
        card = tk.Frame(card_container, bg=self.colors['surface'], relief='flat', bd=1)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_color = self.colors['confidence']
        header = tk.Frame(card, bg=header_color, height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text=title,
                             font=('Segoe UI', 11, 'bold'),
                             bg=header_color, fg='white')
        title_label.pack(side=tk.LEFT, padx=12, pady=8)
        
        # Content
        content = tk.Frame(card, bg=self.colors['surface'])
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        return content

    def save_result(self):
        """Save Advanced analysis results"""
        if self.prediction_result is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Advanced Analysis Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if save_path:
            try:
                # Helper function to safely convert to float for JSON serialization
                def safe_float(value, default=0.0):
                    try:
                        if isinstance(value, (np.ndarray, list)):
                            return float(value[0]) if len(value) > 0 else default
                        elif hasattr(value, 'item'):  # numpy scalar
                            return float(value.item())
                        else:
                            return float(value)
                    except (TypeError, ValueError, IndexError):
                        return default
                
                if save_path.endswith('.json'):
                    # Save as JSON
                    result_data = {
                        'timestamp': datetime.now().isoformat(),
                        'image_path': self.prediction_result['image_path'],
                        'method': self.prediction_result['method'],
                        'prediction': {
                            'score': safe_float(self.prediction_result['score']),
                            'predicted_label': int(self.prediction_result['predicted_label']),
                            'confidence': safe_float(self.prediction_result['confidence']),
                            'final_prediction': self.prediction_result.get('final_prediction', self.prediction_result['predicted_label'])
                        },
                        'ground_truth': int(self.prediction_result['true_label']),
                        'processing_time': safe_float(self.prediction_result['processing_time']),
                        'fp_reduction': {
                            'enabled': self.fp_reduction_enabled.get(),
                            'confidence': safe_float(self.prediction_result.get('fp_confidence', 0)),
                            'uncertainty': safe_float(self.prediction_result.get('uncertainty', 0)),
                            'coverage': bool(self.prediction_result.get('coverage', True))
                        }
                    }
                    
                    # Add Hough analysis if available
                    if 'hough_analysis' in self.prediction_result:
                        hough_analysis = self.prediction_result['hough_analysis']
                        result_data['hough_analysis'] = {
                            'fracture_score': safe_float(hough_analysis.get('fracture_score', 0)),
                            'pattern_type': hough_analysis.get('pattern_type', 'Unknown'),
                            'line_count': int(hough_analysis.get('line_count', 0)),
                            'confidence': safe_float(hough_analysis.get('confidence', 0))
                        }
                    
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=2, ensure_ascii=False)
                    
                    self.status_var.set(f"Results saved: {os.path.basename(save_path)}")
                    messagebox.showinfo("Success", f"Results saved successfully!\n{save_path}")
                    
                else:
                    # Save as image visualization
                    fig = self.create_advanced_visualization(self.prediction_result)
                    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    self.status_var.set(f"Visualization saved: {os.path.basename(save_path)}")
                    messagebox.showinfo("Success", f"Visualization saved successfully!\n{save_path}")
                    
            except Exception as e:
                self.show_error(f"Error saving results: {str(e)}")

    def create_advanced_visualization(self, result):
        """Create advanced visualization of results"""
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, (np.ndarray, list)):
                    return float(value[0]) if len(value) > 0 else default
                elif hasattr(value, 'item'):  # numpy scalar
                    return float(value.item())
                else:
                    return float(value)
            except (TypeError, ValueError, IndexError):
                return default
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[1, 1, 1])
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(result['image'], cmap='gray')
        ax1.set_title('📷 Original X-ray', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(result['image'], cmap='gray')
        heatmap_overlay = ax2.imshow(result['heatmap'], cmap='jet', alpha=0.6)
        ax2.set_title('🚀 Advanced Analysis Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Results summary
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        # Determine prediction text and color with safe conversion
        final_pred = result.get('final_prediction', result['predicted_label'])
        if final_pred == 'uncertain':
            pred_text = "❓ UNCERTAIN\nREQUIRES REVIEW"
            pred_color = 'orange'
        elif final_pred == 1:
            pred_text = "⚠️ FRACTURE\nDETECTED"
            pred_color = 'red'
        else:
            pred_text = "✅ NO FRACTURE\nDETECTED"
            pred_color = 'green'
        
        true_text = "FRACTURE" if result['true_label'] == 1 else "NORMAL"
        
        # Accuracy assessment
        if final_pred == 'uncertain':
            accuracy_text = "UNCERTAIN"
            accuracy_color = 'orange'
        else:
            is_correct = final_pred == result['true_label']
            accuracy_text = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            accuracy_color = 'green' if is_correct else 'red'
        
        # Safe conversions for display
        fp_conf = safe_float(result.get('fp_confidence', result['confidence']/100))
        uncertainty = safe_float(result.get('uncertainty', 1-result['confidence']/100))
        proc_time = safe_float(result['processing_time'])
        
        summary_text = f"""
ADVANCED AI ANALYSIS

Method: {result['method'].upper()}
Processing: {proc_time:.2f}s

PREDICTION:
{pred_text}

Confidence: {fp_conf*100:.1f}%
Uncertainty: {uncertainty*100:.1f}%

GROUND TRUTH:
{true_text}

RESULT: {accuracy_text}

FP Reduction: {'✅ ENABLED' if result.get('coverage', True) else '⚠️ FLAGGED'}
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=accuracy_color, alpha=0.1))
        
        # Confidence metrics visualization
        ax4 = fig.add_subplot(gs[1, :])
        
        confidence_val = safe_float(result['confidence'])
        
        metrics = ['Prediction\nConfidence', 'Uncertainty', 'Model\nConfidence']
        values = [
            fp_conf * 100,
            uncertainty * 100,
            confidence_val
        ]
        colors = [self.colors['confidence'], self.colors['fp_reduction'], self.colors['sota']]
        
        bars = ax4.bar(metrics, values, color=[c for c in colors], alpha=0.8)
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('📊 Confidence Metrics', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Analysis details
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create analysis details text with safe conversions
        score_val = safe_float(result['score'])
        target_spec = safe_float(self.target_specificity.get())
        
        details_text = f"""
DETAILED ANALYSIS REPORT

📍 Image: {os.path.basename(result['image_path'])}
🔬 Analysis Method: {result['method']}
⏱️ Processing Time: {proc_time:.2f} seconds
🎯 Target Specificity: {target_spec:.2f}

📊 PREDICTION DETAILS:
• Raw Score: {score_val:.4f}
• Predicted Label: {result['predicted_label']}
• Final Prediction: {result.get('final_prediction', result['predicted_label'])}
• Coverage: {'Yes' if result.get('coverage', True) else 'Flagged for Expert Review'}

🛡️ FALSE POSITIVE REDUCTION:
• Enabled: {'Yes' if self.fp_reduction_enabled.get() else 'No'}
• FP Confidence: {fp_conf*100:.1f}%
• Calibrated Score: {safe_float(result.get('fp_calibrated_score', 0)):.4f}
        """
        
        # Add Hough analysis if available
        if 'hough_analysis' in result:
            hough = result['hough_analysis']
            frac_score = safe_float(hough.get('fracture_score', 0))
            hough_conf = safe_float(hough.get('confidence', 0))
            
            details_text += f"""

📐 SOTA HOUGH ANALYSIS:
• Fracture Score: {frac_score:.4f}
• Pattern Type: {hough.get('pattern_type', 'Unknown')}
• Lines Detected: {hough.get('line_count', 0)}
• Hough Confidence: {hough_conf*100:.1f}%
        """
        
        ax5.text(0.02, 0.98, details_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        
        # Footer
        footer_text = f"Advanced AI Fracture Detection System with FP Reduction | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fig.text(0.5, 0.02, footer_text, ha='center', fontsize=8, style='italic')
        
        plt.suptitle('🚀 Advanced AI Fracture Detection Analysis Report', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig

    def show_error(self, message):
        """Show error dialog"""
        messagebox.showerror("Error", message)
        self.progress_bar.stop()
        
    def show_info(self, title, message):
        """Show info dialog"""
        messagebox.showinfo(title, message)

def main():
    """Main function to run the Advanced application"""
    print("🚀" * 30)
    print("ADVANCED AI FRACTURE DETECTION SYSTEM")
    print("🚀" * 30)
    print("🔬 Advanced Features:")
    print("   ✅ State-of-the-Art CNN Models")
    print("   ✅ Advanced Hough Transform")
    print("   ✅ False Positive Reduction")
    print("   ✅ Uncertainty Quantification")
    print("   ✅ Confidence Calibration")
    print("   ✅ Hard Negative Mining")
    print("   ✅ Ensemble Learning")
    print("   ✅ Multi-modal Analysis")
    print("🚀" * 30)
    
    # Check for required dependencies
    missing_deps = []
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} available")
    except ImportError:
        missing_deps.append("tensorflow")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__} available")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        from PIL import Image
        print(f"✅ PIL available")
    except ImportError:
        missing_deps.append("pillow")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies before running the application.")
        return
    
    root = tk.Tk()
    
    # Set window icon (optional)
    try:
        # You can add an icon file here
        # root.iconbitmap('advanced_icon.ico')
        pass
    except:
        pass
    
    # Initialize the application
    try:
        app = AdvancedFractureDetectionApp(root)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f'+{x}+{y}')
        
        print("\n🎉 Advanced AI Fracture Detection System launched successfully!")
        print("💡 Tips:")
        print("   • Load a SOTA model first")
        print("   • Initialize FP Reduction for enhanced accuracy")
        print("   • Initialize SOTA Hough for advanced line detection")
        print("   • Enable FP Reduction in settings for optimal results")
        print("   • Use SOTA + FP Reduction mode for best performance")
        
        root.mainloop()
        
    except Exception as e:
        print(f"\n❌ Error launching application: {str(e)}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
