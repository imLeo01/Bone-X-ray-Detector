# app.py
# Giao diện người dùng đồ họa hiện đại cho hệ thống phát hiện gãy xương

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

# Import thêm cho ensemble
from ensemble_prediction import MultiRegionEnsemble

from prediction import FractureDetector

class ModernFractureDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Bone Fracture Detection System")
        self.master.geometry("1400x900")
        self.master.minsize(1300, 800)
        
        # Modern color scheme
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Pink
            'accent': '#F18F01',       # Orange
            'success': '#C73E1D',      # Red for fracture detection
            'background': '#F5F7FA',   # Light gray
            'surface': '#FFFFFF',      # White
            'text_primary': '#2D3748', # Dark gray
            'text_secondary': '#718096', # Medium gray
            'border': '#E2E8F0',       # Light border
            'shadow': '#CBD5E0'        # Shadow color
        }
        
        self.master.configure(bg=self.colors['background'])
        
        # Setup modern styling
        self.setup_styles()
        
        # Variables
        self.current_image_path = None
        self.prediction_result = None
        self.detector = None
        self.ensemble = None  # Ensemble system
        self.model_loaded = False
        self.ensemble_loaded = False  # Ensemble status
        self.load_model_thread = None
        self.selected_model = tk.StringVar(value="resnet50v2")
        self.progress_var = tk.DoubleVar()
        self.current_mode = tk.StringVar(value="single")  # "single" hoặc "ensemble"
        
        # Create modern interface
        self.create_modern_interface()
        
        # Load default model
        self.load_model_async()

    def setup_styles(self):
        """Setup modern styling for ttk widgets"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure modern button style
        self.style.configure('Modern.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=(20, 12),
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('Modern.TButton',
                      background=[('active', self.colors['primary']),
                                ('pressed', '#1A365D'),
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
                                ('pressed', '#E07A00'),
                                ('!active', self.colors['accent'])],
                      foreground=[('active', 'white'),
                                ('pressed', 'white'),
                                ('!active', 'white')])
        
        # Modern frame style
        self.style.configure('Card.TFrame',
                           background=self.colors['surface'],
                           borderwidth=1,
                           relief='flat')
        
        # Modern label styles
        self.style.configure('Title.TLabel',
                           font=('Segoe UI', 24, 'bold'),
                           background=self.colors['background'],
                           foreground=self.colors['text_primary'])
        
        self.style.configure('Subtitle.TLabel',
                           font=('Segoe UI', 14),
                           background=self.colors['surface'],
                           foreground=self.colors['text_secondary'])
        
        self.style.configure('Result.TLabel',
                           font=('Segoe UI', 12, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['text_primary'])
        
        self.style.configure('Success.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['success'])
        
        self.style.configure('Normal.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['primary'])

    def create_modern_interface(self):
        """Create modern, beautiful interface"""
        # Main container with padding
        main_container = tk.Frame(self.master, bg=self.colors['background'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_container)
        
        # Content area
        content_frame = tk.Frame(main_container, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel (controls and info)
        left_panel = tk.Frame(content_frame, bg=self.colors['background'], width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Right panel (image display)
        right_panel = tk.Frame(content_frame, bg=self.colors['background'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create panels content
        self.create_control_panel(left_panel)
        self.create_image_panel(right_panel)
        
        # Footer with status
        self.create_footer(main_container)

    def create_header(self, parent):
        """Create modern header with title and subtitle"""
        header_frame = tk.Frame(parent, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title with icon effect
        title_frame = tk.Frame(header_frame, bg=self.colors['background'])
        title_frame.pack(fill=tk.X)
        
        # Create a modern title with gradient effect simulation
        title_label = ttk.Label(title_frame, 
                               text="🔬 AI Bone Fracture Detection",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Version badge
        version_frame = tk.Frame(title_frame, bg=self.colors['accent'], relief='flat')
        version_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        version_label = tk.Label(version_frame, 
                               text=" v2.0 ",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['accent'],
                               fg='white')
        version_label.pack(padx=8, pady=4)
        
        # Subtitle
        subtitle_label = tk.Label(header_frame,
                                text="Advanced Computer Vision System for Medical Image Analysis",
                                font=('Segoe UI', 12),
                                bg=self.colors['background'],
                                fg=self.colors['text_secondary'])
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))

    def create_control_panel(self, parent):
        """Create modern control panel"""
        # Mode Selection Card (NEW)
        mode_card = self.create_card(parent, "🎯 Analysis Mode")
        
        mode_frame = tk.Frame(mode_card, bg=self.colors['surface'])
        mode_frame.pack(fill=tk.X, pady=10)
        
        single_rb = tk.Radiobutton(mode_frame,
                                  text="🔬 Single Model Analysis",
                                  variable=self.current_mode,
                                  value="single",
                                  font=('Segoe UI', 10),
                                  bg=self.colors['surface'],
                                  fg=self.colors['text_primary'],
                                  selectcolor=self.colors['accent'],
                                  activebackground=self.colors['surface'],
                                  command=self.on_mode_change)
        single_rb.pack(anchor=tk.W, pady=2)
        
        ensemble_rb = tk.Radiobutton(mode_frame,
                                   text="🚀 Multi-Region Ensemble",
                                   variable=self.current_mode,
                                   value="ensemble",
                                   font=('Segoe UI', 10),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text_primary'],
                                   selectcolor=self.colors['accent'],
                                   activebackground=self.colors['surface'],
                                   command=self.on_mode_change)
        ensemble_rb.pack(anchor=tk.W, pady=2)
        
        # Model Selection Card
        self.model_card = self.create_card(parent, "🤖 Model Configuration")
        
        # Model selection buttons
        btn_frame = tk.Frame(self.model_card, bg=self.colors['surface'])
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.model_btn = ttk.Button(btn_frame, 
                              text="📦 Select Model",
                              style='Modern.TButton',
                              command=self.show_model_menu)
        self.model_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.ensemble_btn = ttk.Button(btn_frame,
                                     text="🚀 Initialize Ensemble",
                                     style='Accent.TButton',
                                     command=self.initialize_ensemble)
        self.ensemble_btn.pack(fill=tk.X, pady=(5, 0))
        
        # Model info display
        self.model_info_frame = tk.Frame(self.model_card, bg=self.colors['surface'])
        self.model_info_frame.pack(fill=tk.X, pady=5)
        
        self.model_name_label = tk.Label(self.model_info_frame,
                                        text="Loading model...",
                                        font=('Segoe UI', 10),
                                        bg=self.colors['surface'],
                                        fg=self.colors['text_secondary'])
        self.model_name_label.pack(anchor=tk.W)
        
        self.region_label = tk.Label(self.model_info_frame,
                                   text="Region: XR_HAND",
                                   font=('Segoe UI', 10),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text_secondary'])
        self.region_label.pack(anchor=tk.W)
        
        # Ensemble info (initially hidden)
        self.ensemble_info_frame = tk.Frame(self.model_card, bg=self.colors['surface'])
        
        self.ensemble_status_label = tk.Label(self.ensemble_info_frame,
                                            text="Ensemble not initialized",
                                            font=('Segoe UI', 10),
                                            bg=self.colors['surface'],
                                            fg=self.colors['text_secondary'])
        self.ensemble_status_label.pack(anchor=tk.W)
        
        self.ensemble_models_label = tk.Label(self.ensemble_info_frame,
                                            text="Models: 0",
                                            font=('Segoe UI', 10),
                                            bg=self.colors['surface'],
                                            fg=self.colors['text_secondary'])
        self.ensemble_models_label.pack(anchor=tk.W)
        
        # Progress bar for model loading
        self.progress_bar = ttk.Progressbar(self.model_card,
                                          mode='indeterminate',
                                          length=300)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Image Selection Card
        image_card = self.create_card(parent, "📁 Image Selection")
        
        upload_btn = ttk.Button(image_card,
                               text="🖼️ Choose X-ray Image",
                               style='Accent.TButton',
                               command=self.browse_image)
        upload_btn.pack(fill=tk.X, pady=10)
        
        # Image info
        self.image_info_label = tk.Label(image_card,
                                       text="No image selected",
                                       font=('Segoe UI', 10),
                                       bg=self.colors['surface'],
                                       fg=self.colors['text_secondary'],
                                       wraplength=300)
        self.image_info_label.pack(anchor=tk.W, pady=5)
        
        # Prediction Method Card
        self.method_card = self.create_card(parent, "⚙️ Analysis Method")
        
        self.method_var = tk.StringVar(value="combined")
        
        # Single model methods
        self.single_methods_frame = tk.Frame(self.method_card, bg=self.colors['surface'])
        
        single_methods = [
            ("🧠 CNN Deep Learning", "cnn"),
            ("📐 Hough Transform", "hough"),
            ("🔀 Combined Analysis", "combined")
        ]
        
        for text, value in single_methods:
            rb = tk.Radiobutton(self.single_methods_frame,
                              text=text,
                              variable=self.method_var,
                              value=value,
                              font=('Segoe UI', 10),
                              bg=self.colors['surface'],
                              fg=self.colors['text_primary'],
                              selectcolor=self.colors['accent'],
                              activebackground=self.colors['surface'])
            rb.pack(anchor=tk.W, pady=2)
        
        # Ensemble methods
        self.ensemble_methods_frame = tk.Frame(self.method_card, bg=self.colors['surface'])
        
        self.voting_var = tk.StringVar(value="weighted_average")
        
        ensemble_methods = [
            ("⚖️ Weighted Average", "weighted_average"),
            ("🗳️ Majority Vote", "majority_vote"),
            ("🏆 Max Confidence", "max_confidence")
        ]
        
        for text, value in ensemble_methods:
            rb = tk.Radiobutton(self.ensemble_methods_frame,
                              text=text,
                              variable=self.voting_var,
                              value=value,
                              font=('Segoe UI', 10),
                              bg=self.colors['surface'],
                              fg=self.colors['text_primary'],
                              selectcolor=self.colors['accent'],
                              activebackground=self.colors['surface'])
            rb.pack(anchor=tk.W, pady=2)
        
        # Prediction Button
        self.predict_btn = ttk.Button(self.method_card,
                                     text="🔍 Analyze X-ray",
                                     style='Modern.TButton',
                                     command=self.predict_image,
                                     state=tk.DISABLED)
        self.predict_btn.pack(fill=tk.X, pady=(15, 5))
        
        # Update UI based on initial mode (NOW after all frames are created)
        self.on_mode_change()
        
        # Results Card
        self.create_results_card(parent)

    def create_results_card(self, parent):
        """Create modern results display card"""
        results_card = self.create_card(parent, "📊 Analysis Results")
        
        # Result display area
        self.result_display = tk.Frame(results_card, bg=self.colors['surface'])
        self.result_display.pack(fill=tk.X, pady=10)
        
        # Prediction result
        self.prediction_frame = tk.Frame(self.result_display, bg=self.colors['surface'])
        self.prediction_frame.pack(fill=tk.X, pady=5)
        
        pred_label = tk.Label(self.prediction_frame,
                            text="Diagnosis:",
                            font=('Segoe UI', 10, 'bold'),
                            bg=self.colors['surface'],
                            fg=self.colors['text_primary'])
        pred_label.pack(anchor=tk.W)
        
        self.prediction_result_label = tk.Label(self.prediction_frame,
                                              text="Awaiting analysis...",
                                              font=('Segoe UI', 12, 'bold'),
                                              bg=self.colors['surface'],
                                              fg=self.colors['text_secondary'])
        self.prediction_result_label.pack(anchor=tk.W, padx=10)
        
        # Confidence
        confidence_label = tk.Label(self.result_display,
                                  text="Confidence:",
                                  font=('Segoe UI', 10, 'bold'),
                                  bg=self.colors['surface'],
                                  fg=self.colors['text_primary'])
        confidence_label.pack(anchor=tk.W, pady=(10, 0))
        
        self.confidence_label = tk.Label(self.result_display,
                                       text="-",
                                       font=('Segoe UI', 11),
                                       bg=self.colors['surface'],
                                       fg=self.colors['text_secondary'])
        self.confidence_label.pack(anchor=tk.W, padx=10)
        
        # Scores frame
        scores_frame = tk.Frame(self.result_display, bg=self.colors['surface'])
        scores_frame.pack(fill=tk.X, pady=(10, 0))
        
        # CNN Score
        cnn_frame = tk.Frame(scores_frame, bg=self.colors['surface'])
        cnn_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(cnn_frame, text="CNN Score:",
               font=('Segoe UI', 9),
               bg=self.colors['surface'],
               fg=self.colors['text_primary']).pack(side=tk.LEFT)
        
        self.cnn_score_label = tk.Label(cnn_frame, text="-",
                                      font=('Segoe UI', 9, 'bold'),
                                      bg=self.colors['surface'],
                                      fg=self.colors['primary'])
        self.cnn_score_label.pack(side=tk.RIGHT)
        
        # Hough Score
        hough_frame = tk.Frame(scores_frame, bg=self.colors['surface'])
        hough_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(hough_frame, text="Hough Score:",
               font=('Segoe UI', 9),
               bg=self.colors['surface'],
               fg=self.colors['text_primary']).pack(side=tk.LEFT)
        
        self.hough_score_label = tk.Label(hough_frame, text="-",
                                        font=('Segoe UI', 9, 'bold'),
                                        bg=self.colors['surface'],
                                        fg=self.colors['secondary'])
        self.hough_score_label.pack(side=tk.RIGHT)
        
        # Action buttons
        action_frame = tk.Frame(results_card, bg=self.colors['surface'])
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        save_btn = ttk.Button(action_frame,
                             text="💾 Save Results",
                             style='Accent.TButton',
                             command=self.save_result)
        save_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        eval_btn = ttk.Button(action_frame,
                             text="📈 Evaluate Model",
                             style='Modern.TButton',
                             command=self.show_evaluation_menu)
        eval_btn.pack(side=tk.LEFT)

    def create_card(self, parent, title):
        """Create a modern card with shadow effect"""
        # Card container with shadow effect
        card_container = tk.Frame(parent, bg=self.colors['background'])
        card_container.pack(fill=tk.X, pady=(0, 15))
        
        # Shadow frame
        shadow_frame = tk.Frame(card_container, bg=self.colors['shadow'], height=2)
        shadow_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Main card
        card = tk.Frame(card_container, bg=self.colors['surface'], relief='flat', bd=1)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['primary'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(header,
                             text=title,
                             font=('Segoe UI', 12, 'bold'),
                             bg=self.colors['primary'],
                             fg='white')
        title_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Card content
        content = tk.Frame(card, bg=self.colors['surface'])
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        return content

    def create_image_panel(self, parent):
        """Create modern image display panel"""
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
        
        # Heatmap image card
        heatmap_card = self.create_image_card(images_frame, "🎯 Analysis Heatmap")
        self.canvas_heatmap = tk.Canvas(heatmap_card,
                                      bg='#1A202C',
                                      highlightthickness=0,
                                      relief='flat')
        self.canvas_heatmap.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_image_card(self, parent, title):
        """Create a card specifically for image display"""
        card_frame = tk.Frame(parent, bg=self.colors['background'])
        card_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Card with modern styling
        card = tk.Frame(card_frame, bg=self.colors['surface'], relief='flat', bd=1)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Frame(card, bg=self.colors['primary'], height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_label = tk.Label(header,
                             text=title,
                             font=('Segoe UI', 11, 'bold'),
                             bg=self.colors['primary'],
                             fg='white')
        title_label.pack(side=tk.LEFT, padx=12, pady=8)
        
        # Image area
        image_area = tk.Frame(card, bg=self.colors['surface'])
        image_area.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        return image_area

    def create_footer(self, parent):
        """Create modern footer with status and floating analysis button"""
        footer = tk.Frame(parent, bg=self.colors['surface'], height=40, relief='flat', bd=1)
        footer.pack(fill=tk.X, pady=(20, 0))
        footer.pack_propagate(False)
        
        # Status with icon
        status_frame = tk.Frame(footer, bg=self.colors['surface'])
        status_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=15, pady=8)
        
        status_icon = tk.Label(status_frame,
                             text="🔄",
                             font=('Segoe UI', 12),
                             bg=self.colors['surface'])
        status_icon.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready for analysis")
        self.status_label = tk.Label(status_frame,
                                   textvariable=self.status_var,
                                   font=('Segoe UI', 10),
                                   bg=self.colors['surface'],
                                   fg=self.colors['text_secondary'])
        self.status_label.pack(side=tk.LEFT, padx=(8, 0))
        
        # Create floating analysis button
        self.create_floating_analysis_button(parent)
        
        # Create floating results panel
        self.create_floating_results_panel(parent)

    def create_floating_analysis_button(self, parent):
        """Create a beautiful floating analysis button in bottom right corner"""
        # Container for floating button
        self.floating_container = tk.Frame(parent, bg=self.colors['background'])
        self.floating_container.place(relx=1.0, rely=1.0, anchor='se', x=-30, y=-30)
        
        # Create circular button effect with shadow
        self.create_floating_button_with_shadow()
        
        # Animation variables
        self.button_hover = False
        self.animation_frame = 0
        self.animate_button()

    def create_floating_button_with_shadow(self):
        """Create floating button with shadow effect"""
        # Shadow layer - using gray color instead of transparent
        shadow_canvas = tk.Canvas(
            self.floating_container,
            width=80, height=80,
            bg=self.colors['background'],
            highlightthickness=0
        )
        shadow_canvas.pack()
        
        # Draw shadow circle with multiple layers for blur effect
        shadow_colors = ['#D0D0D0', '#E0E0E0', '#F0F0F0']
        for i, color in enumerate(shadow_colors):
            offset = i + 4
            size = 76 - i
            shadow_canvas.create_oval(offset, offset, size, size, 
                                    fill=color, outline='')
        
        # Main button layer
        self.float_button_canvas = tk.Canvas(
            shadow_canvas,
            width=70, height=70,
            bg=self.colors['background'],
            highlightthickness=0
        )
        self.float_button_canvas.place(x=5, y=2)
        
        # Create gradient effect for button
        self.draw_gradient_button()
        
        # Add icon and text
        self.float_button_canvas.create_text(
            35, 25,
            text="🔬",
            font=('Segoe UI', 16),
            fill='white'
        )
        
        self.float_button_canvas.create_text(
            35, 45,
            text="ANALYZE",
            font=('Segoe UI', 8, 'bold'),
            fill='white'
        )
        
        # Bind events
        self.float_button_canvas.bind("<Button-1>", self.floating_button_click)
        self.float_button_canvas.bind("<Enter>", self.floating_button_enter)
        self.float_button_canvas.bind("<Leave>", self.floating_button_leave)
        
        # Make all canvas items clickable
        for item in self.float_button_canvas.find_all():
            self.float_button_canvas.tag_bind(item, "<Button-1>", self.floating_button_click)
            self.float_button_canvas.tag_bind(item, "<Enter>", self.floating_button_enter)
            self.float_button_canvas.tag_bind(item, "<Leave>", self.floating_button_leave)

    def draw_gradient_button(self, hover=False):
        """Draw gradient button with hover effect"""
        self.float_button_canvas.delete("button_bg")
        
        # Colors for gradient
        if hover:
            colors = ['#F18F01', '#E07A00', '#CC6600']  # Hover colors
        else:
            colors = ['#2E86AB', '#2574A1', '#1A5F7A']  # Normal colors
        
        # Draw gradient circles
        for i, color in enumerate(colors):
            size = 70 - (i * 4)
            offset = i * 2
            self.float_button_canvas.create_oval(
                offset, offset, size, size,
                fill=color, outline='',
                tags="button_bg"
            )
        
        # Move text to front
        self.float_button_canvas.tag_raise("all")

    def create_floating_results_panel(self, parent):
        """Create floating results panel in top-right corner"""
        # Container for floating results panel
        self.floating_results_container = tk.Frame(parent, bg=self.colors['background'])
        self.floating_results_container.place(relx=1.0, rely=0.0, anchor='ne', x=-20, y=20)
        
        # Main results panel (initially hidden)
        self.create_floating_results_with_animation()
        
        # Animation variables for results panel
        self.results_visible = False
        self.results_animation_frame = 0

    def create_floating_results_with_animation(self):
        """Create floating results panel with modern design - WITHOUT detailed scores section"""
        # Main panel frame
        self.results_panel = tk.Frame(
            self.floating_results_container,
            bg=self.colors['surface'],
            relief='flat',
            bd=2,
            width=350,
            height=300  # Reduced height since we removed detailed scores
        )
        
        # Initially hide the panel
        self.results_panel.pack_propagate(False)
        
        # Header with close button
        header_frame = tk.Frame(self.results_panel, bg=self.colors['primary'], height=40)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="📊 Analysis Results",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Close button
        close_btn = tk.Button(
            header_frame,
            text="✕",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            activebackground=self.colors['success'],
            activeforeground='white',
            relief='flat',
            bd=0,
            width=3,
            command=self.hide_floating_results
        )
        close_btn.pack(side=tk.RIGHT, padx=10, pady=8)
        
        # Content area
        content_frame = tk.Frame(self.results_panel, bg=self.colors['surface'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Main diagnosis display
        self.diagnosis_frame = tk.Frame(content_frame, bg=self.colors['surface'])
        self.diagnosis_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Large diagnosis text
        self.floating_diagnosis_label = tk.Label(
            self.diagnosis_frame,
            text="Awaiting Analysis...",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text_secondary'],
            wraplength=300,
            justify=tk.CENTER
        )
        self.floating_diagnosis_label.pack()
        
        # Confidence display with progress bar effect
        confidence_frame = tk.Frame(content_frame, bg=self.colors['surface'])
        confidence_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            confidence_frame,
            text="Confidence Level:",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text_primary']
        ).pack(anchor=tk.W)
        
        # Confidence progress bar
        self.confidence_progress = ttk.Progressbar(
            confidence_frame,
            mode='determinate',
            length=320,
            style='TProgressbar'
        )
        self.confidence_progress.pack(fill=tk.X, pady=(5, 0))
        
        self.floating_confidence_label = tk.Label(
            confidence_frame,
            text="0%",
            font=('Segoe UI', 10),
            bg=self.colors['surface'],
            fg=self.colors['text_secondary']
        )
        self.floating_confidence_label.pack(anchor=tk.E, pady=(2, 0))
        
        # Analysis method display
        method_frame = tk.Frame(content_frame, bg=self.colors['surface'])
        method_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            method_frame,
            text="Analysis Method:",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text_primary']
        ).pack(anchor=tk.W)
        
        self.floating_method_label = tk.Label(
            method_frame,
            text="Not analyzed",
            font=('Segoe UI', 10),
            bg=self.colors['surface'],
            fg=self.colors['text_secondary']
        )
        self.floating_method_label.pack(anchor=tk.W, padx=10)
        
        # Action buttons
        actions_frame = tk.Frame(content_frame, bg=self.colors['surface'])
        actions_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Save button
        save_result_btn = tk.Button(
            actions_frame,
            text="💾 Save Results",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['secondary'],
            activeforeground='white',
            relief='flat',
            pady=8,
            command=self.save_result
        )
        save_result_btn.pack(fill=tk.X, pady=(0, 5))
        
        # View details button
        details_btn = tk.Button(
            actions_frame,
            text="🔍 View Detailed Analysis",
            font=('Segoe UI', 10),
            bg=self.colors['surface'],
            fg=self.colors['text_primary'],
            activebackground=self.colors['border'],
            relief='solid',
            bd=1,
            pady=8,
            command=self.show_detailed_analysis
        )
        details_btn.pack(fill=tk.X)

    def show_floating_results(self):
        """Show floating results panel with animation"""
        if not self.results_visible:
            self.results_panel.pack(fill=tk.BOTH, expand=True)
            self.results_visible = True
            self.animate_results_panel_in()

    def hide_floating_results(self):
        """Hide floating results panel with animation"""
        if self.results_visible:
            self.animate_results_panel_out()

    def animate_results_panel_in(self):
        """Animate results panel sliding in"""
        # Start from right side (hidden)
        start_x = 400
        target_x = -20
        
        # Animate sliding in
        def slide_in(current_x):
            if current_x > target_x:
                new_x = current_x - 20
                self.floating_results_container.place(relx=1.0, rely=0.0, anchor='ne', x=new_x, y=20)
                self.master.after(20, lambda: slide_in(new_x))
            else:
                self.floating_results_container.place(relx=1.0, rely=0.0, anchor='ne', x=target_x, y=20)
        
        slide_in(start_x)

    def animate_results_panel_out(self):
        """Animate results panel sliding out"""
        start_x = -20
        target_x = 400
        
        def slide_out(current_x):
            if current_x < target_x:
                new_x = current_x + 20
                self.floating_results_container.place(relx=1.0, rely=0.0, anchor='ne', x=new_x, y=20)
                self.master.after(20, lambda: slide_out(new_x))
            else:
                self.results_panel.pack_forget()
                self.results_visible = False
        
        slide_out(start_x)

    def update_floating_results(self, result):
        """Update floating results panel with new data"""
        # Show the panel if hidden
        self.show_floating_results()
        
        # Update diagnosis
        if result['predicted_label'] == 1:
            diagnosis_text = "⚠️ FRACTURE DETECTED"
            diagnosis_color = self.colors['success']
        else:
            diagnosis_text = "✅ NO FRACTURE DETECTED"
            diagnosis_color = self.colors['primary']
        
        self.floating_diagnosis_label.config(
            text=diagnosis_text,
            fg=diagnosis_color
        )
        
        # Update confidence with animation
        confidence = result['confidence']
        self.floating_confidence_label.config(text=f"{confidence:.1f}%")
        
        # Animate confidence bar
        self.animate_confidence_bar(confidence)
        
        # Update method
        mode = self.current_mode.get()
        if mode == "single":
            method_text = f"Single Model - {result.get('method', 'Unknown').upper()}"
        else:
            method_text = f"Ensemble - {result.get('voting_method', 'Unknown').upper()}"
            
        self.floating_method_label.config(text=method_text)

    def animate_confidence_bar(self, target_confidence):
        """Animate confidence progress bar"""
        def animate_progress(current_value):
            if current_value < target_confidence:
                new_value = min(current_value + 2, target_confidence)
                self.confidence_progress['value'] = new_value
                self.master.after(50, lambda: animate_progress(new_value))
            else:
                self.confidence_progress['value'] = target_confidence
        
        self.confidence_progress['value'] = 0
        animate_progress(0)

    def show_detailed_analysis(self):
        """Show detailed analysis in separate window"""
        if self.prediction_result is None:
            messagebox.showinfo("Info", "No analysis results available!")
            return
        
        # Create detailed window
        detail_window = tk.Toplevel(self.master)
        detail_window.title("Detailed Analysis Results")
        detail_window.geometry("800x600")
        detail_window.configure(bg=self.colors['background'])
        
        # Create visualization
        mode = self.current_mode.get()
        
        try:
            if mode == "single":
                fig = self.detector.visualize_result(self.prediction_result)
            else:
                fig = self.ensemble.visualize_ensemble_result(self.prediction_result)
            
            # Embed in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=detail_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        except Exception as e:
            error_label = tk.Label(
                detail_window,
                text=f"Error creating visualization: {str(e)}",
                font=('Segoe UI', 12),
                bg=self.colors['background'],
                fg=self.colors['success']
            )
            error_label.pack(expand=True)

    def floating_button_enter(self, event):
        """Handle floating button hover enter"""
        self.button_hover = True
        self.draw_gradient_button(hover=True)
        self.float_button_canvas.configure(cursor="hand2")

    def floating_button_leave(self, event):
        """Handle floating button hover leave"""
        self.button_hover = False
        self.draw_gradient_button(hover=False)
        self.float_button_canvas.configure(cursor="")

    def floating_button_click(self, event):
        """Handle floating button click with animation"""
        # Button press animation
        self.animate_button_press()
        
        # Trigger prediction
        self.predict_image()

    def animate_button_press(self):
        """Animate button press effect"""
        # Scale down
        self.float_button_canvas.place(x=7, y=4)
        self.float_button_canvas.configure(width=66, height=66)
        
        # Scale back up after 100ms
        self.master.after(100, self.reset_button_scale)

    def reset_button_scale(self):
        """Reset button to normal scale"""
        self.float_button_canvas.place(x=5, y=2)
        self.float_button_canvas.configure(width=70, height=70)

    def animate_button(self):
        """Create subtle floating animation"""
        if not self.button_hover:
            # Subtle floating effect
            self.animation_frame += 1
            offset_y = 2 + int(2 * np.sin(self.animation_frame * 0.1))
            
            try:
                self.float_button_canvas.place(x=5, y=offset_y)
            except:
                pass  # Handle case where widget is destroyed
        
        # Continue animation
        try:
            self.master.after(50, self.animate_button)
        except:
            pass  # Handle case where master is destroyed

    def update_floating_button_state(self):
        """Update floating button state based on app state"""
        if hasattr(self, 'float_button_canvas'):
            mode = self.current_mode.get()
            
            if mode == "single":
                ready = self.current_image_path and self.model_loaded
            else:  # ensemble
                ready = self.current_image_path and self.ensemble_loaded
            
            if ready:
                # Enable button
                self.float_button_canvas.configure(state='normal')
                if not self.button_hover:
                    self.draw_gradient_button(hover=False)
            else:
                # Disable button - make it gray
                self.float_button_canvas.delete("button_bg")
                self.float_button_canvas.create_oval(
                    0, 0, 70, 70,
                    fill='#CCCCCC', outline='',
                    tags="button_bg"
                )
                self.float_button_canvas.tag_raise("all")

    def on_mode_change(self):
        """Handle mode change between single and ensemble"""
        mode = self.current_mode.get()
        
        if mode == "single":
            # Show single model UI
            self.model_info_frame.pack(fill=tk.X, pady=5)
            self.ensemble_info_frame.pack_forget()
            self.model_btn.pack(fill=tk.X, pady=(0, 5))
            self.ensemble_btn.pack_forget()
            
            # Show single methods
            self.single_methods_frame.pack(fill=tk.X, pady=5)
            self.ensemble_methods_frame.pack_forget()
            
            # Update card title
            for widget in self.model_card.master.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Frame):
                            for label in child.winfo_children():
                                if isinstance(label, tk.Label) and "Model Configuration" in label.cget("text"):
                                    label.config(text="🤖 Single Model Configuration")
                                    break
            
        else:  # ensemble
            # Show ensemble UI
            self.model_info_frame.pack_forget()
            self.ensemble_info_frame.pack(fill=tk.X, pady=5)
            self.model_btn.pack_forget()
            self.ensemble_btn.pack(fill=tk.X, pady=(0, 5))
            
            # Show ensemble methods
            self.single_methods_frame.pack_forget()
            self.ensemble_methods_frame.pack(fill=tk.X, pady=5)
            
            # Update card title
            for widget in self.model_card.master.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Frame):
                            for label in child.winfo_children():
                                if isinstance(label, tk.Label) and ("Model Configuration" in label.cget("text") or "Single Model" in label.cget("text")):
                                    label.config(text="🚀 Ensemble Configuration")
                                    break
        
        # Update predict button state
        self.update_predict_button_state()

    def update_predict_button_state(self):
        """Update predict button state based on current mode"""
        mode = self.current_mode.get()
        
        if mode == "single":
            # Enable if single model loaded and image selected
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

    def initialize_ensemble(self):
        """Initialize ensemble system"""
        self.ensemble_btn.config(state=tk.DISABLED)
        self.status_var.set("Initializing ensemble system...")
        self.ensemble_status_label.config(text="Initializing...", fg=self.colors['accent'])
        self.progress_bar.start(10)
        
        # Run in separate thread
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
        
        if self.ensemble and len(self.ensemble.models) > 0:
            self.ensemble_status_label.config(
                text="✅ Ensemble Ready",
                fg=self.colors['primary']
            )
            self.ensemble_models_label.config(
                text=f"Models: {len(self.ensemble.models)} regions loaded"
            )
            self.status_var.set(f"Ensemble initialized with {len(self.ensemble.models)} models")
        else:
            self.ensemble_status_label.config(
                text="❌ No models found",
                fg=self.colors['success']
            )
            self.ensemble_models_label.config(text="Models: 0")
            self.status_var.set("No ensemble models found")
        
        self.ensemble_btn.config(state=tk.NORMAL)
        self.update_predict_button_state()

    def show_model_menu(self):
        """Show modern model selection menu"""
        menu_window = tk.Toplevel(self.master)
        menu_window.title("Select Model")
        menu_window.geometry("400x300")
        menu_window.configure(bg=self.colors['background'])
        menu_window.resizable(False, False)
        
        # Center the window
        menu_window.transient(self.master)
        menu_window.grab_set()
        
        # Title
        title_label = tk.Label(menu_window,
                             text="🤖 Select AI Model",
                             font=('Segoe UI', 16, 'bold'),
                             bg=self.colors['background'],
                             fg=self.colors['text_primary'])
        title_label.pack(pady=20)
        
        # Model options
        options_frame = tk.Frame(menu_window, bg=self.colors['background'])
        options_frame.pack(fill=tk.BOTH, expand=True, padx=30)
        
        models = [
            ("DenseNet121", "densenet121", "🧠 Advanced neural network architecture"),
            ("ResNet50V2", "resnet50v2", "🔬 Residual network with skip connections")
        ]
        
        for name, code, desc in models:
            btn = tk.Button(options_frame,
                          text=f"{name}\n{desc}",
                          font=('Segoe UI', 11, 'bold'),
                          bg=self.colors['primary'],
                          fg='white',
                          activebackground=self.colors['accent'],
                          activeforeground='white',
                          relief='flat',
                          pady=15,
                          command=lambda c=code: self.select_model(c, menu_window))
            btn.pack(fill=tk.X, pady=5)
        
        # Custom model option
        custom_btn = tk.Button(options_frame,
                             text="📁 Browse Custom Model",
                             font=('Segoe UI', 11),
                             bg=self.colors['surface'],
                             fg=self.colors['text_primary'],
                             activebackground=self.colors['border'],
                             relief='solid',
                             bd=1,
                             pady=10,
                             command=lambda: self.browse_custom_model(menu_window))
        custom_btn.pack(fill=tk.X, pady=(10, 5))

    def select_model(self, model_type, window):
        """Select and load model"""
        window.destroy()
        self.load_model_async(model_type)

    def browse_custom_model(self, window):
        """Browse for custom model file"""
        window.destroy()
        self.browse_model()

    def show_evaluation_menu(self):
        """Show evaluation options menu"""
        mode = self.current_mode.get()
        
        if mode == "single":
            if not self.model_loaded:
                messagebox.showwarning("Warning", "Please load a model first!")
                return
        else:  # ensemble
            if not self.ensemble_loaded:
                messagebox.showwarning("Warning", "Please initialize ensemble system first!")
                return
            
        menu_window = tk.Toplevel(self.master)
        menu_window.title(f"{'Ensemble' if mode == 'ensemble' else 'Model'} Evaluation")
        menu_window.geometry("350x250")
        menu_window.configure(bg=self.colors['background'])
        menu_window.resizable(False, False)
        
        # Center the window
        menu_window.transient(self.master)
        menu_window.grab_set()
        
        # Title
        title_label = tk.Label(menu_window,
                             text=f"📈 {'Ensemble' if mode == 'ensemble' else 'Model'} Evaluation",
                             font=('Segoe UI', 16, 'bold'),
                             bg=self.colors['background'],
                             fg=self.colors['text_primary'])
        title_label.pack(pady=20)
        
        # Options
        options_frame = tk.Frame(menu_window, bg=self.colors['background'])
        options_frame.pack(fill=tk.BOTH, expand=True, padx=30)
        
        if mode == "single":
            methods = [
                ("CNN Evaluation", "cnn"),
                ("Hough Transform Evaluation", "hough"),
                ("Combined Method Evaluation", "combined")
            ]
        else:  # ensemble
            methods = [
                ("Weighted Average Voting", "weighted_average"),
                ("Majority Vote Evaluation", "majority_vote"),
                ("Max Confidence Evaluation", "max_confidence")
            ]
        
        for name, method in methods:
            btn = tk.Button(options_frame,
                          text=name,
                          font=('Segoe UI', 11),
                          bg=self.colors['accent'],
                          fg='white',
                          activebackground=self.colors['secondary'],
                          activeforeground='white',
                          relief='flat',
                          pady=10,
                          command=lambda m=method: self.start_evaluation(m, menu_window))
            btn.pack(fill=tk.X, pady=3)

    def start_evaluation(self, method, window):
        """Start model evaluation"""
        window.destroy()
        self.evaluate_model(method)

    # Model loading methods
    def load_model_async(self, model_type="resnet50v2", region="XR_HAND"):
        """Load model asynchronously with progress indication"""
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set("Loading model...")
        self.model_name_label.config(text="Loading model...", fg=self.colors['accent'])
        self.progress_bar.start(10)
        
        self.load_model_thread = threading.Thread(target=self.load_model, args=(model_type, region))
        self.load_model_thread.daemon = True
        self.load_model_thread.start()

    def load_model(self, model_type="resnet50v2", region="XR_HAND"):
        """Load CNN model from .h5 file"""
        try:
            # Define model paths
            res_model_path = f"C:\\Users\\USER\\Documents\\coze\\models\\res\\{model_type}_{region}_best.h5"
            den_model_path = f"C:\\Users\\USER\\Documents\\coze\\models\\den\\{model_type}_{region}_best.h5"

            # Check for model file existence
            if os.path.exists(res_model_path):
                model_path = res_model_path
            elif os.path.exists(den_model_path):
                model_path = den_model_path
            else:
                raise FileNotFoundError(f"Model file not found: {model_type}_{region}_best.h5")

            self.detector = FractureDetector(model_path)
            self.model_loaded = True
            self.master.after(0, self.update_model_info, model_type, region, model_path)
        except Exception as e:
            self.master.after(0, self.show_error, f"Error loading model: {str(e)}")

    def update_model_info(self, model_type, region, model_path):
        """Update model information in UI"""
        self.progress_bar.stop()
        model_display_name = {"densenet121": "DenseNet121", "resnet50v2": "ResNet50V2"}.get(model_type, model_type)
        
        self.model_name_label.config(
            text=f"✅ {model_display_name}",
            fg=self.colors['primary']
        )
        self.region_label.config(text=f"Region: {region}")
        self.status_var.set(f"Model loaded: {model_display_name} for {region}")
        
        if self.current_image_path:
            self.predict_btn.config(state=tk.NORMAL)
        
        self.update_predict_button_state()

    # Image handling methods
    def browse_image(self):
        """Browse for X-ray image"""
        supported_formats = ('.png', '.jpg', '.jpeg')
        image_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        
        if image_path:
            if not image_path.lower().endswith(supported_formats):
                self.show_error(f"Unsupported file format. Please select .png, .jpg or .jpeg file.")
                return

            if not os.path.exists(image_path):
                self.show_error(f"File not found: {image_path}")
                return

            self.current_image_path = image_path
            self.load_image(image_path)
            if self.model_loaded:
                self.predict_btn.config(state=tk.NORMAL)
            self.reset_prediction_results()
            
            # Update floating button state
            self.update_floating_button_state()
            self.update_predict_button_state()

    def load_image(self, image_path):
        """Load and display image with modern styling"""
        try:
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image file: {image_path}")

            # Display original image
            self.display_image(image, self.canvas_original)

            # Display placeholder for heatmap
            placeholder = np.zeros_like(image)
            self.display_image_with_text(placeholder, self.canvas_heatmap, "Awaiting analysis...")

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
            self.canvas_original.delete("all")
            self.canvas_heatmap.delete("all")
            self.display_placeholder(self.canvas_original, "❌ Error loading image")
            self.display_placeholder(self.canvas_heatmap, "❌ No image")

    def display_image(self, image, canvas):
        """Display image on canvas with modern styling"""
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
        
        # Convert to PIL for better display
        if len(resized_image.shape) == 2:
            pil_image = Image.fromarray(resized_image).convert('RGB')
        else:
            pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        
        # Add subtle border
        bordered_image = Image.new('RGB', (new_width + 4, new_height + 4), self.colors['border'])
        bordered_image.paste(pil_image, (2, 2))
        
        photo = ImageTk.PhotoImage(image=bordered_image)
        canvas.delete("all")
        
        # Center the image
        x_pos = (canvas_width - new_width - 4) // 2
        y_pos = (canvas_height - new_height - 4) // 2
        
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

    def display_placeholder(self, canvas, text):
        """Display placeholder with modern styling"""
        canvas.delete("all")
        canvas.update()
        
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 400)
        
        # Create gradient background effect
        for i in range(0, height, 20):
            alpha = 1 - (i / height) * 0.3
            color_intensity = int(26 + alpha * 10)  # Gradient from dark to slightly lighter
            color = f"#{color_intensity:02x}{color_intensity + 5:02x}{color_intensity + 10:02x}"
            canvas.create_rectangle(0, i, width, i + 20, fill=color, outline="")
        
        # Add icon and text
        canvas.create_text(
            width // 2,
            height // 2 - 20,
            text="🖼️",
            font=('Segoe UI', 24),
            fill='white',
            anchor='center'
        )
        
        canvas.create_text(
            width // 2,
            height // 2 + 20,
            text=text,
            font=('Segoe UI', 12),
            fill='#A0AEC0',
            anchor='center'
        )

    # Prediction methods
    def predict_image(self):
        """Predict fracture with modern progress indication"""
        if not self.current_image_path:
            self.show_error("Please select an image first")
            return
        
        mode = self.current_mode.get()
        
        if mode == "single":
            if not self.model_loaded:
                self.show_error("Please load a model first")
                return
        else:  # ensemble
            if not self.ensemble_loaded:
                self.show_error("Please initialize ensemble system first")
                return
            
        self.predict_btn.config(state=tk.DISABLED)
        self.status_var.set("Analyzing X-ray image...")
        self.progress_bar.start(10)
        
        # Update UI to show analysis in progress
        self.prediction_result_label.config(
            text="🔄 Analyzing...",
            fg=self.colors['accent']
        )
        
        if mode == "single":
            method = self.method_var.get()
            prediction_thread = threading.Thread(
                target=self.predict_in_thread,
                args=(self.current_image_path, method)
            )
        else:  # ensemble
            voting_method = self.voting_var.get()
            prediction_thread = threading.Thread(
                target=self.predict_ensemble_in_thread,
                args=(self.current_image_path, voting_method)
            )
        
        prediction_thread.daemon = True
        prediction_thread.start()

    def predict_in_thread(self, image_path, method):
        """Perform prediction in separate thread"""
        try:
            result = self.detector.predict(image_path, method=method)
            self.prediction_result = result
            self.master.after(0, self.update_prediction_results, result)
        except Exception as e:
            self.master.after(0, self.show_error, f"Prediction error: {str(e)}")
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

    def update_ensemble_results(self, result):
        """Update ensemble prediction results with modern styling"""
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
        confidence = result['confidence']
        self.confidence_label.config(text=f"{confidence:.1f}%")
        
        # Update scores - show ensemble score and top individual scores
        self.cnn_score_label.config(text=f"Ensemble: {result['ensemble_score']:.4f}")
        
        # Show best individual model score
        best_model = max(result['individual_predictions'], key=result['individual_predictions'].get)
        best_score = result['individual_predictions'][best_model]
        self.hough_score_label.config(text=f"Best: {best_score:.4f} ({best_model.split('_')[1]})")
        
        # Display heatmap
        self.display_heatmap(result['image'], result['heatmap'])
        
        # Update status
        diagnosis = "fracture detected" if result['predicted_label'] == 1 else "no fracture"
        self.status_var.set(f"Ensemble analysis complete: {diagnosis} ({confidence:.1f}% confidence)")
        self.predict_btn.config(state=tk.NORMAL)
        
        # Update floating results  
        self.update_floating_results(result)
        
        # Update floating button state
        self.update_floating_button_state()

    def update_prediction_results(self, result):
        """Update prediction results with modern styling"""
        self.progress_bar.stop()
        
        # Update prediction label
        if result['predicted_label'] == 1:
            self.prediction_result_label.config(
                text="⚠️ FRACTURE DETECTED",
                fg=self.colors['success']
            )
        else:
            self.prediction_result_label.config(
                text="✅ NO FRACTURE DETECTED",
                fg=self.colors['primary']
            )
        
        # Update confidence with progress bar effect
        confidence = result['confidence']
        self.confidence_label.config(text=f"{confidence:.1f}%")
        
        # Update scores
        if result['method'] == 'combined':
            self.cnn_score_label.config(text=f"{result['cnn_score']:.4f}")
            self.hough_score_label.config(text=f"{result['hough_score']:.4f}")
        elif result['method'] == 'cnn':
            self.cnn_score_label.config(text=f"{result['score']:.4f}")
            self.hough_score_label.config(text="-")
        elif result['method'] == 'hough':
            self.cnn_score_label.config(text="-")
            self.hough_score_label.config(text=f"{result['score']:.4f}")
        
        # Display heatmap
        self.display_heatmap(result['image'], result['heatmap'])
        
        # Update status
        diagnosis = "fracture detected" if result['predicted_label'] == 1 else "no fracture"
        self.status_var.set(f"Analysis complete: {diagnosis} ({confidence:.1f}% confidence)")
        self.predict_btn.config(state=tk.NORMAL)
        
        # Update floating results
        self.update_floating_results(result)
        
        # Update floating button state
        self.update_floating_button_state()

    def display_heatmap(self, original_image, heatmap):
        """Display heatmap with enhanced visualization"""
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        alpha = 0.6
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        overlaid = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        self.display_image(overlaid, self.canvas_heatmap)

    def reset_prediction_results(self):
        """Reset prediction results with modern styling"""
        self.prediction_result_label.config(
            text="Awaiting analysis...",
            fg=self.colors['text_secondary']
        )
        self.confidence_label.config(text="-")
        self.cnn_score_label.config(text="-")
        self.hough_score_label.config(text="-")
        self.prediction_result = None

    # Enhanced utility methods
    def save_result(self):
        """Save prediction results with modern file dialog"""
        if self.prediction_result is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if save_path:
            try:
                mode = self.current_mode.get()
                
                if mode == "single":
                    # Save single model result
                    fig = self.detector.visualize_result(self.prediction_result)
                else:
                    # Save ensemble result
                    fig = self.ensemble.visualize_ensemble_result(self.prediction_result)
                
                fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                self.status_var.set(f"Results saved: {os.path.basename(save_path)}")
                messagebox.showinfo("Success", f"Results saved successfully!\n{save_path}")
            except Exception as e:
                self.show_error(f"Error saving results: {str(e)}")

    def browse_model(self):
        """Browse for custom model with enhanced dialog"""
        model_path = filedialog.askopenfilename(
            title="Select Custom Model",
            filetypes=[("Model files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            self.predict_btn.config(state=tk.DISABLED)
            self.status_var.set("Loading custom model...")
            self.model_name_label.config(text="Loading custom model...", fg=self.colors['accent'])
            self.progress_bar.start(10)
            
            self.load_model_thread = threading.Thread(target=self.load_custom_model, args=(model_path,))
            self.load_model_thread.daemon = True
            self.load_model_thread.start()

    def load_custom_model(self, model_path):
        """Load custom model with error handling"""
        try:
            self.detector = FractureDetector(model_path)
            self.model_loaded = True
            filename = os.path.basename(model_path)
            parts = filename.split('_')
            model_type = parts[0] if len(parts) >= 3 else "Custom"
            region = '_'.join(parts[1:-1]) if len(parts) >= 3 else "Unknown"
            self.master.after(0, self.update_model_info, model_type, region, model_path)
        except Exception as e:
            self.master.after(0, self.show_error, f"Error loading custom model: {str(e)}")

    def evaluate_model(self, method):
        """Enhanced model evaluation with modern progress"""
        mode = self.current_mode.get()
        
        if mode == "single":
            if not self.model_loaded:
                messagebox.showwarning("Warning", "Please load a model first!")
                return
        else:  # ensemble
            if not self.ensemble_loaded:
                messagebox.showwarning("Warning", "Please initialize ensemble system first!")
                return
            
        test_dir = filedialog.askdirectory(
            title="Select Test Directory (containing 'normal' and 'abnormal' folders)"
        )
        
        if not test_dir:
            return
            
        # Validate directory structure
        normal_dir = os.path.join(test_dir, 'normal')
        abnormal_dir = os.path.join(test_dir, 'abnormal')
        
        if not os.path.exists(normal_dir) or not os.path.exists(abnormal_dir):
            messagebox.showerror(
                "Invalid Directory",
                "Selected directory must contain 'normal' and 'abnormal' subdirectories!"
            )
            return
        
        self.predict_btn.config(state=tk.DISABLED)
        
        if mode == "single":
            self.status_var.set(f"Evaluating single model with {method} method...")
        else:
            self.status_var.set(f"Evaluating ensemble with {method} voting...")
            
        self.progress_bar.start(10)
        
        eval_thread = threading.Thread(target=self.evaluate_in_thread, args=(test_dir, method))
        eval_thread.daemon = True
        eval_thread.start()

    def evaluate_in_thread(self, test_dir, method):
        """Perform evaluation in separate thread"""
        try:
            mode = self.current_mode.get()
            
            if mode == "single":
                # Single model evaluation
                output_dir = f"evaluation_results_{method}"
                results = self.detector.evaluate_on_directory(
                    test_dir, 
                    method=method, 
                    visualize=True, 
                    output_dir=output_dir
                )
            else:
                # Ensemble evaluation
                # Get all images from test directory
                normal_dir = os.path.join(test_dir, 'normal')
                abnormal_dir = os.path.join(test_dir, 'abnormal')
                
                normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                abnormal_images = [os.path.join(abnormal_dir, f) for f in os.listdir(abnormal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                all_images = normal_images + abnormal_images
                np.random.shuffle(all_images)
                
                output_dir = f"ensemble_evaluation_{method}"
                results = self.ensemble.evaluate_ensemble(
                    all_images,
                    voting_methods=[method],
                    visualize=True,
                    output_dir=output_dir
                )
                
                # Extract single method result
                results = results[method]
            
            self.master.after(0, self.show_evaluation_results, results, output_dir, method)
        except Exception as e:
            self.master.after(0, self.show_error, f"Evaluation error: {str(e)}")
            self.master.after(0, lambda: self.predict_btn.config(state=tk.NORMAL))

    def show_evaluation_results(self, results, output_dir, method):
        """Show evaluation results in modern window"""
        self.progress_bar.stop()
        
        mode = self.current_mode.get()
        title_prefix = "Ensemble" if mode == "ensemble" else "Single Model"
        
        # Create modern results window
        result_window = tk.Toplevel(self.master)
        result_window.title(f"{title_prefix} Evaluation Results - {method.upper()}")
        result_window.geometry("900x700")
        result_window.configure(bg=self.colors['background'])
        
        # Header
        header_frame = tk.Frame(result_window, bg=self.colors['primary'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text=f"📊 {title_prefix} Evaluation - {method.upper()}",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Create notebook for results
        notebook = ttk.Notebook(result_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Metrics tab
        metrics_frame = tk.Frame(notebook, bg=self.colors['surface'])
        notebook.add(metrics_frame, text="📈 Metrics")
        
        # Create metrics display
        self.create_metrics_display(metrics_frame, results)
        
        # Confusion matrix tab (if available)
        if 'confusion_matrix' in results:
            cm_frame = tk.Frame(notebook, bg=self.colors['surface'])
            notebook.add(cm_frame, text="🔄 Confusion Matrix")
            self.create_confusion_matrix_display(cm_frame, results)
        
        # Classification report tab (if available)
        if 'classification_report' in results:
            report_frame = tk.Frame(notebook, bg=self.colors['surface'])
            notebook.add(report_frame, text="📋 Detailed Report")
            self.create_report_display(report_frame, results, output_dir)
        
        # Ensemble details tab (for ensemble mode)
        if mode == "ensemble" and 'detailed_results' in results:
            ensemble_frame = tk.Frame(notebook, bg=self.colors['surface'])
            notebook.add(ensemble_frame, text="🚀 Ensemble Details")
            self.create_ensemble_details_display(ensemble_frame, results)
        
        self.status_var.set(f"{title_prefix} evaluation complete: {method}")
        self.predict_btn.config(state=tk.NORMAL)

    def create_ensemble_details_display(self, parent, results):
        """Create ensemble-specific details display"""
        details_container = tk.Frame(parent, bg=self.colors['surface'])
        details_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Ensemble summary
        summary_frame = tk.LabelFrame(details_container, text="Ensemble Summary", 
                                    font=('Segoe UI', 12, 'bold'), 
                                    bg=self.colors['surface'])
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        summary_text = f"""
        Total Models: {len(self.ensemble.models)}
        Voting Method: {results.get('detailed_results', [{}])[0].get('voting_method', 'N/A')}
        Total Images Processed: {len(results.get('detailed_results', []))}
        """
        
        summary_label = tk.Label(summary_frame, text=summary_text,
                               font=('Segoe UI', 10),
                               bg=self.colors['surface'],
                               justify=tk.LEFT)
        summary_label.pack(anchor=tk.W, padx=10, pady=10)
        
        # Individual results
        results_frame = tk.LabelFrame(details_container, text="Individual Results",
                                    font=('Segoe UI', 12, 'bold'),
                                    bg=self.colors['surface'])
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable text widget
        text_frame = tk.Frame(results_frame, bg=self.colors['surface'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg=self.colors['surface'],
            fg=self.colors['text_primary'],
            relief='flat'
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Add detailed results
        if 'detailed_results' in results:
            for i, result in enumerate(results['detailed_results'][:20]):  # Show first 20 results
                filename = os.path.basename(result['image_path'])
                status = "✓" if result['predicted_label'] == result['true_label'] else "✗"
                
                detail_text = f"{i+1:3d}. {status} {filename}\n"
                detail_text += f"     Prediction: {'FRACTURE' if result['predicted_label'] == 1 else 'NORMAL'} "
                detail_text += f"({result['confidence']:.1f}%)\n"
                detail_text += f"     Ensemble Score: {result['ensemble_score']:.4f}\n"
                detail_text += f"     Individual Scores: "
                
                # Show top 3 individual model scores
                sorted_scores = sorted(result['individual_predictions'].items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                score_text = ", ".join([f"{k.split('_')[1]}: {v:.3f}" for k, v in sorted_scores])
                detail_text += score_text + "\n\n"
                
                text_widget.insert(tk.END, detail_text)
        
        text_widget.config(state=tk.DISABLED)

    def create_metrics_display(self, parent, results):
        """Create modern metrics display"""
        # Metrics grid
        metrics_container = tk.Frame(parent, bg=self.colors['surface'])
        metrics_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        metrics = [
            ("Accuracy", results['accuracy'], self.colors['primary']),
            ("Precision", results['precision'], self.colors['accent']),
            ("Recall", results['recall'], self.colors['secondary']),
            ("F1 Score", results['f1_score'], self.colors['success'])
        ]
        
        for i, (name, value, color) in enumerate(metrics):
            # Create metric card
            metric_card = tk.Frame(metrics_container, bg=color, relief='flat')
            metric_card.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='nsew')
            
            # Configure grid weights
            metrics_container.grid_rowconfigure(i//2, weight=1)
            metrics_container.grid_columnconfigure(i%2, weight=1)
            
            # Metric content
            content_frame = tk.Frame(metric_card, bg=color)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            value_label = tk.Label(
                content_frame,
                text=f"{value:.3f}",
                font=('Segoe UI', 24, 'bold'),
                bg=color,
                fg='white'
            )
            value_label.pack()
            
            name_label = tk.Label(
                content_frame,
                text=name,
                font=('Segoe UI', 12),
                bg=color,
                fg='white'
            )
            name_label.pack()

    def create_confusion_matrix_display(self, parent, results):
        """Create confusion matrix visualization"""
        cm_container = tk.Frame(parent, bg=self.colors['surface'])
        cm_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(self.colors['surface'])
        
        cm = results['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Set tick labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Fracture'])
        ax.set_yticklabels(['Normal', 'Fracture'])
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=cm_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_report_display(self, parent, results, output_dir):
        """Create detailed report display"""
        report_container = tk.Frame(parent, bg=self.colors['surface'])
        report_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Report text
        text_frame = tk.Frame(report_container, bg=self.colors['surface'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg=self.colors['surface'],
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=10
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Insert report
        text_widget.insert(tk.END, results['report'])
        text_widget.config(state=tk.DISABLED)
        
        # Action buttons
        button_frame = tk.Frame(report_container, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        open_folder_btn = tk.Button(
            button_frame,
            text="📁 Open Results Folder",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['secondary'],
            relief='flat',
            pady=8,
            command=lambda: self.open_folder(output_dir)
        )
        open_folder_btn.pack(side=tk.LEFT, padx=(0, 10))

    def open_folder(self, folder_path):
        """Open folder in file explorer"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(os.path.abspath(folder_path))
            else:  # Linux/Mac
                os.system(f'xdg-open {folder_path}')
        except Exception as e:
            self.show_error(f"Cannot open folder: {str(e)}")

    def show_error(self, message):
        """Show modern error dialog"""
        messagebox.showerror("Error", message)
        self.progress_bar.stop()

    def show_info(self, title, message):
        """Show modern info dialog"""
        messagebox.showinfo(title, message)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Set window icon (optional)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    app = ModernFractureDetectionApp(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f'+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()
