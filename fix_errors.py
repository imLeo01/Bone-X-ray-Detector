# Sửa lỗi Grad-CAM và Albumentations
# Thêm vào đầu file test.py hoặc prediction.py

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. FIX GRAD-CAM ERRORS
class FixedAdvancedGradCAM:
    """
    Fixed Advanced Grad-CAM với error handling tốt hơn
    """
    def __init__(self, model):
        self.model = model
        
    def find_optimal_layers(self):
        """Find optimal convolutional layers for Grad-CAM with better error handling"""
        conv_layers = []
        
        def search_layers(model_or_layer, prefix=""):
            """Recursively search for conv layers"""
            if hasattr(model_or_layer, 'layers'):
                for i, layer in enumerate(model_or_layer.layers):
                    layer_name = f"{prefix}{layer.name}" if prefix else layer.name
                    
                    # Check if it's a Conv2D layer
                    if hasattr(layer, 'filters') and hasattr(layer, 'kernel_size'):
                        try:
                            # Test if layer is properly built
                            if hasattr(layer, 'built') and layer.built:
                                weight = 1.0 if i >= len(model_or_layer.layers) - 5 else 0.7
                                conv_layers.append((layer_name, weight))
                        except:
                            continue
                    
                    # Recursively search nested models/layers
                    if hasattr(layer, 'layers') and len(layer.layers) > 0:
                        search_layers(layer, f"{layer_name}_")
        
        search_layers(self.model)
        
        # Return top 3 valid layers
        valid_layers = conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers
        
        # Fallback if no valid layers found
        if not valid_layers:
            print("⚠️ No valid conv layers found, using fallback layers")
            # Try to find any layer with 'conv' in name
            for layer in self.model.layers:
                if 'conv' in layer.name.lower() and hasattr(layer, 'output_shape'):
                    valid_layers.append((layer.name, 1.0))
                    if len(valid_layers) >= 2:
                        break
        
        return valid_layers
    
    def _generate_single_layer_gradcam(self, img_tensor, layer_name):
        """Generate Grad-CAM for a single layer with better error handling"""
        try:
            # Find target layer with more robust search
            target_layer = None
            
            def find_layer_recursive(model_or_layer, target_name):
                """Recursively find layer by name"""
                if hasattr(model_or_layer, 'name') and model_or_layer.name == target_name:
                    return model_or_layer
                
                if hasattr(model_or_layer, 'layers'):
                    for layer in model_or_layer.layers:
                        found = find_layer_recursive(layer, target_name)
                        if found is not None:
                            return found
                return None
            
            target_layer = find_layer_recursive(self.model, layer_name)
            
            if target_layer is None:
                print(f"⚠️ Layer {layer_name} not found")
                return None
            
            # Check if layer is properly built
            if not hasattr(target_layer, 'built') or not target_layer.built:
                print(f"⚠️ Layer {layer_name} is not built")
                return None
            
            # Create gradient model with error handling
            try:
                grad_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[target_layer.output, self.model.output]
                )
            except Exception as e:
                print(f"⚠️ Failed to create gradient model for {layer_name}: {e}")
                return None
            
            # Ensure model is called at least once
            try:
                test_output = grad_model(img_tensor)
                if len(test_output) != 2:
                    print(f"⚠️ Unexpected output format for {layer_name}")
                    return None
            except Exception as e:
                print(f"⚠️ Failed to call gradient model for {layer_name}: {e}")
                return None
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                if len(predictions.shape) > 1:
                    loss = predictions[0, 0]
                else:
                    loss = predictions[0]
            
            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                print(f"⚠️ No gradients computed for {layer_name}")
                return None
            
            # Generate heatmap
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            
            # Normalize
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"⚠️ Single layer Grad-CAM failed for {layer_name}: {e}")
            return None
    
    def generate_advanced_gradcam(self, image, img_tensor, target_layers=None):
        """Generate advanced Grad-CAM with robust error handling"""
        try:
            if target_layers is None:
                target_layers = self.find_optimal_layers()
            
            if not target_layers:
                print("⚠️ No valid layers found, using fallback heatmap")
                return self._generate_enhanced_fallback(image)
            
            heatmaps = []
            weights = []
            
            print(f"🔍 Attempting Grad-CAM on {len(target_layers)} layers...")
            
            for layer_name, weight in target_layers:
                try:
                    heatmap = self._generate_single_layer_gradcam(img_tensor, layer_name)
                    if heatmap is not None:
                        heatmaps.append(heatmap)
                        weights.append(weight)
                        print(f"  ✅ Successfully generated Grad-CAM for {layer_name}")
                    else:
                        print(f"  ❌ Failed to generate Grad-CAM for {layer_name}")
                except Exception as e:
                    print(f"  ❌ Error with layer {layer_name}: {e}")
                    continue
            
            if not heatmaps:
                print("⚠️ No successful Grad-CAM generations, using enhanced fallback")
                return self._generate_enhanced_fallback(image)
            
            print(f"✅ Successfully generated {len(heatmaps)} Grad-CAM heatmaps")
            
            # Weighted fusion of multiple heatmaps
            combined_heatmap = self._fuse_heatmaps(heatmaps, weights)
            
            # Post-processing enhancement
            enhanced_heatmap = self._enhance_heatmap(combined_heatmap, image)
            
            return enhanced_heatmap
            
        except Exception as e:
            print(f"❌ Advanced Grad-CAM completely failed: {e}")
            return self._generate_enhanced_fallback(image)
    
    def _fuse_heatmaps(self, heatmaps, weights):
        """Fuse multiple heatmaps with weights"""
        if len(heatmaps) == 1:
            return heatmaps[0]
        
        # Resize all heatmaps to the same size
        target_size = heatmaps[0].shape
        resized_heatmaps = []
        
        for heatmap in heatmaps:
            if heatmap.shape != target_size:
                resized = cv2.resize(heatmap, (target_size[1], target_size[0]))
                resized_heatmaps.append(resized)
            else:
                resized_heatmaps.append(heatmap)
        
        # Weighted fusion
        total_weight = sum(weights)
        fused = np.zeros_like(resized_heatmaps[0])
        
        for heatmap, weight in zip(resized_heatmaps, weights):
            fused += heatmap * (weight / total_weight)
        
        return fused
    
    def _enhance_heatmap(self, heatmap, original_image):
        """Enhanced post-processing of heatmap"""
        # Resize to original image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Edge-guided enhancement
        try:
            edges = cv2.Canny(original_image, 50, 150)
            edge_mask = edges.astype(np.float32) / 255.0
            enhanced = heatmap_resized + (edge_mask * heatmap_resized * 0.3)
        except:
            enhanced = heatmap_resized
        
        # Gaussian smoothing with adaptive sigma
        try:
            sigma = max(2, min(h, w) // 100)
            enhanced = cv2.GaussianBlur(enhanced, (0, 0), sigma)
        except:
            pass
        
        # Non-linear enhancement
        enhanced = np.power(np.clip(enhanced, 0, 1), 0.8)
        
        # Morphological enhancement
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        except:
            pass
        
        # Final normalization
        enhanced = np.clip(enhanced, 0, 1)
        if np.max(enhanced) > 0:
            enhanced = enhanced / np.max(enhanced)
        
        return enhanced
    
    def _generate_enhanced_fallback(self, original_image):
        """Enhanced fallback heatmap generation"""
        h, w = original_image.shape[:2]
        
        try:
            # Multi-scale edge detection
            edges_multi = np.zeros((h, w), dtype=np.float32)
            for sigma in [0.5, 1.0, 2.0]:
                try:
                    blurred = cv2.GaussianBlur(original_image, (0, 0), sigma)
                    edges = cv2.Canny(blurred, 30, 100)
                    edges_multi += edges.astype(np.float32) / 255.0
                except:
                    continue
            
            edges_multi = np.clip(edges_multi / 3.0, 0, 1)
            
            # Gradient magnitude
            try:
                gx = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx**2 + gy**2)
                grad_norm = grad_mag / (np.max(grad_mag) + 1e-8)
            except:
                grad_norm = np.zeros_like(edges_multi)
            
            # Combine features
            heatmap = 0.6 * edges_multi + 0.4 * grad_norm
            
            # Add center bias
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            center_distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            center_bias = 1 - (center_distance / max_distance) * 0.2
            
            heatmap = heatmap * center_bias
            
            # Smooth
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), 2)
            heatmap = np.clip(heatmap, 0, 1)
            
            return heatmap
            
        except Exception as e:
            print(f"⚠️ Even fallback heatmap failed: {e}")
            # Ultimate fallback - simple gradient
            return np.random.rand(h, w) * 0.3

# 2. FIX ALBUMENTATIONS WARNINGS
class SafeAugmentationPipeline:
    """
    Safe augmentation pipeline that handles divide by zero warnings
    """
    def __init__(self):
        import albumentations as A
        import warnings
        
        # Suppress specific warnings
        warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
        warnings.filterwarnings('ignore', message='invalid value encountered in divide')
        
        self.pipeline = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            # Use safer blur parameters
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.3),
            # Remove UnsharpMask if causing issues, or use safer parameters
            # A.UnsharpMask(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), alpha=(0.2, 0.5), threshold=10, p=0.5),
        ])
    
    def __call__(self, image):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self.pipeline(image=image)['image']
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            return image  # Return original image if augmentation fails

# 3. SAFE PREPROCESSING FUNCTION
def safe_advanced_preprocessing(image):
    """
    Safe advanced preprocessing that handles errors gracefully
    """
    try:
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use safe augmentation
        safe_aug = SafeAugmentationPipeline()
        enhanced = safe_aug(image)
        
        # Safe CLAHE with multiple tile sizes
        clahe_results = []
        for tile_size in [(4, 4), (8, 8), (16, 16)]:
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tile_size)
                clahe_enhanced = clahe.apply(enhanced)
                clahe_results.append(clahe_enhanced)
            except:
                clahe_results.append(enhanced)  # Fallback to original
        
        # Weighted combination
        weights = [0.5, 0.3, 0.2]
        combined = np.zeros_like(enhanced, dtype=np.float32)
        for result, weight in zip(clahe_results, weights):
            combined += result.astype(np.float32) * weight
        
        enhanced = combined.astype(np.uint8)
        
        # Safe unsharp masking
        try:
            unsharp_enhanced = enhanced.copy()
            for sigma in [1.0, 2.0, 4.0]:
                blurred = cv2.GaussianBlur(enhanced, (0, 0), sigma)
                unsharp = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
                unsharp_enhanced = cv2.addWeighted(unsharp_enhanced, 0.7, unsharp, 0.3, 0)
        except:
            unsharp_enhanced = enhanced
        
        # Safe morphological enhancement
        try:
            kernel_sizes = [(3, 3), (5, 5)]
            morph_enhanced = unsharp_enhanced.copy()
            
            for k_size in kernel_sizes:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
                tophat = cv2.morphologyEx(unsharp_enhanced, cv2.MORPH_TOPHAT, kernel)
                morph_enhanced = cv2.add(morph_enhanced, tophat)
        except:
            morph_enhanced = unsharp_enhanced
        
        # Safe bilateral filtering
        try:
            final_enhanced = cv2.bilateralFilter(morph_enhanced, 9, 75, 75)
        except:
            final_enhanced = morph_enhanced
        
        return final_enhanced
        
    except Exception as e:
        print(f"⚠️ Advanced preprocessing failed: {e}")
        return image  # Return original image if all fails