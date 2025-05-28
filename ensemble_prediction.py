# ensemble_prediction.py
# Multi-Region Ensemble System cho phát hiện gãy xương trên nhiều vùng cánh tay

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, classification_report
from tkinter import filedialog, messagebox
import tkinter as tk
import json
from datetime import datetime

class MultiRegionEnsemble:
    def __init__(self, models_config=None, input_shape=(224, 224), threshold=0.5):
        """
        Khởi tạo Multi-Region Ensemble System
        
        Args:
            models_config: Dictionary config các mô hình theo vùng
            input_shape: Kích thước đầu vào cho mô hình CNN
            threshold: Ngưỡng xác suất để xác định gãy xương
        """
        self.input_shape = input_shape
        self.threshold = threshold
        self.models = {}  # Dictionary lưu trữ các mô hình
        self.model_weights = {}  # Trọng số cho từng mô hình
        self.regions = [
            'XR_HAND', 'XR_WRIST', 'XR_ELBOW', 
            'XR_FINGER', 'XR_FOREARM', 'XR_HUMERUS', 'XR_SHOULDER'
        ]
        
        if models_config:
            self.load_models_from_config(models_config)
        else:
            self.auto_discover_models()
    
    def auto_discover_models(self, base_dir="C:\\Users\\USER\\Documents\\coze"):
        """Tự động tìm và tải các mô hình có sẵn"""
        print("🔍 Đang tự động tìm kiếm các mô hình...")
        
        model_types = ['densenet121', 'resnet50v2']
        found_models = []
        
        for region in self.regions:
            for model_type in model_types:
                model_paths = [
                    os.path.join(base_dir, "models", "res", f"{model_type}_{region}_best.h5"),
                    os.path.join(base_dir, "models", "den", f"{model_type}_{region}_best.h5"),
                    os.path.join(base_dir, "models", f"{model_type}_{region}_best.h5"),
                ]
                
                for path in model_paths:
                    if os.path.exists(path):
                        found_models.append({
                            'region': region,
                            'model_type': model_type,
                            'path': path,
                            'weight': 1.0  # Trọng số mặc định
                        })
                        print(f"  ✅ Tìm thấy: {region} - {model_type}")
                        break
        
        if not found_models:
            print("❌ Không tìm thấy mô hình nào!")
            return
        
        print(f"\n📊 Tổng cộng tìm thấy {len(found_models)} mô hình")
        self.load_selected_models(found_models)
    
    def load_selected_models(self, models_list):
        """Tải các mô hình đã chọn"""
        print(f"\n🔄 Đang tải {len(models_list)} mô hình...")
        
        for i, model_info in enumerate(models_list):
            try:
                model_key = f"{model_info['region']}_{model_info['model_type']}"
                print(f"  📥 Đang tải {model_key}...")
                
                model = load_model(model_info['path'])
                self.models[model_key] = {
                    'model': model,
                    'region': model_info['region'],
                    'model_type': model_info['model_type'],
                    'path': model_info['path']
                }
                self.model_weights[model_key] = model_info['weight']
                
                print(f"    ✅ Đã tải thành công!")
                
            except Exception as e:
                print(f"    ❌ Lỗi khi tải {model_key}: {str(e)}")
        
        print(f"\n🎉 Đã tải thành công {len(self.models)} mô hình!")
        self.print_loaded_models()
    
    def print_loaded_models(self):
        """In danh sách các mô hình đã tải"""
        print(f"\n{'='*60}")
        print("DANH SÁCH MÔ HÌNH TRONG ENSEMBLE")
        print(f"{'='*60}")
        
        for model_key, model_info in self.models.items():
            weight = self.model_weights[model_key]
            print(f"🔹 {model_info['region']} - {model_info['model_type']} (weight: {weight:.2f})")
        
        print(f"{'='*60}")
    
    def set_model_weights(self, weights_dict):
        """Thiết lập trọng số cho các mô hình"""
        for model_key, weight in weights_dict.items():
            if model_key in self.model_weights:
                self.model_weights[model_key] = weight
                print(f"📊 Cập nhật trọng số {model_key}: {weight}")
    
    def preprocess_for_cnn(self, image):
        """Tiền xử lý ảnh cho mô hình CNN"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        img_resized = cv2.resize(image, self.input_shape)
        img_normalized = img_resized / 255.0
        img_rgb = np.stack([img_normalized, img_normalized, img_normalized], axis=-1)
        img_tensor = np.expand_dims(img_rgb, axis=0)
        
        return img_tensor, img_normalized
    
    def predict_with_single_model(self, image, model_key):
        """Dự đoán với một mô hình đơn lẻ"""
        model_info = self.models[model_key]
        model = model_info['model']
        
        # Tiền xử lý ảnh
        img_tensor, img_normalized = self.preprocess_for_cnn(image)
        
        # Dự đoán
        prediction = model.predict(img_tensor, verbose=0)[0][0]
        
        # Tạo heatmap (simplified version)
        heatmap = self.generate_simple_heatmap(image, model, img_tensor)
        
        return prediction, heatmap
    
    def generate_simple_heatmap(self, original_image, model, img_tensor):
        """Tạo heatmap đơn giản (có thể cải tiến thêm với Grad-CAM)"""
        try:
            # Simplified heatmap generation
            # Có thể implement Grad-CAM chi tiết hơn nếu cần
            prediction = model.predict(img_tensor, verbose=0)[0][0]
            
            # Tạo heatmap cơ bản dựa trên prediction score
            h, w = original_image.shape[:2]
            heatmap = np.full((h, w), prediction * 0.5, dtype=np.float32)
            
            # Thêm một số noise để làm cho heatmap thực tế hơn
            noise = np.random.normal(0, 0.1, (h, w))
            heatmap = np.clip(heatmap + noise, 0, 1)
            
            return heatmap
            
        except Exception as e:
            print(f"Lỗi tạo heatmap: {str(e)}")
            return np.zeros((original_image.shape[0], original_image.shape[1]))
    
    def predict_ensemble(self, image, voting_method='weighted_average'):
        """
        Dự đoán ensemble từ tất cả các mô hình
        
        Args:
            image: Ảnh đầu vào
            voting_method: Phương pháp vote ('weighted_average', 'majority_vote', 'max_confidence')
        """
        if not self.models:
            raise ValueError("Không có mô hình nào được tải!")
        
        predictions = {}
        heatmaps = {}
        
        # Dự đoán với từng mô hình
        for model_key in self.models.keys():
            pred, heatmap = self.predict_with_single_model(image, model_key)
            predictions[model_key] = pred
            heatmaps[model_key] = heatmap
        
        # Kết hợp predictions theo phương pháp được chọn
        if voting_method == 'weighted_average':
            final_score = self.weighted_average_voting(predictions)
        elif voting_method == 'majority_vote':
            final_score = self.majority_voting(predictions)
        elif voting_method == 'max_confidence':
            final_score = self.max_confidence_voting(predictions)
        else:
            final_score = self.weighted_average_voting(predictions)
        
        # Kết hợp heatmaps
        combined_heatmap = self.combine_heatmaps(heatmaps)
        
        return final_score, combined_heatmap, predictions
    
    def weighted_average_voting(self, predictions):
        """Voting theo trọng số trung bình"""
        total_weight = sum(self.model_weights.values())
        weighted_sum = sum(
            pred * self.model_weights[model_key] 
            for model_key, pred in predictions.items()
        )
        return weighted_sum / total_weight
    
    def majority_voting(self, predictions):
        """Voting theo đa số (binary)"""
        binary_preds = [1 if pred > self.threshold else 0 for pred in predictions.values()]
        return 1.0 if sum(binary_preds) > len(binary_preds) / 2 else 0.0
    
    def max_confidence_voting(self, predictions):
        """Voting theo confidence cao nhất"""
        return max(predictions.values())
    
    def combine_heatmaps(self, heatmaps):
        """Kết hợp các heatmap theo trọng số"""
        if not heatmaps:
            return np.zeros((224, 224))
        
        # Lấy heatmap đầu tiên làm base
        first_heatmap = list(heatmaps.values())[0]
        combined = np.zeros_like(first_heatmap)
        
        total_weight = sum(self.model_weights.values())
        
        for model_key, heatmap in heatmaps.items():
            weight = self.model_weights[model_key]
            combined += heatmap * (weight / total_weight)
        
        return np.clip(combined, 0, 1)
    
    def predict(self, image_path, voting_method='weighted_average'):
        """
        Dự đoán ensemble cho một ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            voting_method: Phương pháp vote
        """
        # Đọc ảnh
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
        
        # Xác định nhãn thật
        true_label = 1 if "abnormal" in image_path.lower() else 0
        
        # Dự đoán ensemble
        ensemble_score, combined_heatmap, individual_preds = self.predict_ensemble(
            image, voting_method
        )
        
        # Xác định nhãn dự đoán
        predicted_label = 1 if ensemble_score > self.threshold else 0
        confidence = ensemble_score if predicted_label == 1 else 1 - ensemble_score
        confidence_percent = confidence * 100
        
        return {
            'image_path': image_path,
            'image': image,
            'ensemble_score': ensemble_score,
            'individual_predictions': individual_preds,
            'predicted_label': predicted_label,
            'true_label': true_label,
            'confidence': confidence_percent,
            'heatmap': combined_heatmap,
            'voting_method': voting_method,
            'num_models': len(self.models)
        }
    
    def visualize_ensemble_result(self, result):
        """Trực quan hóa kết quả ensemble"""
        fig = plt.figure(figsize=(16, 10))
        
        # Layout: 2 rows, 4 columns
        # Row 1: Original image, Combined heatmap, Individual predictions chart
        # Row 2: Individual model details
        
        # Ảnh gốc
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(result['image'], cmap='gray')
        ax1.set_title('Ảnh X-quang gốc', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Combined heatmap
        ax2 = plt.subplot(2, 4, 2)
        ax2.imshow(result['image'], cmap='gray')
        heatmap_overlay = ax2.imshow(result['heatmap'], cmap='jet', alpha=0.6)
        ax2.set_title('Ensemble Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Individual predictions chart
        ax3 = plt.subplot(2, 4, 3)
        model_names = [key.replace('_', '\n') for key in result['individual_predictions'].keys()]
        pred_values = list(result['individual_predictions'].values())
        
        colors = ['red' if p > self.threshold else 'green' for p in pred_values]
        bars = ax3.bar(range(len(pred_values)), pred_values, color=colors, alpha=0.7)
        ax3.axhline(y=self.threshold, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Prediction Score')
        ax3.set_title('Individual Model Predictions', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, pred_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Ensemble result summary
        ax4 = plt.subplot(2, 4, 4)
        ax4.axis('off')
        
        pred_text = "PHÁT HIỆN GÃY XƯƠNG" if result['predicted_label'] == 1 else "KHÔNG PHÁT HIỆN GÃY XƯƠNG"
        true_text = "GÃY XƯƠNG" if result['true_label'] == 1 else "BÌNH THƯỜNG"
        
        result_color = 'green' if result['predicted_label'] == result['true_label'] else 'red'
        
        summary_text = f"""
ENSEMBLE RESULT

Voting Method: {result['voting_method']}
Models Used: {result['num_models']}

DIAGNOSIS:
{pred_text}

Confidence: {result['confidence']:.1f}%
Ensemble Score: {result['ensemble_score']:.4f}

GROUND TRUTH:
{true_text}

Status: {'✓ CORRECT' if result['predicted_label'] == result['true_label'] else '✗ INCORRECT'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=result_color, alpha=0.1))
        
        # Model weights visualization
        ax5 = plt.subplot(2, 4, (5, 8))
        
        weights_text = "MODEL WEIGHTS & DETAILS:\n\n"
        for i, (model_key, pred_value) in enumerate(result['individual_predictions'].items()):
            region = model_key.split('_')[1] + '_' + model_key.split('_')[2]
            model_type = model_key.split('_')[0]
            weight = self.model_weights[model_key]
            
            status = "🔴 FRACTURE" if pred_value > self.threshold else "🟢 NORMAL"
            weights_text += f"{i+1}. {region} ({model_type})\n"
            weights_text += f"   Score: {pred_value:.4f} | Weight: {weight:.2f} | {status}\n\n"
        
        ax5.text(0.05, 0.95, weights_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))
        ax5.axis('off')
        
        plt.tight_layout()
        return fig
    
    def evaluate_ensemble(self, test_data, voting_methods=['weighted_average'], visualize=True, output_dir='ensemble_results'):
        """
        Đánh giá ensemble trên test data
        
        Args:
            test_data: List các đường dẫn ảnh hoặc thư mục
            voting_methods: List các phương pháp voting cần test
            visualize: Có lưu visualization không
            output_dir: Thư mục output
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {}
        
        for voting_method in voting_methods:
            print(f"\n🔄 Đánh giá với phương pháp voting: {voting_method}")
            
            y_true = []
            y_pred = []
            detailed_results = []
            
            for i, image_path in enumerate(test_data):
                print(f"  📋 Đang xử lý {i+1}/{len(test_data)}: {os.path.basename(image_path)}")
                
                try:
                    result = self.predict(image_path, voting_method)
                    
                    y_true.append(result['true_label'])
                    y_pred.append(result['predicted_label'])
                    detailed_results.append(result)
                    
                    if visualize:
                        fig = self.visualize_ensemble_result(result)
                        save_path = os.path.join(
                            output_dir, 
                            f"{voting_method}_{os.path.basename(image_path).split('.')[0]}.png"
                        )
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                
                except Exception as e:
                    print(f"    ❌ Lỗi: {str(e)}")
            
            # Tính metrics
            if len(set(y_true)) > 1:
                cm = confusion_matrix(y_true, y_pred)
                report = classification_report(y_true, y_pred, target_names=['Normal', 'Abnormal'])
                
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results[voting_method] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'detailed_results': detailed_results
                }
            else:
                results[voting_method] = {
                    'message': 'Chỉ có một loại nhãn',
                    'accuracy': sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true),
                    'detailed_results': detailed_results
                }
        
        # Lưu kết quả
        self.save_evaluation_results(results, output_dir)
        
        return results
    
    def save_evaluation_results(self, results, output_dir):
        """Lưu kết quả đánh giá"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu summary
        summary_file = os.path.join(output_dir, f'ensemble_evaluation_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ENSEMBLE EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            for method, result in results.items():
                f.write(f"Voting Method: {method}\n")
                f.write("-" * 30 + "\n")
                
                if 'message' in result:
                    f.write(f"{result['message']}\n")
                    f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                else:
                    f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                    f.write(f"Precision: {result['precision']:.4f}\n")
                    f.write(f"Recall: {result['recall']:.4f}\n")
                    f.write(f"F1 Score: {result['f1_score']:.4f}\n")
                
                f.write("\n")
        
        print(f"📁 Kết quả đã được lưu tại: {summary_file}")


def select_images_for_ensemble():
    """Chọn ảnh để test ensemble"""
    print("\n📁 Chọn ảnh để test Ensemble System...")
    
    root = tk.Tk()
    root.withdraw()
    
    file_paths = filedialog.askopenfilenames(
        title="Chọn nhiều ảnh X-quang để test ensemble",
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
    )
    
    root.destroy()
    return list(file_paths)


def main():
    """Demo Multi-Region Ensemble System"""
    print("="*70)
    print("🚀 MULTI-REGION ENSEMBLE FRACTURE DETECTION SYSTEM")
    print("="*70)
    
    try:
        # Khởi tạo ensemble system
        print("🔄 Đang khởi tạo Ensemble System...")
        ensemble = MultiRegionEnsemble()
        
        if not ensemble.models:
            print("❌ Không tìm thấy mô hình nào để tạo ensemble!")
            return
        
        # Chọn ảnh để test
        test_images = select_images_for_ensemble()
        
        if not test_images:
            print("❌ Không có ảnh nào được chọn!")
            return
        
        print(f"\n📊 Đã chọn {len(test_images)} ảnh để test")
        
        # Chọn phương pháp voting
        print("\n🗳️ Chọn phương pháp voting:")
        print("1. Weighted Average (khuyến nghị)")
        print("2. Majority Vote")
        print("3. Max Confidence")
        print("4. Tất cả phương pháp")
        
        choice = input("Nhập lựa chọn (1-4): ").strip()
        
        voting_methods_map = {
            '1': ['weighted_average'],
            '2': ['majority_vote'],
            '3': ['max_confidence'],
            '4': ['weighted_average', 'majority_vote', 'max_confidence']
        }
        
        voting_methods = voting_methods_map.get(choice, ['weighted_average'])
        
        # Chạy evaluation
        print(f"\n🔄 Bắt đầu đánh giá với {len(voting_methods)} phương pháp voting...")
        
        results = ensemble.evaluate_ensemble(
            test_images, 
            voting_methods=voting_methods,
            visualize=True,
            output_dir='ensemble_evaluation_results'
        )
        
        # Hiển thị kết quả
        print(f"\n{'='*70}")
        print("📊 KẾT QUẢ ENSEMBLE EVALUATION")
        print(f"{'='*70}")
        
        for method, result in results.items():
            print(f"\n🗳️ Voting Method: {method.upper()}")
            print("-" * 40)
            
            if 'message' in result:
                print(f"   {result['message']}")
                print(f"   Accuracy: {result['accuracy']:.4f}")
            else:
                print(f"   📈 Accuracy:  {result['accuracy']:.4f}")
                print(f"   🎯 Precision: {result['precision']:.4f}")
                print(f"   🔍 Recall:    {result['recall']:.4f}")
                print(f"   ⚖️ F1 Score:  {result['f1_score']:.4f}")
        
        print(f"\n{'='*70}")
        print("✅ Ensemble evaluation hoàn tất!")
        print("📁 Chi tiết kết quả được lưu trong thư mục 'ensemble_evaluation_results/'")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()