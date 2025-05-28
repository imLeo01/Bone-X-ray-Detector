# tensorflow_compatibility_fix.py
# Fix TensorFlow 2.x compatibility issues

import tensorflow as tf
import numpy as np
import warnings

def fix_tensorflow_compatibility():
    """
    Fix common TensorFlow 2.x compatibility issues
    """
    print("🔧 Applying TensorFlow 2.x compatibility fixes...")
    
    # Check TensorFlow version
    tf_version = tf.__version__
    print(f"📦 TensorFlow version: {tf_version}")
    
    # Disable warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    tf.get_logger().setLevel('ERROR')
    
    # Enable mixed precision if available
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled")
    except:
        print("⚠️ Mixed precision not available")
    
    # Configure GPU memory growth
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth configured for {len(gpus)} GPU(s)")
        else:
            print("💻 Running on CPU")
    except:
        print("⚠️ GPU configuration failed")
    
    print("✅ TensorFlow compatibility fixes applied")

# Monkey patch for deprecated functions
def safe_reduce_std(input_tensor, axis=None, keepdims=False):
    """Safe reduce_std that works with TensorFlow 2.x"""
    try:
        return tf.math.reduce_std(input_tensor, axis=axis, keepdims=keepdims)
    except AttributeError:
        # Fallback calculation
        mean = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
        squared_diff = tf.square(input_tensor - mean)
        variance = tf.reduce_mean(squared_diff, axis=axis, keepdims=keepdims)
        return tf.sqrt(variance)

def safe_reduce_var(input_tensor, axis=None, keepdims=False):
    """Safe reduce_var that works with TensorFlow 2.x"""
    try:
        return tf.math.reduce_variance(input_tensor, axis=axis, keepdims=keepdims)
    except AttributeError:
        # Fallback calculation
        mean = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
        squared_diff = tf.square(input_tensor - mean)
        return tf.reduce_mean(squared_diff, axis=axis, keepdims=keepdims)

# Apply monkey patches
if not hasattr(tf, 'reduce_std'):
    tf.reduce_std = safe_reduce_std

if not hasattr(tf, 'reduce_var'):
    tf.reduce_var = safe_reduce_var

# Initialize compatibility fixes
fix_tensorflow_compatibility()