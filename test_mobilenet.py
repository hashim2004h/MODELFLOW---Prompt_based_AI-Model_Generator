import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

print("Loading MobileNet...")
try:
    imagenet_validator = MobileNetV2(weights='imagenet')
    print("Success!")
except Exception as e:
    print("Failed!")
    print(e)
