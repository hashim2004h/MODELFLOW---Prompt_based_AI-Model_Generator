"""
Real-time Image Classification Model Trainer
Uses TensorFlow/Keras to train CNN models
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
from datetime import datetime
from scipy import ndimage

class ImageClassifier:
    def __init__(self, num_classes, image_size=(224, 224), backbone_name='mobilenet_v2'):
        self.num_classes = num_classes
        self.image_size = image_size
        self.backbone_name = backbone_name
        self.model = None
        self.class_names = []
        self.history = None
        
    def build_model(self):
        """Build a powerful Transfer Learning model using the selected backbone"""
        # Map names to Keras applications
        backbones = {
            'mobilenet_v2': tf.keras.applications.MobileNetV2,
            'resnet50': tf.keras.applications.ResNet50,
            'efficientnet_b0': tf.keras.applications.EfficientNetB0,
            'vgg16': tf.keras.applications.VGG16
        }
        
        if self.backbone_name not in backbones:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
            
        BackboneClass = backbones[self.backbone_name]
        
        # Load Backbone (pretrained on ImageNet)
        # Note: EfficientNet expects [0-255], others might expect [-1, 1] or [0, 1]
        # We will add specific preprocessing layers if needed, or rely on the backbone's preprocess_input
        # but for simplicity in this custom trainer, we'll use a generic rescaling if not EfficientNet
        
        base_model = BackboneClass(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        inputs = layers.Input(shape=(*self.image_size, 3))
        
        # Preprocessing per model type
        x = inputs
        if self.backbone_name == 'mobilenet_v2':
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        elif self.backbone_name == 'resnet50':
            x = tf.keras.applications.resnet50.preprocess_input(x)
        elif self.backbone_name == 'vgg16':
            x = tf.keras.applications.vgg16.preprocess_input(x)
        # EfficientNet has internal preprocessing, so we don't need to do anything for it 
        # (it expects 0-255 float inputs)
            
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=4, validation_split=0.2, callbacks=None):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Data augmentation for better generalization
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Split data
        val_samples = int(len(X_train) * validation_split)
        X_val = X_train[:val_samples]
        y_val = y_train[:val_samples]
        X_train = X_train[val_samples:]
        y_train = y_train[val_samples:]
        
        # Train with augmentation
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, image):
        """Make prediction on single image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        return predictions[0]
    
    def save(self, path):
        """Save model and class names"""
        if self.model:
            model_dir = os.path.dirname(path)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            self.model.save(path)
            
            # Save class names
            class_names_path = path.replace('.h5', '_classes.json')
            with open(class_names_path, 'w') as f:
                json.dump(self.class_names, f)
            
            print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model and class names"""
        self.model = keras.models.load_model(path)
        
        # Load class names
        class_names_path = path.replace('.h5', '_classes.json')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        
        print(f"Model loaded from {path}")
