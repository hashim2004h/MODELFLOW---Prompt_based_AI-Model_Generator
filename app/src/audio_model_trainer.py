"""
Audio Classification Model Trainer
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
from datetime import datetime

class AudioClassifier:
    def __init__(self, num_classes, input_shape=(128, 128, 3), backbone_name='simple_cnn'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.backbone_name = backbone_name
        self.model = None
        self.class_names = []
        self.history = None
        
    def build_model(self):
        """Build a model for Audio spectrogram classification"""
        
        if self.backbone_name == 'simple_cnn':
            model = keras.Sequential([
                layers.Input(shape=self.input_shape),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            # Transfer learning with MobileNetV2 (spectrogram images as RGB)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            inputs = layers.Input(shape=self.input_shape)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
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
    
    def train(self, X_train, y_train, epochs=20, batch_size=8, validation_split=0.2, callbacks=None):
        if self.model is None:
            self.build_model()
            
        # Optional validation split manually to just pass arrays
        val_samples = max(1, int(len(X_train) * validation_split))
        X_val = X_train[:val_samples]
        y_val = y_train[:val_samples]
        X_train_split = X_train[val_samples:]
        y_train_split = y_train[val_samples:]
        
        if len(X_train_split) == 0:
            X_train_split = X_train
            y_train_split = y_train
            
        active_callbacks = callbacks if callbacks else []
        
        self.history = self.model.fit(
            X_train_split, y_train_split,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=active_callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, feature):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if len(feature.shape) == 3:
            feature = np.expand_dims(feature, axis=0)
            
        predictions = self.model.predict(feature, verbose=0)
        return predictions[0]
    
    def save(self, path):
        if self.model:
            model_dir = os.path.dirname(path)
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(path)
            
            metadata = {
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Model and metadata saved to {path}")
            
    def load(self, path):
        self.model = keras.models.load_model(path)
        metadata_path = path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names', [])
                self.num_classes = metadata.get('num_classes', len(self.class_names))
                self.input_shape = tuple(metadata.get('input_shape', (128, 128, 3)))
        print(f"Model loaded from {path}")
