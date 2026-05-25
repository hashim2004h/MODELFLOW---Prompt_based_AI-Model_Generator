"""
Improved Real-time Image Classification Model Trainer
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
from datetime import datetime

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
        
        if self.backbone_name == 'custom_cnn':
            # Custom 3-layer CNN (Original implementation)
            model = keras.Sequential([
                layers.Input(shape=(*self.image_size, 3)),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            self.model = model
            return model

        if self.backbone_name not in backbones:
            # Fallback to MobileNetV2 if unknown
            print(f"Unknown backbone {self.backbone_name}, defaulting to MobileNetV2")
            BackboneClass = tf.keras.applications.MobileNetV2
            self.backbone_name = 'mobilenet_v2'
        else:
            BackboneClass = backbones[self.backbone_name]
        
        # Load Backbone (pretrained on ImageNet)
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
        # EfficientNet has internal preprocessing built-in (expects 0-255)
            
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Use a reasonable learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=20, batch_size=8, validation_split=0.2, callbacks=None):
        """Train with transfer learning settings"""
        if self.model is None:
            self.build_model()
        
        # Data augmentation
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Split data locally to ensure consistent validation set if needed, 
        # but datagen.flow handles it if we pass subset in flow.
        # Here we manually split as before for clarity/control
        val_samples = max(1, int(len(X_train) * validation_split))
        X_val = X_train[:val_samples]
        y_val = y_train[:val_samples]
        X_train_split = X_train[val_samples:]
        y_train_split = y_train[val_samples:]
        
        # Combine passed callbacks with internal ones if needed
        active_callbacks = callbacks if callbacks else []
        
        # Add EarlyStopping to speed up training when model converges quickly
        from tensorflow.keras.callbacks import EarlyStopping
        active_callbacks.append(EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))
        
        self.history = self.model.fit(
            datagen.flow(X_train_split, y_train_split, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=active_callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, image):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Note: Preprocessing is now handled INSIDE the model via layers or preprocess_input calls in build_model
        # We assume input image is [0, 255] RGB for consistency with training data ingestion
        
        predictions = self.model.predict(image, verbose=0)
        return predictions[0]
    
    def save(self, path):
        """Save model and metadata"""
        if self.model:
            model_dir = os.path.dirname(path)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            self.model.save(path)
            
            # Save metadata
            metadata = {
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'image_size': self.image_size,
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model and metadata saved to {path}")
    
    def load(self, path):
        """Load model and metadata"""
        self.model = keras.models.load_model(path)
        
        # Load metadata
        metadata_path = path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.class_names = metadata.get('class_names', [])
                self.num_classes = metadata.get('num_classes', len(self.class_names))
                self.image_size = tuple(metadata.get('image_size', (224, 224)))
        
        print(f"Model loaded from {path}")
