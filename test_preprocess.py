import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

inputs = layers.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# save and load
model.save("test_model.h5")
print("Saved successfully!")

new_model = tf.keras.models.load_model("test_model.h5")
print("Loaded successfully!")
new_model.predict(np.zeros((1, 224, 224, 3)))
print("Predicted successfully!")
