import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# create an image mimicking a car or use random noise that will be classified as something
# actually just load MobileNetV2 and test a specific label
# It's better to understand if Anomaly Detection in `routes.py` applies to the final predictions
category_mapping = {
    'dog': ['dog', 'hound', 'terrier', 'spaniel', 'retriever', 'pug', 'corgi', 'husky', 'poodle', 'chihuahua', 'collie', 'mastiff', 'malamute', 'shepherd', 'shephard', 'dachshund', 'beagle', 'bulldog', 'dalmatian', 'shiba', 'akita', 'mutt', 'puppy'],
    'cat': ['cat', 'feline', 'tabby', 'tiger', 'lion', 'leopard', 'cheetah', 'panther', 'puma', 'jaguar', 'lynx', 'kitten', 'siamese', 'persian', 'maine coon'],
    'car': ['car', 'vehicle', 'truck', 'cab', 'wagon', 'sedan', 'suv', 'jeep', 'bus', 'van', 'automobile', 'sports car', 'convertible', 'motor', 'scooter', 'motorcycle', 'moped'],
    'person': ['suit', 'groom', 'abaya', 'bikini', 'sunglasses', 'man', 'woman', 'face', 'hair', 'mask', 'jersey', 'tie', 'wig', 'boy', 'girl', 'human', 'person', 'people'],
    'bird': ['bird', 'kite', 'eagle', 'hawk', 'owl', 'parrot', 'sparrow', 'pigeon', 'dove', 'crow', 'raven', 'swan', 'duck', 'goose', 'chicken', 'rooster', 'hen', 'turkey', 'ostrich', 'penguin'],
}

def get_broad_categories(word_list):
    concepts = set(word_list)
    for word in word_list:
        for category, keywords in category_mapping.items():
            if word in keywords or any(word in kw for kw in keywords) or any(kw in word for kw in keywords):
                concepts.add(category)
                concepts.update(keywords)
    return concepts

user_words = set("german shephard".split() + "doberman".split())
user_concepts = get_broad_categories(user_words)
print("User concepts:", user_concepts)

label = "motor scooter"
prob = 0.99
label_words = set(label.lower().replace('_', ' ').split())
imagenet_concepts = get_broad_categories(label_words)
print("Label:", label, "Concepts:", imagenet_concepts)
if not bool(imagenet_concepts & user_concepts):
    print("Mismatch!")
else:
    print("Match!")
