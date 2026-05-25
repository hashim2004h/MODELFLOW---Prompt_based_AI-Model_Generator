import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

img_array = np.random.randint(0, 255, (224, 224, 3)).astype(float)

imagenet_validator = MobileNetV2(weights='imagenet')

val_img = np.expand_dims(img_array, axis=0)
val_img = preprocess_input(val_img)
val_preds = imagenet_validator.predict(val_img, verbose=0)
decoded_top = decode_predictions(val_preds, top=5)[0] 

category_mapping = {
    'dog': ['dog', 'hound', 'terrier', 'spaniel', 'retriever', 'pug', 'corgi', 'husky', 'poodle', 'chihuahua', 'collie', 'mastiff', 'malamute', 'shepherd', 'shephard', 'dachshund', 'beagle', 'bulldog', 'dalmatian', 'shiba', 'akita', 'mutt', 'puppy'],
    'cat': ['cat', 'feline', 'tabby', 'tiger', 'lion', 'leopard', 'cheetah', 'panther', 'puma', 'jaguar', 'lynx', 'kitten', 'siamese', 'persian', 'maine coon'],
    'car': ['car', 'vehicle', 'truck', 'cab', 'wagon', 'sedan', 'suv', 'jeep', 'bus', 'van', 'automobile', 'sports car', 'convertible'],
    'person': ['suit', 'groom', 'abaya', 'bikini', 'sunglasses', 'man', 'woman', 'face', 'hair', 'mask', 'jersey', 'tie', 'wig', 'boy', 'girl', 'human', 'person', 'people'],
    'bird': ['bird', 'kite', 'eagle', 'hawk', 'owl', 'parrot', 'sparrow', 'pigeon', 'dove', 'crow', 'raven', 'swan', 'duck', 'goose', 'chicken', 'rooster', 'hen', 'turkey', 'ostrich', 'penguin'],
    'food': ['food', 'fruit', 'vegetable', 'apple', 'orange', 'lemon', 'banana', 'burger', 'pizza', 'cake', 'bread', 'meat', 'dish', 'plate', 'bowl', 'cup', 'sandwich', 'hotdog', 'taco', 'burrito', 'sushi', 'soup', 'salad', 'pasta', 'noodle', 'rice', 'egg', 'cheese', 'milk', 'water', 'juice', 'coffee', 'tea', 'beer', 'wine', 'cookie', 'chocolate', 'candy', 'ice cream', 'pie', 'donut'],
    'clothing': ['shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'sweater', 'hoodie', 'socks', 'shoes', 'boots', 'sneakers', 'hat', 'cap', 'gloves', 'scarf', 'belt', 'watch', 'glasses', 'jewelry', 'bag', 'backpack', 'purse', 'wallet'],
    'furniture': ['chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'cabinet', 'shelf', 'wardrobe', 'dresser', 'rug', 'carpet', 'curtain', 'lamp', 'mirror', 'clock', 'television', 'tv', 'computer', 'laptop', 'phone', 'tablet', 'camera', 'book', 'pen', 'pencil', 'paper', 'notebook', 'magazine', 'newspaper'],
    'building': ['house', 'apartment', 'building', 'skyscraper', 'office', 'school', 'hospital', 'store', 'shop', 'mall', 'restaurant', 'cafe', 'bar', 'hotel', 'motel', 'factory', 'warehouse', 'church', 'mosque', 'temple', 'museum', 'library', 'stadium', 'arena', 'theater', 'cinema', 'park', 'garden', 'zoo', 'farm', 'bridge', 'road', 'street', 'highway', 'tunnel', 'airport', 'station', 'port', 'harbor', 'dock', 'beach', 'mountain', 'hill', 'valley', 'forest', 'wood', 'jungle', 'desert', 'river', 'lake', 'sea', 'ocean', 'island']
}

def get_broad_categories(word_list):
    concepts = set(word_list)
    for word in word_list:
        for category, keywords in category_mapping.items():
            if word in keywords or any(word in kw for kw in keywords) or any(kw in word for kw in keywords):
                concepts.add(category)
                concepts.update(keywords)
    return concepts

class_names = ["german shephard", "doberman"]
user_words = set()
for c in class_names:
    user_words.update(c.lower().replace('_', ' ').split())
    
user_concepts = get_broad_categories(user_words)
print("User Concepts:", user_concepts)

mismatch_mass = 0.0
for _, label, prob in decoded_top:
    label_words = set(label.lower().replace('_', ' ').split())
    imagenet_concepts = get_broad_categories(label_words)
    print(f"Top 5: {label} ({prob:.2f}) -> Concepts: {imagenet_concepts}")
    if not bool(imagenet_concepts & user_concepts):
        mismatch_mass += prob

print(f"Mismatch: {mismatch_mass}")
if mismatch_mass > 0.65:
    print("ANOMALY DETECTED!")
