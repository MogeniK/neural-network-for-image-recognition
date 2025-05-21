import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2

# Завантаження моделі з попередньо натренованими вагами
model = MobileNetV2(weights='imagenet')

# Функція для завантаження та обробки зображення
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Нормалізація
    img_array = np.expand_dims(img_array, axis=0)  # Додавання batch dimension
    return img_array

# Завантаження зображення
image_path = 'your_image.jpg'  # Замініть на шлях до вашого зображення
image = load_and_preprocess_image(image_path)

# Класифікація зображення
predictions = model.predict(image)
predicted_class = np.argmax(predictions[0])

# Завантаження міток класів ImageNet
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
)
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

# Виведення результату
print(f"Передбачений клас: {labels[predicted_class]}")
