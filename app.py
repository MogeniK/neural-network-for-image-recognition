from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

app = Flask(__name__)

# Завантаження моделі
model = MobileNetV2(weights='imagenet')

# Функція для обробки зображення
def load_and_preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Нормалізація
    img_array = np.expand_dims(img_array, axis=0)  # Додавання batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Отримання зображення від користувача
        file = request.files['image']
        if file:
            # Обробка зображення
            image = load_and_preprocess_image(file)
            predictions = model.predict(image)
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            
            # Форматування результатів
            results = [(label, f"{prob*100:.2f}%") for (_, label, prob) in decoded_predictions]
            return render_template('index.html', results=results)
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)