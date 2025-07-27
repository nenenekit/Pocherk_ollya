from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import pandas as pd

# Конфигурация
TEST_IMAGES_DIR = "train"
TEST_LABELS_FILE = "train.tsv"
MODEL_DIR = "trocr-finetuned"  # Папка с сохраненной моделью
NUM_SAMPLES = 12  # Количество примеров для теста


# Загрузка меток тестовых данных
def load_labels(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['file', 'text'])
    return dict(zip(df['file'], df['text']))


# Загрузка модели и процессора
print("Загрузка модели...")
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)

# Загрузка тестовых меток
test_labels = load_labels(TEST_LABELS_FILE)

# Выбираем случайные примеры для проверки
sample_files = list(test_labels.keys())[:NUM_SAMPLES]

print("\nПроверка модели:")
for file_name in sample_files:
    # Загрузка изображения
    image_path = os.path.join(TEST_IMAGES_DIR, file_name)
    image = Image.open(image_path).convert("RGB")

    # Обработка изображения
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Генерация предсказания
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Получение истинного текста
    true_text = test_labels[file_name]

    # Вывод результатов
    print(f"\nИзображение: {file_name}")
    print(f"Истинный текст: {true_text}")
    print(f"Предсказанный текст: {predicted_text}")
    print("-" * 50)

print("\nПроверка завершена!")