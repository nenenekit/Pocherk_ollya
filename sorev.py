from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import pandas as pd

# Конфигурация
TEST_IMAGES_DIR = "sorev"
TEST_LABELS_FILE = "sorev.tsv"
MODEL_DIR = "trocr-finetuned"  # Папка с сохраненной моделью
NUM_SAMPLES = 24 # Количество примеров для теста


def calculate_score(true_text, pred_text):
    """Вычисляет очки за качество распознавания текста."""
    # Обработка пустых текстов
    if true_text == "":
        if pred_text == "":
            return 20  # Оба текста пустые: 10 за >50% + 10 за полное
        return 0

    # Подсчет очков за слова
    true_words = true_text.split()
    pred_words = pred_text.split()
    word_score = 0

    for i, true_word in enumerate(true_words):
        # Берем соответствующее слово из предсказания (или пустую строку)
        pred_word = pred_words[i] if i < len(pred_words) else ""

        # Полное совпадение слова
        if true_word == pred_word:
            word_score += 10
            continue

        # Проверка частичного совпадения
        min_len = min(len(true_word), len(pred_word))
        matches = 0

        # Считаем совпадающие символы
        for j in range(min_len):
            if true_word[j] == pred_word[j]:
                matches += 1

        # Проверяем условие >50% совпадений
        half_len = (len(true_word) + 1) // 2
        if matches >= half_len:
            word_score += 5

    # Подсчет очков за текст
    total_true_chars = len(true_text)
    min_len = min(len(true_text), len(pred_text))
    correct_chars = 0

    # Считаем совпадающие символы в тексте
    for i in range(min_len):
        if true_text[i] == pred_text[i]:
            correct_chars += 1

    text_bonus = 0
    # Бонус за >50% верных символов
    if correct_chars > total_true_chars / 2:
        text_bonus += 10

    # Бонус за полное совпадение текста
    if true_text == pred_text:
        text_bonus += 10

    return word_score + text_bonus


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
full_scores = 0

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

    # Расчет очков
    score = calculate_score(true_text, predicted_text)
    full_scores+=score
    # Вывод результатов
    print(f"\nИзображение: {file_name}")
    print(f"Истинный текст: {true_text}")
    print(f"Предсказанный текст: {predicted_text}")
    print(f"Очки: {score}")
    print("-" * 50)

print("\nПроверка завершена!")
print(f"Финальные очки: {full_scores}")