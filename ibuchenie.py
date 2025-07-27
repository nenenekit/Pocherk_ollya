from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch
from transformers.trainer_utils import IntervalStrategy

# Конфигурация
TRAIN_IMAGES_DIR = "train"
TEST_IMAGES_DIR = "train"
TRAIN_LABELS_FILE = "train.tsv"
TEST_LABELS_FILE = "train.tsv"
MODEL_NAME = "trocr-finetuned"
OUTPUT_DIR = "trocr-finetuned"
BATCH_SIZE = 4  # Уменьшаем для CPU
EPOCHS = 6  # Начнем с 1 эпохи для теста
LEARNING_RATE = 5e-5
SAVE_STEPS = 1000
LOGGING_STEPS = 100
MAX_LENGTH = 64  # Максимальная длина текста

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")
print(f"Размер батча: {BATCH_SIZE}, Эпохи: {EPOCHS}, LR: {LEARNING_RATE}")


# Загрузка меток с проверкой существования файлов
def load_labels(file_path):
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл меток {file_path} не найден!")
        return {}

    df = pd.read_csv(file_path, sep='\t', header=None, names=['file', 'text'])
    return dict(zip(df['file'], df['text']))


print("Загрузка меток...")
train_labels = load_labels(TRAIN_LABELS_FILE)
test_labels = load_labels(TEST_LABELS_FILE)

if not train_labels or not test_labels:
    print("Ошибка: Не удалось загрузить метки!")
    exit()

print(f"Загружено тренировочных меток: {len(train_labels)}")
print(f"Загружено тестовых меток: {len(test_labels)}")


# Создание кастомного датасета с проверкой файлов
class HandwritingDataset(Dataset):
    def __init__(self, images_dir, labels_dict, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.labels = {}
        self.file_list = []

        # Проверяем существование файлов
        for file_name, text in labels_dict.items():
            file_path = os.path.join(images_dir, file_name)
            if os.path.exists(file_path):
                self.file_list.append(file_name)
                self.labels[file_name] = text
            else:
                print(f"Предупреждение: Файл {file_path} не найден, пропускаем")

        print(f"Валидных файлов в датасете: {len(self.file_list)}/{len(labels_dict)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        text = self.labels[file_name]
        image_path = os.path.join(self.images_dir, file_name)

        try:
            # Загрузка изображения
            image = Image.open(image_path).convert("RGB")

            # Предобработка
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # Токенизация текста
            labels = self.processor.tokenizer(
                text,
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]

            return {
                "pixel_values": pixel_values.squeeze(),
                "labels": labels
            }

        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")
            # Возвращаем пустые данные для этого элемента
            return {
                "pixel_values": torch.zeros(3, 384, 384),  # Размер как у TrOCR
                "labels": torch.zeros(MAX_LENGTH, dtype=torch.long)
            }


# Проверка существования папок
for folder in [TRAIN_IMAGES_DIR, TEST_IMAGES_DIR]:
    if not os.path.exists(folder):
        print(f"Ошибка: Папка {folder} не существует!")
        exit()

print("\nИнициализация модели...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(device)

print(f"Процессор и модель загружены: {MODEL_NAME}")

print("\nСоздание датасетов...")
train_dataset = HandwritingDataset(TRAIN_IMAGES_DIR, train_labels, processor)
test_dataset = HandwritingDataset(TEST_IMAGES_DIR, test_labels, processor)

if len(train_dataset) == 0 or len(test_dataset) == 0:
    print("Ошибка: Один из датасетов пуст!")
    exit()

print(f"Размер тренировочного датасета: {len(train_dataset)}")
print(f"Размер тестового датасета: {len(test_dataset)}")

# Конфигурация обучения
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    report_to="none",
    save_total_limit=2,
    fp16=False,  # Отключаем для CPU
    logging_dir=f"{OUTPUT_DIR}/logs",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    dataloader_num_workers=0,  # Для CPU ставим 0
)

# Инициализация тренера (используем только токенизатор)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    data_collator=default_data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("\nНачало обучения...")
try:
    trainer.train()
    print("Обучение завершено успешно!")
except Exception as e:
    print(f"Ошибка во время обучения: {str(e)}")

# Сохранение результатов
try:
    print("\nСохранение модели...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Модель сохранена в: {OUTPUT_DIR}")
except Exception as e:
    print(f"Ошибка при сохранении модели: {str(e)}")

# Оценка модели
if len(test_dataset) > 0:
    print("\nОценка модели на тестовом наборе...")
    try:
        metrics = trainer.evaluate()
        print("Результаты оценки:")
        print(f"Средняя ошибка: {metrics.get('eval_loss', 'N/A')}")
        print(f"Метрики: {metrics}")
    except Exception as e:
        print(f"Ошибка при оценке: {str(e)}")
else:
    print("Пропуск оценки: тестовый датасет пуст")

print("\nПрограмма завершена!")