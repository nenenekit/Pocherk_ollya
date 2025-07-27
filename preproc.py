from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import numpy as np


def preprocess_images(
        input_folder: str,
        output_folder: str,
        resize: tuple = None,
        grayscale: bool = False,
        contrast: float = 1.0,
        brightness: float = 1.0,
        sharpness: float = 1.0,
        border: int = 0,
        threshold: int = -1,
        denoise: bool = False,
        filename_prefix: str = 'prep_'
):
    """
    Предобрабатывает изображения в папке с заданными параметрами

    Параметры:
    input_folder - папка с исходными изображениями
    output_folder - папка для сохранения обработанных изображений
    resize - новый размер (ширина, высота) или None
    grayscale - конвертировать в оттенки серого (True/False)
    contrast - коэффициент контрастности (0.0-3.0)
    brightness - коэффициент яркости (0.0-3.0)
    sharpness - коэффициент резкости (0.0-3.0)
    border - размер белой рамки в пикселях
    threshold - порог бинаризации (0-255), -1 = не применять
    denoise - применять удаление шумов (True/False)
    filename_prefix - префикс для имен новых файлов
    """
    # Создаем папку для результатов, если ее нет
    os.makedirs(output_folder, exist_ok=True)

    # Проверка корректности параметров
    if contrast < 0 or contrast > 3:
        raise ValueError("Контрастность должна быть между 0.0 и 3.0")

    if brightness < 0 or brightness > 3:
        raise ValueError("Яркость должна быть между 0.0 и 3.0")

    if sharpness < 0 or sharpness > 3:
        raise ValueError("Резкость должна быть между 0.0 и 3.0")

    if border < 0 or border > 100:
        raise ValueError("Размер рамки должен быть между 0 и 100 пикселями")

    if threshold not in range(-1, 256):
        raise ValueError("Порог бинаризации должен быть между 0 и 255 или -1")

    # Обработка всех изображений в папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Загрузка изображения
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path)

            # Конвертация в RGB для обработки
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Изменение размера
            if resize is not None:
                image = image.resize(resize, resample=Image.LANCZOS)

            # Конвертация в оттенки серого
            if grayscale:
                image = image.convert('L')

            # Управление контрастностью
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)

            # Управление яркостью
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)

            # Управление резкостью
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpness)

            # Добавление рамки
            if border > 0:
                image = ImageOps.expand(image, border=border, fill='white')

            # Бинаризация (черно-белое преобразование)
            if threshold > -1:
                if image.mode != 'L':
                    image = image.convert('L')
                image = image.point(lambda p: 255 if p > threshold else 0)

            # Удаление шумов
            if denoise:
                image = image.filter(ImageFilter.MedianFilter(size=3))

            # Сохранение результата
            new_filename = f"{filename_prefix}{filename}"
            output_path = os.path.join(output_folder, new_filename)

            # Сохраняем в том же формате, что оригинал
            if filename.lower().endswith('.png'):
                image.save(output_path, format='PNG')
            else:
                image.save(output_path, format='JPEG', quality=95)

            print(f"Обработано: {filename} -> {new_filename}")


# Пример использования
if __name__ == "__main__":
    # Настройки препроцессинга (можно менять)
    INPUT_FOLDER = "train"  # Папка с исходными изображениями
    OUTPUT_FOLDER = "train_prep"  # Папка для обработанных изображений
    RESIZE = None  # Новый размер или None
    GRAYSCALE = True  # Конвертировать в оттенки серого
    CONTRAST = 1.5  # Усиление контрастности
    BRIGHTNESS = 1.4  # Усиление яркости
    SHARPNESS = 2.0  # Усиление резкости
    BORDER = 16  # Размер белой рамки
    THRESHOLD = 180  # Порог бинаризации (-1 = не применять)
    DENOISE = False  # Удалить шумы
    FILENAME_PREFIX = ""  # Префикс для новых файлов

    preprocess_images(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        resize=RESIZE,
        grayscale=GRAYSCALE,
        contrast=CONTRAST,
        brightness=BRIGHTNESS,
        sharpness=SHARPNESS,
        border=BORDER,
        threshold=THRESHOLD,
        denoise=DENOISE,
        filename_prefix=FILENAME_PREFIX
    )