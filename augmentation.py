from PIL import Image, ImageEnhance, ImageOps
import os
import numpy as np


def augment_images(
        input_folder: str,
        output_folder: str,
        rotation_angle: float = 0,
        color_percent: float = 0,
        add_color: str = 'False',
        width_percent: float = 100,
        filename_prefix: str = 'aug_'
):
    """
    Аугментирует изображения в папке с заданными параметрами

    Параметры:
    input_folder - папка с исходными изображениями
    output_folder - папка для сохранения аугментированных изображений
    rotation_angle - угол поворота (-360 до 360 градусов)
    color_percent - интенсивность цвета (0-100%)
    add_color - цвет для добавления ('синий','зеленый','красный','фиолетовый','желтый','False')
    width_percent - изменение ширины (1-200%)
    filename_prefix - префикс для имен новых файлов
    """
    # Создаем папку для результатов, если ее нет
    os.makedirs(output_folder, exist_ok=True)

    # Соответствие цветов их RGB-значениям
    colors = {
        'синий': (0, 0, 255),
        'зеленый': (0, 255, 0),
        'красный': (255, 0, 0),
        'фиолетовый': (128, 0, 128),
        'желтый': (255, 255, 0),
        'False': None
    }

    # Проверка корректности параметров
    if rotation_angle < -360 or rotation_angle > 360:
        raise ValueError("Угол поворота должен быть между -360 и 360 градусами")

    if color_percent < 0 or color_percent > 100:
        raise ValueError("Процент цвета должен быть между 0 и 100")

    if add_color not in colors:
        raise ValueError("Недопустимый цвет. Используйте: 'синий','зеленый','красный','фиолетовый','желтый','False'")

    if width_percent < 1 or width_percent > 200:
        raise ValueError("Процент ширины должен быть между 1 и 200")

    # Обработка всех изображений в папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Загрузка изображения
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert('RGB')

            # Изменение ширины
            if width_percent != 100:
                original_width, original_height = image.size
                new_width = int(original_width * width_percent / 100)
                image = image.resize((new_width, original_height))

            # Поворот изображения
            if rotation_angle != 0:
                image = image.rotate(rotation_angle, expand=True)

            # Добавление цветового оттенка
            if add_color != 'False' and color_percent > 0:
                # Создаем цветной слой
                color_layer = Image.new('RGB', image.size, colors[add_color])

                # Рассчитываем прозрачность
                alpha = color_percent / 100.0

                # Смешиваем изображения
                image = Image.blend(image, color_layer, alpha)

            # Сохранение результата
            new_filename = f"{filename_prefix}{filename}"
            output_path = os.path.join(output_folder, new_filename)
            image.save(output_path)
            print(f"Создано: {new_filename}")


# Пример использования
if __name__ == "__main__":
    # Настройки аугментации (можно менять)
    INPUT_FOLDER = "train"  # Папка с исходными изображениями
    OUTPUT_FOLDER = "train_aug"  # Папка для аугментированных изображений
    ROTATION = 2  # Угол поворота в градусах
    COLOR_PERCENT = 14  # Интенсивность цвета (0-100%)
    ADD_COLOR = "желтый"  # Цвет для добавления или 'False'
    WIDTH_PERCENT = 150  # Процент изменения ширины
    FILENAME_PREFIX = ""  # Префикс для новых файлов

    augment_images(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        rotation_angle=ROTATION,
        color_percent=COLOR_PERCENT,
        add_color=ADD_COLOR,
        width_percent=WIDTH_PERCENT,
        filename_prefix=FILENAME_PREFIX
    )