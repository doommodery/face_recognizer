# Проект по распознаванию лиц (OpenCV + Python)

Данный проект реализует систему распознавания лиц в режиме реального времени. Для этого используются алгоритмы LBPH (Local Binary Patterns Histograms) и Fisherfaces. Проект включает в себя скрипты для:
1. Сбора датасета (dataset_creator.py).
2. Автоматической* или ручной тренировки модели (trainer.py).
3. Распознавания лиц с веб-камеры, удалённого видеопотока или на изображении (face_recognition.py).
4. Управления процессом через консольное меню (main.py).*Автоматическая тренировака происходит при выборе пункта 1


## Возможности
1. Добавление нового пользователя и автоматическая тренировка модели.
2. Ручная тренировка модели с поддержкой алгоритмов LBPH и Fisherfaces.
3. Распознавание лиц:
   - С веб-камеры в реальном времени.
   - На удалённом видеопотоке (URL или локальный файл).
   - На загруженном изображении.
4. Ведение лога распознаваний (face_log.txt), где фиксируются дата, время и процент уверенности.

## Установка и запуск
1. Склонируйте репозиторий или скачайте файлы проекта.
2. Установите все зависимости из requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
3. Убедитесь, что в папке с проектом присутствует файл [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) (его можно взять из репозитория OpenCV).
4. Запустите основной скрипт:
   ```bash
   python main.py
   ```
5. Выберите нужное действие в меню (1–6).

## Структура проекта
- **main.py** – Точка входа. Предоставляет консольное меню.
- **dataset_creator.py** – Сбор датасета: создание папки для нового пользователя и захват его лиц с веб-камеры.
- **trainer.py** – Тренировка или пересборка обученной модели (trained_faces_lbph.yml и trained_faces_fisher.yml).
- **face_recognition.py** – Распознавание лиц (с веб-камеры, видеопотока или на изображении) с логированием результатов.
- **trained_faces_lbph.yml** и **trained_faces_fisher.yml** – Хранят коэффициенты обученных моделей (LBPH и FisherFaces соответсвенно).
- **faces/** – Каталог, где в подпапках хранятся изображения лиц каждого пользователя.
- **users.json** – Хранит соответствие ID пользователя и имени.
- **requirements.txt** – Перечень библиотек, необходимых для работы проекта.
- **face_log.txt** – Файл лога распознаваний (создаётся автоматически).

## Быстрый старт
После установки зависимостей (см. выше) и запуска main.py:
1. Выберите «1» (Добавить нового человека) – будет создан пользователь, соберутся его изображения и обучится модель.
2. Выберите «3» (Определение лиц c вебкамеры) – протестируйте систему распознавания. Для выхода нажмите «q».

Если нужно распознавать лица на видео:
- Выберите «4» (Определение лиц на потоковом видео) – введите адрес потокового видео или путь к локальному файлу и (опционально) задержку между кадрами.

Если нужно распознавать лица на изображении:
- Выберите «5» (Определение лиц на изображении) – введите путь к изображению (локальному).

## Логи и результаты
- Все распознавания пишутся в face_log.txt (дата, время, имя пользователя или «Неопознанный», а также процент уверенности).
- Для пересборки модели в любое время можно использовать пункт «2» (trainer.py).

## Изменения

1.Изменения в фаиле main.py:
Добавление нового пункта меню для распознавания лиц на изображении:
В меню добавлен новый пункт "5. Определение лиц на изображении".
Добавлена функция recognize_image(), которая запрашивает у пользователя путь к изображению и вызывает скрипт face_recognition.py с параметром --image для распознавания лиц на изображении.
Выбор алгоритма распознавания:
Добавлена функция select_algorithm(), которая позволяет пользователю выбрать алгоритм распознавания лиц (LBPH или Fisherfaces).
В функциях start_recognition(), start_stream_recognition() и recognize_image() добавлен вызов функции select_algorithm() для выбора алгоритма и передачи его в качестве параметра скрипту face_recognition.py.

2.Изменения в фаиле trainer.py:
Поддержка нескольких алгоритмов распознавания лиц:
Добавлены две модели распознавания лиц: LBPH и Fisherfaces.
Определены файлы для сохранения обученных моделей: TRAINED_FILES = {"LBPH": "trained_faces_lbph.yml", "Fisherfaces": "trained_faces_fisher.yml"}.
Изменение функции train_model():
Функция train_model() теперь обучает обе модели (LBPH и Fisherfaces) и сохраняет их в соответствующие файлы.
Добавлены комментарии о том, что при изменении параметров алгоритма LBPH файл с результатами обучения может стать слишком большим.
Изменение функции main():
Функция main() теперь проверяет наличие всех файлов моделей (TRAINED_FILES.values()) и предлагает пользователю пересоздать модели, если они уже существуют.
Изменение функции collect_faces_and_ids():
Добавлено изменение размера изображений лиц до 200x200 пикселей перед добавлением их в список face_samples.

3.Изменения в фаиле trainer.py:
Поддержка нескольких алгоритмов распознавания лиц:
Добавлена поддержка двух алгоритмов распознавания лиц: LBPH и Fisherfaces.
Определены файлы для сохранения обученных моделей: TRAINED_FILES = {"LBPH": "trained_faces_lbph.yml", "Fisherfaces": "trained_faces_fisher.yml"}.
Добавление функции load_recognizer:
Функция load_recognizer загружает выбранный алгоритм распознавания (LBPH или Fisherfaces) и подгружает обученную модель из соответствующего файла.
Добавление функции recognize_from_image:
Функция recognize_from_image позволяет распознавать лица на изображении. Она загружает изображение, выполняет распознавание лиц и отображает результаты.
Добавлено масштабирование изображения для отображения в окне с максимальной шириной 800 пикселей и высотой 600 пикселей.
Изменение функции process_video_stream:
Функция process_video_stream теперь принимает параметр algorithm для выбора алгоритма распознавания.
Добавлено изменение размера изображений лиц до 200x200 пикселей перед распознаванием.
Изменение функции main:
Добавлен аргумент --algorithm для выбора алгоритма распознавания (LBPH или Fisherfaces).
Добавлен аргумент --image для указания пути к изображению для распознавания.
Добавлена проверка наличия файла модели для выбранного алгоритма и загрузка соответствующего распознавателя.
Изменение обработки уверенности (confidence):
Для алгоритма Fisherfaces изменено преобразование значения confidence в процент уверенности.