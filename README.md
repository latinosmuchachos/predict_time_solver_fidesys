Данные модели являются частью вузовской работы
Результаты этих моделей вы можете использовать в своих работах на свое усмотрение без различных притязаний

# Настройка окружения
1. Создайте виртуальное окружение, например, с помощью команды `python -m venv venv`
2. Активируйте виртуальное окружение, например, с помощью команды `source venv/bin/activate`
3. Запустите команду pip install -r requirements.txt для установки зависимостей

# Настройка модели
В src/train_models вы можете найти две модели, которые обучались на данных из src/learn_data. Эта папка сейчас пуста, но она должна содержать данные для обучения в следующем формате:
src 
|__ /learn_data
    |__ /id_solver
        |__ calc.json

calc.json должен содержать параметры для обучения и значение пресказываемого параметра в формате JSON

Далее, вам необходимо указать какие параметры будут использоваться для обучения и предсказания. Указать их надо в файле с моделью (train_params, predict_parameter)

Далее запускаете файл с моделью и она обучается - Вуаля!

