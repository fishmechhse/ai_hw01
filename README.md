# Сервис предсказания цены на FastAPI

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Описание проделанной работы по подготовке предиктора:

Пробовал разные модели: `Lasso, LinearRegression, ElasticNet, Ridge`.
Все показывают примерно одинаковое значение по r2 метрике. Честно говоря самое высокое значение на текущих трансформерах
было достигнуто на линейной регрессии.
Пробовал тренировать на разном количестве признаков.
Чем больше новых признаков создавал, тем ближе к единице было значение r2.
Например, при использовании `HasherTransformer(column="name", n_features=1000)` и указании большого количества фичей,
сильно увеличивался r2 (в текущей модели я его убрал, т.к. мне кажется, что модель становится переобученной из-за такого
количества фичей).
Так же использование OneHotEncoder на всех категориальных признаках давало r2=0.93 (Ridge, alpha=1). Конечно количество
фичей при этом получалось очень большим и как мне кажется, модель была переобучена.

Я остановился на варианте использования TargetEncoder для столбца name. OneHotEncoder для остальных категориальных
признаков.
Столбец torque удалил (как было сказано в задании, хотя конечно его можно было бы обработать).
В принципе из name можно было бы вычленить марку автомобиля и для нее применить OneHot кодирование.
Экспериментов было проведено много. Не думаю, что я взял самый оптимальный вариант. Текущая модель имеет `r2=0.66` на
тестовой выборки из задания.
Сервис принимает в запросе и отдает в ответе json (не csv). Вроде как это не принципиально. Обсуждал в учебном чате.


## Структура репозитория:

* `./model/pipeline.pkl` - файл с натренированной моделью Ridge регрессии.
* `./converter` - Конвертирует csv тестовой выборки в json, но при этом не обрабатывает пропуски. Использовал для
  получения json объектов для тестирования API сервиса
* `./predictor` - Класс, который инкапсулирует логику предсказания цены.
* `./train` - Проверка эстиматора на тестовой выборке.
* `./trained_pipeline` - Модуль кастомного трансформера `TrimTransformer`, который выполняет преобразование столбцов. Так же из файла pipeline.py можно запустить обучение модели.
* `mileage, engine, max_power`
* `./main` - код эндпоинтов сервиса.

## Запуск сервера:

Для запуска сервиса нужно выполнить команду

```bash
uvicorn main:app --reload 
```

## Тестирование сервиса:

Запрос на предсказание цены для одного автомобиля:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict_item' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '  {
    "name": "Mahindra Xylo E4 BS IV",
    "year": 2010,
    "km_driven": 168000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "14.0 kmpl",
    "engine": "2498 CC",
    "max_power": "112 bhp",
    "torque": "260 Nm at 1800-2200 rpm",
    "seats": 7.0
  }'
```

Пример ответа:

```json
{
  "name": "Mahindra Xylo E4 BS IV",
  "year": 2010,
  "selling_price": 692842.4939734193,
  "km_driven": 168000,
  "fuel": "Diesel",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner",
  "mileage": "14.0 kmpl",
  "engine": "2498 CC",
  "max_power": "112 bhp",
  "torque": "260 Nm at 1800-2200 rpm",
  "seats": 7
}
```

Запрос на предсказание цены для массива автомобилей:

```bash
curl -X 'POST' \
'http://127.0.0.1:8000/predict_items' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d ' [{
"name": "Honda Civic 1.8 S AT",
"year": 2007,
"km_driven": 218463,
"fuel": "Petrol",
"seller_type": "Individual",
"transmission": "Automatic",
"owner": "First Owner",
"mileage": "12.9 kmpl",
"engine": "1799 CC",
"max_power": "130 bhp",
"torque": "172Nm@ 4300rpm",
"seats": 5.0
},
{
"name": "Honda City i DTEC VX",
"year": 2015,
"km_driven": 173000,
"fuel": "Diesel",
"seller_type": "Individual",
"transmission": "Manual",
"owner": "First Owner",
"mileage": "25.1 kmpl",
"engine": "1498 CC",
"max_power": "98.6 bhp",
"torque": "200Nm@ 1750rpm",
"seats": 5.0
}]'
```

Пример ответа:

```json
[
  {
    "name": "Honda Civic 1.8 S AT",
    "year": 2007,
    "selling_price": 856454.6524704068,
    "km_driven": 218463,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Automatic",
    "owner": "First Owner",
    "mileage": "12.9 kmpl",
    "engine": "1799 CC",
    "max_power": "130 bhp",
    "torque": "172Nm@ 4300rpm",
    "seats": 5
  },
  {
    "name": "Honda City i DTEC VX",
    "year": 2015,
    "selling_price": 738235.6858571514,
    "km_driven": 173000,
    "fuel": "Diesel",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "25.1 kmpl",
    "engine": "1498 CC",
    "max_power": "98.6 bhp",
    "torque": "200Nm@ 1750rpm",
    "seats": 5
  }
]
```

## С какими сложностями столкнулся:
Не удалось нормально импортировать кастомный трансформер из коллаба. Пробовал pickle, cloudpickle, joblib. 
Каждый раз возникала ошибка при загрузке модели при запуске сервиса. При этом загрузка в не модуля сервиса выполнялась нормально. 
Скорее всего что-то не так с названием модуля трансформера, т.к. проблема 
с загрузкой модели возникала только при использовании моего кастомного трансформера.
В итоге скопировал код из ноутбука в этот проект, обучил модель (выполнилось гораздо быстрее), сделал дамп модели и восстановление.