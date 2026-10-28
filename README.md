# credit-scoring-model

Автоматизированный пайплайн для PD-модели: вероятность дефолта клиента по кредитной карте в следующем месяце. Данные: Default of Credit Card Clients Dataset (UCI), 30000 клиентов, 22.1% дефолтов, целевая переменная `default.payment.next.month`.

Пайплайн: валидация данных (Great Expectations), очистка и сплит, признаки внутри sklearn Pipeline, подбор гиперпараметров (RandomizedSearchCV), трекинг в MLflow, версии данных и модели в DVC, FastAPI в Docker, проверка дрифта по PSI, CI в GitHub Actions.

## Структура

```
data/raw/            исходный CSV (под DVC)
data/processed/      train.csv и test.csv после стадии prepare (под DVC)
data/expectations/   выгрузка набора правил Great Expectations
models/              обученная модель (под DVC)
reports/             ROC-кривая, отчёт о дрифте
notebooks/           EDA
src/data/            make_dataset.py (очистка, сплит), validation.py (GE)
src/features/        build_features.py (признаки)
src/models/          pipeline.py, train.py, evaluate.py, predict.py
src/api/             app.py (FastAPI)
src/monitoring/      drift.py (PSI), simulate_drift.py
tests/               pytest, tests/data/sample.csv (600 строк для тестов и CI)
scripts/             run_docker.sh
```

## Установка

Python 3.11.

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Модель сохраняется pickle-ом, поэтому версия scikit-learn при обучении и в Docker-образе должна совпадать. После первого успешного запуска зафиксируй версии: `pip freeze > requirements.lock` и сверь с `requirements.txt`.

## DVC

```
git init
dvc init
dvc add data/raw/UCI_Credit_Card.csv
dvc remote add -d local ../dvc-storage
git add .dvc data/raw/UCI_Credit_Card.csv.dvc data/raw/.gitignore
dvc repro
dvc push
```

`dvc repro` выполняет две стадии из `dvc.yaml`: `prepare` (валидация, очистка, сплит) и `train` (6 экспериментов, лучшая модель в `models/credit_default_model.pkl`, метрики в `metrics.json`, ROC в `reports/roc_curve.png`). Метрики: `dvc metrics show`.

## Валидация данных

```
python -m src.data.validation data/raw/UCI_Credit_Card.csv --dump-suite data/expectations/credit_data_suite.json
```

Код выхода 1, если нарушено хотя бы одно правило. Тесты на аномалии: `pytest tests/test_validation.py`. В CI валидация идёт на `tests/data/sample.csv`.

## MLflow

```
mlflow ui
```

Откройте http://localhost:5000, эксперимент `Credit_Default_Prediction`. Каждый запуск логирует параметры, `cv_roc_auc`, `threshold`, `test_roc_auc`, `test_precision`, `test_recall`, `test_f1`, ROC-кривую и модель. Лучшая модель выбирается по `cv_roc_auc` на train, порог классификации по out-of-fold предсказаниям train, test не участвует в выборе.

## Тесты и стиль

```
pytest
black --check .
flake8
```

## API и Docker

После `dvc repro` или `dvc pull` в `models/` лежит модель.

```
bash scripts/run_docker.sh
```

Скрипт собирает образ, запускает контейнер на порту 8000 и отправляет пример из `tests/data/example_request.json` на `POST /predict`. Ответ: `default_prediction` (0 или 1) и `default_probability`. Документация: http://localhost:8000/docs.

Без Docker: `uvicorn src.api.app:app --port 8000`.

## Дрифт

```
python -m src.monitoring.simulate_drift --n 500
python -m src.monitoring.simulate_drift --n 500 --shift
```

Скрипт берёт 500 строк test, отправляет их в API и считает PSI по `PAY_0`, `LIMIT_BAL`, `AGE`, `BILL_AMT1`, `PAY_AMT1` и по предсказанной вероятности относительно train. Пороги: PSI < 0.1 стабильно, 0.1-0.25 умеренный сдвиг, от 0.25 значимый. Флаг `--shift` сдвигает `PAY_0` на единицу и режет `LIMIT_BAL` на 30%. Отчёт: `reports/drift_report.json`.
