# MLOps Lab 4 — CI/CD для ML-проєкту (GitHub Actions + CML)

Ця лабораторна є кумулятивним продовженням [mlops_lab_3](../mlops_lab_3):

- збережено Hydra + Optuna + DVC + MLflow,
- додано автоматичні тести (`pytest`) для pre-train/post-train,
- додано Quality Gate (`f1_test >= 0.70`),
- додано GitHub Actions workflow і CML-звіт у Pull Request.

## Структура

- `src/prepare.py` — підготовка даних.
- `src/train.py` — тренування і створення артефактів ЛР4.
- `src/optimize.py` — HPO (Optuna).
- `src/compare_samplers.py` — порівняння sampler'ів.
- `src/compare_baseline.py` — порівняння поточних метрик з baseline.
- `tests/test_pretrain.py` — швидкі перевірки даних.
- `tests/test_posttrain.py` — перевірка артефактів і Quality Gate.
- `scripts/run_hpo_samplers.sh` — запуск HPO для TPE і Random (`n_trials >= 20`).
- `baseline/metrics.json` — еталонні метрики для порівняння.
- `.github/workflows/cml.yaml` (у корені репозиторію) — CI workflow.

## Встановлення

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Обов'язкові артефакти після тренування

Скрипт `src/train.py` створює:

- `model.pkl`
- `metrics.json`
- `confusion_matrix.png`

## Локальний запуск

```bash
python src/prepare.py
python src/train.py
pytest -q tests/test_pretrain.py
F1_THRESHOLD=0.70 pytest -q tests/test_posttrain.py
```

## Quality Gate

У `tests/test_posttrain.py` реалізовано перевірку:

- `f1_test >= F1_THRESHOLD`
- поріг за замовчуванням: `0.70`.

У `metrics.json` поле `f1_test` рахується як **weighted F1** (підходить для незбалансованих класів цього датасету).

Якщо умова не виконується, `pytest` повертає помилку і CI завершується `failed`.

## HPO (не менше 20 trials)

```bash
bash scripts/run_hpo_samplers.sh
```

За замовчуванням використовується `N_TRIALS=20`. Можна змінити:

```bash
N_TRIALS=30 bash scripts/run_hpo_samplers.sh
```

## DVC pipeline

```bash
dvc repro prepare train
```

У `dvc.yaml` додано stage `train`, що генерує `model.pkl`, `metrics.json`, `confusion_matrix.png`.

## CI workflow (GitHub Actions + CML)

Workflow `.github/workflows/cml.yaml` налаштований на:

- `pull_request` — повний CI + CML коментар,
- `push` у `main/master` — CI + upload `model.pkl` як artifact.

CI кроки:

1. install dependencies,
2. lint (`flake8`) + format check (`black --check`),
3. pre-train tests,
4. training,
5. post-train tests (Quality Gate),
6. baseline comparison,
7. CML report у PR.

## Нотатки про токени

Для CML у PR використовується `REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}`.
Якщо прав недостатньо (наприклад fork PR), використайте PAT в секреті (наприклад `CML_TOKEN`) і підставте його в `REPO_TOKEN`.
