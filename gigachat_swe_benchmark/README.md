# GigaChat3-10B-A1.8B SWE-bench Benchmark

Полный пайплайн для оценки модели [GigaChat3-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B) на бенчмарке SWE-bench с использованием mini-swe-agent.

## Требования

- **GPU**: NVIDIA A100 (40GB+) или аналогичная
- **Docker**: Для запуска SWE-bench контейнеров
- **Python**: 3.10+
- **CUDA**: 12.0+

## Быстрый старт

```bash
# Сделать скрипт исполняемым
chmod +x run_gigachat_swebench.sh

# Запуск полного бенчмарка (скачивание модели + оценка + аналитика)
./run_gigachat_swebench.sh

# Запуск на первых 10 инстансах SWE-bench Lite
./run_gigachat_swebench.sh --subset lite --slice 0:10

# Запуск с параллельными воркерами
./run_gigachat_swebench.sh --subset lite --workers 4 --output ./my_results
```

## Параметры запуска

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--subset` | Датасет SWE-bench (lite, verified, full) | lite |
| `--split` | Сплит датасета (dev, test) | dev |
| `--slice` | Срез инстансов (например, 0:10) | все |
| `--workers` | Количество параллельных воркеров | 1 |
| `--output` | Директория результатов | ./gigachat_results |
| `--skip-model` | Пропустить скачивание модели | false |
| `--skip-deps` | Пропустить установку зависимостей | false |
| `--gpu-util` | Утилизация памяти GPU (0.0-1.0) | 0.90 |
| `--tp` | Tensor parallel size | 1 |

## Структура файлов

```
gigachat_swe_benchmark/
├── run_gigachat_swebench.sh   # Главный скрипт запуска
├── gigachat_swebench.yaml     # Конфигурация для mini-swe-agent
├── model_registry.json        # Регистрация модели в LiteLLM
├── start_vllm_server.py       # Скрипт запуска vLLM сервера
├── analyze_results.py         # Анализатор результатов
└── README.md                  # Документация
```

## Результаты

После выполнения в директории результатов будут:

```
gigachat_results/
├── preds.json              # Предсказания в формате SWE-bench
├── analysis_report.json    # JSON отчет аналитики
├── results_detailed.csv    # Детальные результаты по инстансам
├── results_summary.csv     # Сводка по репозиториям
├── vllm_server.log         # Логи vLLM сервера
├── exit_statuses_*.yaml    # Статусы завершения
├── plots/                  # Визуализации
│   ├── exit_status_distribution.png
│   ├── step_distribution.png
│   ├── repo_performance.png
│   └── error_categories.png
└── {instance_id}/          # Траектории по инстансам
    └── {instance_id}.traj.json
```

## Запуск отдельных компонентов

### Только vLLM сервер

```bash
python start_vllm_server.py --model-id ai-sage/GigaChat3-10B-A1.8B --port 8000
```

### Только анализ результатов

```bash
python analyze_results.py ./gigachat_results --all
```

### Только SWE-bench (при запущенном vLLM)

```bash
python -m minisweagent.run.extra.swebench \
    --subset lite \
    --split dev \
    --config gigachat_swebench.yaml \
    --output ./results
```

## Валидация результатов с SWE-bench

После получения предсказаний можно запустить официальную оценку SWE-bench:

```bash
# Установка SWE-bench
pip install swebench

# Запуск оценки
python -m swebench.harness.run_evaluation \
    --predictions_path ./gigachat_results/preds.json \
    --swe_bench_tasks lite \
    --run_id gigachat_eval
```

## Особенности модели

GigaChat3-10B-A1.8B - это:
- Mixture of Experts (MoE) модель
- 10B параметров всего, ~1.8B активных
- Instruct-tuned для диалогов на русском и английском
- Требует ~20-25GB VRAM на A100

## Baseline сравнение

Референсные значения для SWE-bench Lite:

| Модель | Resolved% |
|--------|-----------|
| Claude 3.5 Sonnet | 49.0% |
| GPT-4o | 38.0% |
| Claude 3 Opus | 22.0% |
| Llama 3.1 405B | 14.0% |
| DeepSeek-V2 | 12.0% |
| Mixtral 8x22B | 4.3% |

## Troubleshooting

### GPU Out of Memory
```bash
# Уменьшить утилизацию GPU памяти
./run_gigachat_swebench.sh --gpu-util 0.80

# Или уменьшить max_model_len в gigachat_swebench.yaml
```

### Docker permission denied
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### vLLM server не запускается
```bash
# Проверить логи
tail -100 ./gigachat_results/vllm_server.log

# Проверить GPU
nvidia-smi
```
