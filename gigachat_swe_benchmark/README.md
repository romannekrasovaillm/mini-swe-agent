# GigaChat3-10B SWE-bench Benchmark

Бенчмарк модели [GigaChat3-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B) на [SWE-bench](https://www.swebench.com/) с использованием [mini-swe-agent](https://github.com/klieret/mini-swe-agent).

## Требования

- **GPU**: NVIDIA A100 40GB+ (или аналог с 40GB+ VRAM)
- **Python**: 3.10+
- **CUDA**: 12.0+
- **RAM**: 32GB+
- **Диск**: 50GB+ свободного места

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/YOUR_USERNAME/gigachat-swe-benchmark.git
cd gigachat-swe-benchmark
```

### 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Установка зависимостей

```bash
# Установить mini-swe-agent
pip install -e .

# Установить vLLM для inference
pip install vllm>=0.6.0 huggingface_hub transformers

# Установить зависимости для аналитики
pip install pandas matplotlib pyyaml
```

### 4. Скачивание модели

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ai-sage/GigaChat3-10B-A1.8B')
"
```

### 5. Запуск vLLM сервера

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model ai-sage/GigaChat3-10B-A1.8B \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --served-model-name gigachat-10b \
    --trust-remote-code \
    --enforce-eager &

# Подождать запуска (2-5 минут)
sleep 120

# Проверить что сервер работает
curl http://localhost:8000/health
```

### 6. Запуск бенчмарка

```bash
cd gigachat_swe_benchmark

# Тест на 5 инстансах (dev split)
python3 run_swebench_local.py --subset lite --slice 0:5

# Полный бенчмарк (300 инстансов, test split)
python3 run_swebench_local.py --subset lite --split test --slice 0:300
```

### 7. Анализ результатов

```bash
python3 analyze_results.py ./gigachat_local_results --all
```

## Структура проекта

```
gigachat_swe_benchmark/
├── run_swebench_local.py       # Запуск SWE-bench без контейнеров
├── run_gigachat_swebench.sh    # Bash-скрипт полного пайплайна (требует Docker)
├── gigachat_swebench_local.yaml # Конфиг для локального запуска
├── gigachat_swebench.yaml      # Конфиг для Docker-запуска
├── model_registry.json         # Регистрация модели в LiteLLM
├── analyze_results.py          # Анализатор результатов
├── start_vllm_server.py        # Скрипт запуска vLLM
└── README.md                   # Документация
```

## Результаты

После запуска результаты сохраняются в `gigachat_local_results/`:

```
gigachat_local_results/
├── preds.json                  # Предсказания (патчи) в формате SWE-bench
├── analysis_report.json        # JSON отчет
├── results_detailed.csv        # Детальные результаты
├── results_summary.csv         # Сводка по репозиториям
├── plots/                      # Графики
│   ├── exit_status_distribution.png
│   ├── step_distribution.png
│   └── repo_performance.png
└── {instance_id}/              # Траектории по инстансам
    └── {instance_id}.traj.json
```

## Параметры запуска

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--subset` | Датасет: `lite`, `verified`, `full` | `lite` |
| `--split` | Сплит: `dev` (23), `test` (300) | `dev` |
| `--slice` | Срез инстансов (например `0:10`) | все |
| `--filter` | Фильтр по regex | - |
| `-m` | Переопределить модель | - |
| `-c` | Путь к конфигу | `gigachat_swebench_local.yaml` |
| `-o` | Директория результатов | `gigachat_local_results` |

## Деплой результатов в удалённый репозиторий

### Создание нового репозитория

```bash
# 1. Создайте репозиторий на GitHub (через веб-интерфейс или gh cli)
gh repo create gigachat-swe-results --public --description "GigaChat SWE-bench results"

# Или вручную на https://github.com/new
```

### Загрузка результатов

```bash
# 2. Перейти в директорию с результатами
cd gigachat_local_results

# 3. Инициализировать git
git init
git add .
git commit -m "GigaChat3-10B SWE-bench results - $(date +%Y-%m-%d)"

# 4. Добавить remote и запушить
git remote add origin https://github.com/YOUR_USERNAME/gigachat-swe-results.git
git branch -M main
git push -u origin main
```

### Загрузка проекта (без модели)

```bash
# Из корня проекта mini-swe-agent
cd /path/to/mini-swe-agent

# Создать новый репозиторий для проекта
gh repo create gigachat-swe-benchmark --public

# Скопировать только нужные файлы (без модели и результатов)
mkdir -p /tmp/gigachat-deploy
cp -r gigachat_swe_benchmark/*.py /tmp/gigachat-deploy/
cp -r gigachat_swe_benchmark/*.yaml /tmp/gigachat-deploy/
cp -r gigachat_swe_benchmark/*.json /tmp/gigachat-deploy/
cp -r gigachat_swe_benchmark/*.sh /tmp/gigachat-deploy/
cp -r gigachat_swe_benchmark/README.md /tmp/gigachat-deploy/

# Запушить
cd /tmp/gigachat-deploy
git init
git add .
git commit -m "Initial commit: GigaChat SWE-bench benchmark setup"
git remote add origin https://github.com/YOUR_USERNAME/gigachat-swe-benchmark.git
git branch -M main
git push -u origin main
```

## Baseline сравнение

| Модель | SWE-bench Lite Resolved% |
|--------|--------------------------|
| Claude 3.5 Sonnet | 49.0% |
| GPT-4o | 38.0% |
| Claude 3 Opus | 22.0% |
| Llama 3.1 405B | 14.0% |
| DeepSeek-V2 | 12.0% |
| Mixtral 8x22B | 4.3% |
| **GigaChat3-10B** | **TBD** |

## Troubleshooting

### CUDA Out of Memory

```bash
# Уменьшить контекст
--max-model-len 16384

# Или уменьшить использование памяти
--gpu-memory-utilization 0.80
```

### vLLM не запускается

```bash
# Проверить CUDA
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Проверить версию vLLM
pip show vllm
```

### Ошибка Context Window Exceeded

Уменьшите `max_tokens` в конфиге `gigachat_swebench_local.yaml`:
```yaml
model:
  model_kwargs:
    max_tokens: 1024  # уменьшить
```

## Лицензия

MIT

## Ссылки

- [GigaChat3-10B-A1.8B](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)
- [SWE-bench](https://www.swebench.com/)
- [mini-swe-agent](https://github.com/klieret/mini-swe-agent)
- [vLLM](https://github.com/vllm-project/vllm)
