# yolo-self-training

Пайплайн self-training детектора YOLO (ultralytics) на псевдоразметке:
учитель размечает неразмеченный пул → пороги уверенности калибруются по ручной
разметке → ученик обучается на «ручная разметка + отобранная псевдоразметка» →
раунд принимается, только если стало лучше на замороженной ручной валидации.

Подробное описание подхода, правил и типовых ошибок — **[docs/APPROACH.md](docs/APPROACH.md)**.
Начинать лучше с него.

## Установка

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # ultralytics (+ torch подтянется сам)
pip install pytest                       # для тестов
```

Для GPU поставьте сборку torch под свою версию CUDA до ultralytics
(см. https://pytorch.org/get-started/locally/).

## Быстрая проверка

```bash
make test     # юнит-тесты чистой логики, без GPU и без ultralytics
make smoke    # весь цикл на синтетических данных, CPU, ~2 минуты
```

`make smoke` генерирует `data/synthetic/`, прогоняет раунд 0 и два раунда
self-training и пишет результаты в `runs/smoke/`. Метрики на синтетике ничего не
значат — это проверка, что окружение и цикл работают.

## Запуск на своих данных

1. Разметка в формате YOLO: `…/images/<группа>/<файл>.jpg` и
   `…/labels/<группа>/<файл>.txt` (строка на бокс: `cls cx cy w h`, нормированные).
   Неразмеченный пул — любая папка с изображениями.
2. Скопируйте `configs/default.yaml`, заполните `names`, `data.*`, `groups.regex`.
3. Проверьте группы: `python -m selftrain.split --config my.yaml --dry-run`.
4. Запустите цикл: `python -m selftrain.loop --config my.yaml`.
   Повторный запуск продолжает с первого незавершённого раунда.

## Что получается на выходе

```
<workdir>/
  splits/{train,val,test,pool}.txt, groups.json   # замороженный сплит
  rounds/<k>/
    thresholds.json      # пороги t_low / t_high по классам + precision/recall на val
    preds_val.jsonl, preds_pool.jsonl   # сырые предсказания учителя (кэш)
    pseudo_stats.json    # сколько изображений/боксов пошло в обучение
    review_queue.csv     # изображения из серой зоны — кандидаты на ручную разметку
    data.yaml, train.txt, data/         # датасет раунда
    train/weights/best.pt
    metrics.json         # mAP на GT-val: общий, по классам, по группам
    result.json          # принят ли раунд и почему
  history.json           # все раунды, лучший раунд
  final.json             # лучшая модель + оценка на GT-test
```

## Структура кода

| Модуль | Что делает |
| --- | --- |
| `selftrain/split.py` | сплит по группам, заморозка, исключение val/test-групп из пула |
| `selftrain/calibrate.py` | сопоставление предсказаний с GT, пороги по классам под целевую точность |
| `selftrain/filter.py` | три зоны уверенности, мин. размер, временная согласованность, очередь на разметку |
| `selftrain/merge.py` | отбор псевдоразметки (редкие классы первыми, лимиты), сборка датасета раунда |
| `selftrain/decide.py` | правила приёма раунда и остановки |
| `selftrain/yolo_ops.py` | обёртки ultralytics: predict / train / evaluate (общий и по группам) |
| `selftrain/loop.py` | оркестрация раундов, резюмирование |
| `scripts/make_synthetic.py` | синтетический датасет для smoke-теста |

Все модули, кроме `yolo_ops.py` и `loop.py`, не зависят от ultralytics и покрыты тестами.
