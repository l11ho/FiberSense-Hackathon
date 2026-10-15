"""Экспорт таблицы обнаруженных кластеров с русскими заголовками."""
from pathlib import Path
import csv

NAMES={'file':'Файл','type':'Тип события','first_seen':'Первое наблюдение','last_seen':'Последнее наблюдение',
'duration_observed_sec':'Длительность наблюдения, с','frames':'Число измерений','x_start':'Координата X начала',
'x_end':'Координата X конца','width_x_units':'Ширина по X','positions':'Число позиций',
'peak_x':'Координата пика X','peak_time':'Время пика','peak_y':'Значение Y в пике',
'reference_median_y':'Фоновая медиана Y','peak_delta_y':'Отклонение в пике Y',
'max_abs_delta_y':'Макс. абсолютное отклонение Y','affected_cells':'Затронутых ячеек',
'score':'Оценка приоритета','method':'Метод','assessment':'Статус'}

def make(source_dir):
    source_dir=Path(source_dir)
    with (source_dir/'anomaly_events.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        return
    with (source_dir/'anomaly_events_ru.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[NAMES.get(k,k) for k in rows[0]])
        writer.writeheader()
        for row in rows:
            converted={NAMES.get(k,k): ('Подозрительное повышение Y' if v in ('tang_nhiet','повышение')
                         else 'Подозрительное понижение Y' if v in ('giam_nhiet','понижение')
                         else 'Требует проверки' if v in ('thong_ke_can_xac_minh','требует проверки')
                         else 'Медиана по X + общая коррекция; MAD; пороги 3 и 5' if k=='method'
                         else v) for k,v in row.items()}
            writer.writerow(converted)
if __name__=='__main__':
    make(Path(__file__).resolve().parent)
