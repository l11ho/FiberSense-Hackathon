"""Проверка алгоритмов FiberSense без изменений основного проекта.
Поместите файл рядом с analysis.py и запустите: python verify_analysis.py
Отчёт: verification_results_ru.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
try:
    import analysis
except ImportError as exc:
    raise SystemExit('Поместите verify_analysis.py в папку FiberSense_RU рядом с analysis.py') from exc

lines = []
def log(msg):
    print(msg, flush=True)
    lines.append(str(msg))

def must(condition, label):
    if not condition:
        raise AssertionError(label)
    log('PASS: ' + label)


def make_synthetic(heated=True):
    rng = np.random.default_rng(42)
    t = np.arange(80)
    x = np.arange(50, dtype=float)
    base = 20.0 + 0.8*np.sin(x/8.)
    drift = 0.35*np.sin(t/14.)
    y = base[None,:] + drift[:,None] + rng.normal(0,.13,(len(t),len(x)))
    if heated:
        y[18:43,12:19] += 7
        y[55:69,32:38] -= 7
    return {'x':x, 'y':y.astype('float32'), 'ts':[datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(seconds=10*int(i)) for i in t]}


def independent_corr(x,y):
    good=np.isfinite(x) & np.isfinite(y)
    a=np.asarray(x[good],dtype=np.float64)
    b=np.asarray(y[good],dtype=np.float64)
    if len(a)<10 or np.std(a)<1e-10 or np.std(b)<1e-10:
        return np.nan
    return float(pearsonr(a,b).statistic)


def tests_synthetic():
    log('=== 1. Тесты с известной истиной (искусственные данные) ===')
    empty,_=analysis.calculate_anomalies(make_synthetic(False), 'synthetic',min_delta=3,robust_z=5)
    must(len(empty)==0, 'Без добавленных событий алгоритм не создаёт ложных тревог на этом тестовом сигнале')
    ev,dv=analysis.calculate_anomalies(make_synthetic(True), 'synthetic',min_delta=3,robust_z=5)
    hot=[e for e in ev if e['type']=='повышение']
    cold=[e for e in ev if e['type']=='понижение']
    def match_event(events, xl, xh, start, stop):
        return any(e['x_start']<=xh and e['x_end']>=xl and e['frames']>=3 and
                   e['x_start']<=xh and e['x_end']>=xl and
                   any(start <= k < stop and e['first_seen']<=analysis.fmt_ts(make_synthetic(True)['ts'][k])<=e['last_seen'] for k in [start, (start+stop)//2, stop-1]) for e in events)
    must(match_event(hot,12,18,18,43), 'Добавленный нагрев (+7) найден')
    must(match_event(cold,32,37,55,69), 'Добавленное охлаждение (-7) найдено')
    log(f'  Искусственные события: {len(hot)} повышений, {len(cold)} понижений')
    return True


def validate_real(path):
    log(f'=== 2. Проверка файла {path.name} ===')
    d=analysis.load_csv(path)
    y=d['y']; x=d['x']; ts=d['ts']
    must(np.all(np.diff(x)>0),f'{path.name}: сетка X строго возрастает')
    must(all(t2>t1 for t1,t2 in zip(ts[:-1],ts[1:])), f'{path.name}: метки времени строго возрастают')
    must(not np.any(y == -300),f'{path.name}: -300 исключено из вычислений')
    must(np.sum(np.any(np.isfinite(y),axis=0))>0,f'{path.name}: существуют корректные измерения')
    summ=analysis.summarize_data(d)
    reg=analysis.region_stats(d)
    for meth in ('corr_raw','corr_detr','corr_diff','corr_spearman'):
        c=reg[meth]
        must(c.shape==(10,10) and np.allclose(c,c.T,equal_nan=True,atol=1e-10),f'{path.name}: матрица {meth} симметрична (10×10)')
        must(np.all(np.abs(c[np.isfinite(c)])<=1+1e-10),f'{path.name}: {meth} в диапазоне [-1,1]')
    i,j=2,6  # R3 и R7
    a=reg['means']
    r_ind=independent_corr(a[:,i],a[:,j])
    must(np.isclose(r_ind,reg['corr_raw'][i,j],atol=1e-10),f'{path.name}: Pearson R3–R7 совпал с независимым scipy.stats.pearsonr')
    s_ind=spearmanr(a[:,i],a[:,j],nan_policy='omit').statistic
    must(np.isclose(s_ind,reg['corr_spearman'][i,j],atol=1e-10),f'{path.name}: Spearman R3–R7 совпал с scipy.stats.spearmanr')
    d_ind=independent_corr(np.diff(a[:,i]),np.diff(a[:,j]))
    must(np.isclose(d_ind,reg['corr_diff'][i,j],atol=1e-10),f'{path.name}: корреляция первых разностей рассчитана верно')
    events, _ = analysis.calculate_anomalies(d,path.name)
    normal_count=len([e for e in events if e['type']=='повышение'])
    cold_count=len([e for e in events if e['type']=='понижение'])
    log(f'  Строк: {len(ts)}, корректных позиций: {summ["valid_positions"]}, значений -300: {summ["sentinel_minus_300_cells"]}')
    log(f'  События при порогах min_delta=3, MAD=5: повышений {normal_count}, понижений {cold_count}')
    log(f'  R3–R7: raw={reg["corr_raw"][i,j]:.4f}, detrended={reg["corr_detr"][i,j]:.4f}, diff={reg["corr_diff"][i,j]:.4f}')
    # Чувствительность: высокая оценка не гарантирует физическую аварию.
    base_top=events[0] if events else None
    if base_top:
        px=base_top['peak_x']
        for minimum in (2.0,3.0,4.0):
            ev,_ = analysis.calculate_anomalies(d,path.name,min_delta=minimum,robust_z=5)
            hot=[e for e in ev if e['type']=='повышение']
            nearby=[e for e in hot if e['x_start']-1e-5 <= px <= e['x_end']+1e-5]
            log(f'  Порог ΔY≥{minimum:g}, MAD×5: {len(hot)} групп повышения; группы, включающие пик X={px}: {len(nearby)}')
        for k in (4.0,5.0,6.0):
            ev,_ = analysis.calculate_anomalies(d,path.name,min_delta=3,robust_z=k)
            hot=[e for e in ev if e['type']=='повышение']
            nearby=[e for e in hot if e['x_start']-1e-5 <= px <= e['x_end']+1e-5]
            log(f'  Порог ΔY≥3, MAD×{k:g}: {len(hot)} групп повышения; группы, включающие пик X={px}: {len(nearby)}')
        # Диагностика: нельзя оценить precision/recall на реальных данных без разметки.
    if path.name=='5.csv':
        dt=np.diff([v.timestamp() for v in ts])
        mins=reg['means']
        di=np.diff(mins,axis=0)
        rates=di / dt[:,None]
        rrate=independent_corr(rates[:,i],rates[:,j])
        log(f'  Учёт неравномерного шага: R3–R7 корреляция разностей={reg["corr_diff"][i,j]:.4f}, скоростей ΔY/Δt={rrate:.4f}')
        must(summ['sentinel_minus_300_cells']>0,'5.csv: служебные значения выявлены и исключены')
    return events


def main():
    log('ПРОВЕРКА АЛГОРИТМОВ FIBERSENSE')
    log('Важно: прохождение тестов НЕ доказывает обнаружение реальных аварий кабеля.')
    try:
        tests_synthetic()
        for filename in ('4.csv','5.csv'):
            path=BASE.parent/filename
            if path.exists():
                validate_real(path)
            else:
                log(f'SKIP: {filename} не найден рядом с папкой проекта. Проверены только искусственные тесты.')
        log('ЗАКЛЮЧЕНИЕ: указанные программные и численные проверки завершены.')
        log('ОГРАНИЧЕНИЯ: требуется документация о единицах X/Y, коде -300, паспортных порогах и реальные метки отказов.')
    except Exception as err:
        log(f'FAIL: {type(err).__name__}: {err}')
        raise
    finally:
        (BASE/'verification_results_ru.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(f'Отчёт записан: {BASE/"verification_results_ru.txt"}',flush=True)

if __name__=='__main__':main()
