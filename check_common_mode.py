"""Диагностика общего компонента колебаний между 10 зонами FiberSense.

Скопируйте рядом с analysis.py и запустите:
    python check_common_mode.py

CSV 4.csv и 5.csv должны лежать на уровень выше папки проекта.
Результаты создаются в папке common_mode_results/.
Это статистическая диагностика, а НЕ доказательство неисправности прибора.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
try:
    import analysis
except ImportError as exc:
    raise SystemExit('ОШИБКА: скопируйте check_common_mode.py в папку рядом с analysis.py') from exc

OUT = BASE / 'common_mode_results'
OUT.mkdir(exist_ok=True)


def matrix_stat(c):
    vals = c[np.triu_indices(c.shape[0], 1)]
    return {'median_abs': float(np.median(np.abs(vals))),
            'mean_abs': float(np.mean(np.abs(vals))),
            'median_signed': float(np.median(vals)),
            'strong_05': int(np.sum(np.abs(vals) >= 0.5)),
            'pairs': len(vals)}


def corrcoef_cols(x):
    return np.corrcoef(x, rowvar=False)


def remove_shared_component(x):
    """Для каждой пары двух зон компонент оценивается по ОСТАЛЬНЫМ 8 зонам.
    Исключаем обе тестируемые зоны из опорного сигнала, чтобы не получить
    искусственную корреляцию из-за вычитания собственных измерений.
    """
    k = x.shape[1]
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            others = [q for q in range(k) if q not in (i,j)]
            shared = np.median(x[:, others], axis=1)
            design = np.column_stack((np.ones(len(shared)), shared))
            a = x[:, i] - design @ np.linalg.lstsq(design, x[:, i], rcond=None)[0]
            b = x[:, j] - design @ np.linalg.lstsq(design, x[:, j], rcond=None)[0]
            corr[i,j] = corr[j,i] = np.corrcoef(a,b)[0,1]
    return corr


def component_diagnostics(x):
    k=x.shape[1]
    explanation=[]
    for i in range(k):
        factor=np.median(x[:,[z for z in range(k) if z!=i]],axis=1)
        r=float(np.corrcoef(x[:,i],factor)[0,1])
        explanation.append(r)
    z=(x-x.mean(axis=0))/x.std(axis=0, ddof=1)
    s=np.linalg.svd(z,compute_uv=False)
    explained=float(s[0]**2/np.sum(s**2))
    return np.asarray(explanation), explained


def circular_shift_test(x, repeat=250, seed=20261015):
    """Приближённый контроль: сдвигаем зоны отдельно, разрушая синхронность.
    Не является строгим p-value при нестационарности временных рядов.
    """
    rng=np.random.default_rng(seed)
    n,k=x.shape
    obs=matrix_stat(corrcoef_cols(x))['median_abs']
    null=[]
    for b in range(repeat):
        shifted=np.column_stack([np.roll(x[:,i], int(rng.integers(1,n))) for i in range(k)])
        null.append(matrix_stat(corrcoef_cols(shifted))['median_abs'])
    return obs,float(np.quantile(null,0.95)),int(np.sum(np.asarray(null)>=obs)),len(null)


def write_matrix(path, names, matrix):
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        w=csv.writer(file)
        w.writerow(['Зона']+names)
        for label,row in zip(names,matrix):
            w.writerow([label]+[f'{v:.6f}' for v in row])


def draw_image(fname, names, before, after, timestamps, shared, title):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return False
    plt.rcParams.update({'font.family':'DejaVu Sans'})
    fig, axs=plt.subplots(2,2,figsize=(14.2,9),height_ratios=[2.6,1.0],layout='constrained')
    for ax,mat,lab in [(axs[0,0],before,'Корреляция первых разностей'),
                       (axs[0,1],after,'Частная корреляция после учёта общего компонента')]:
        im=ax.imshow(mat, vmin=-1,vmax=1,cmap='RdBu_r',origin='upper')
        ax.set_title(lab, fontsize=11, pad=10)
        ax.set_xticks(np.arange(len(names)),[f'R{i+1}' for i in range(len(names))])
        ax.set_yticks(np.arange(len(names)),[f'R{i+1}' for i in range(len(names))])
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7,
                        color='white' if abs(mat[i,j])>.67 else '#202020')
        fig.colorbar(im,ax=ax,shrink=.72,label='Коэффициент r')
    axs[1,0].plot(timestamps,shared,lw=.7)
    axs[1,0].axhline(0,lw=.7,ls='--')
    axs[1,0].set_title('Медианное общее изменение (10 зон)',fontsize=10)
    axs[1,0].set_ylabel('ΔY за замер')
    axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[1,0].tick_params(axis='x',rotation=25)
    b=matrix_stat(before)['median_abs'];a=matrix_stat(after)['median_abs']
    axs[1,1].bar(['До','После'],[b,a])
    axs[1,1].set_title('Медиана |r| для 45 пар',fontsize=10)
    axs[1,1].set_ylim(0,1)
    for i,val in enumerate([b,a]):axs[1,1].text(i,val+.03,f'{val:.3f}',ha='center')
    fig.suptitle(title,fontsize=15,fontweight='bold')
    fig.savefig(OUT/fname,dpi=135)
    plt.close(fig)
    return True


def inspect_file(csv_path):
    print('ЗАГРУЗКА:',csv_path.name,flush=True)
    d=analysis.load_csv(csv_path)
    reg=analysis.region_stats(d)
    names=reg['regions']
    means=reg['means']
    dt=np.diff([t.timestamp() for t in d['ts']])
    first=np.diff(means,axis=0)
    valid=(dt>0)&np.all(np.isfinite(first),axis=1)
    arr=first[valid]
    seconds=dt[valid]
    times=np.asarray(d['ts'][1:],dtype=object)[valid]
    before=corrcoef_cols(arr)
    after=remove_shared_component(arr)
    baseline=matrix_stat(before)
    adjusted=matrix_stat(after)
    loo,pc1=component_diagnostics(arr)
    shared=np.median(arr,axis=1)
    obs,null95,nnull,sim_n=circular_shift_test(arr)
    print(f"  Пары зон: {baseline['pairs']}; интервалов: {len(arr)}; исключено некорректных интервалов: {len(dt)-len(arr)}")
    print(f"  Медиана |r| до: {baseline['median_abs']:.3f}; после: {adjusted['median_abs']:.3f}")
    print(f'  Доля вариации стандартизованных изменений в PC1: {pc1:.1%}')
    print(f'  Контроль случайных сдвигов: фактически {obs:.3f}, 95-й процентиль {null95:.3f}')
    write_matrix(OUT/f'corr_{csv_path.stem}_diff_before.csv',names,before)
    write_matrix(OUT/f'corr_{csv_path.stem}_diff_after_common.csv',names,after)
    raw_pairs=matrix_stat(before)
    report=[
      f'ФАЙЛ: {csv_path.name}',
      f'Наблюдений: {len(d["ts"])}, пригодных временных интервалов: {len(arr)}, исключено: {len(dt)-len(arr)}',
      f'Зон: {len(names)}, пар зон: {raw_pairs["pairs"]}',
      f'Медиана |r| по разностям до коррекции: {baseline["median_abs"]:.4f}',
      f'Медиана |r| после исключения общего компонента: {adjusted["median_abs"]:.4f}',
      f'Количество пар с |r| >= 0.5: {baseline["strong_05"]} до; {adjusted["strong_05"]} после',
      f'Первая главная компонента стандартизованных разностей (PCA): {pc1:.1%} общей вариации',
      f'Корреляция каждой зоны с медианой ОСТАЛЬНЫХ зон: ' + ', '.join(f'R{i+1}={v:+.3f}' for i,v in enumerate(loo)),
      f'Контроль независимыми циклическими сдвигами: наблюдаемая медиана |r|={obs:.4f}, 95-й процентиль после рассинхронизации={null95:.4f} (экспериментов={sim_n})',
    ]
    # Сравнение разностей и скоростей изменений, учитывает нерегулярные интервалы времени
    velocity=arr/seconds[:,None]
    speedcorr=corrcoef_cols(velocity)
    report.append(f'Медиана |r| корреляций скоростей ΔY/Δt: {matrix_stat(speedcorr)["median_abs"]:.4f}')
    # Проверяем, что экстремальные интервалы не доминируют во всём результате.
    cutoffs=np.quantile(np.abs(arr),0.99,axis=0)
    regular=np.all(np.abs(arr)<=cutoffs[None,:],axis=1)
    if regular.sum()>=50:
        stable=corrcoef_cols(arr[regular])
        stable_after=remove_shared_component(arr[regular])
        report.append(f'После отсечения верхнего 1% |ΔY| в каждой зоне: оставлено {regular.sum()}/{len(arr)} интервалов; медиана |r|={matrix_stat(stable)["median_abs"]:.4f}, после общего компонента={matrix_stat(stable_after)["median_abs"]:.4f}')
    i,j=2,6 # R3, R7
    report.append(f'R3–R7: исходные разности r={before[i,j]:+.4f}, после общего компонента r={after[i,j]:+.4f}, по скоростям r={speedcorr[i,j]:+.4f}')
    report.append('ИНТЕРПРЕТАЦИЯ: если |r| существенно падает, есть синхронный общий компонент, но доказать, что это именно шум прибора, нельзя.')
    report.append('ОГРАНИЧЕНИЯ: метод с исключением двух исследуемых зон из опорного сигнала; коррекция служит диагностике, а не автоматической фильтрации физического сигнала.')
    report.append('ОГРАНИЧЕНИЯ: корреляционные матрицы с попарной регрессией являются матрицами попарных коэффициентов; они могут не быть положительно полуопределёнными.')
    report.append('ОГРАНИЧЕНИЯ: циклические сдвиги используются как ориентировочный контроль синхронности, без строгих выводов о p-value.')
    made=draw_image(f'common_mode_{csv_path.stem}.png',names,before,after,times,shared,f'{csv_path.name}: диагностика общего компонента')
    report.append(f'График создан: {made}')
    return report


def main():
    print('ДИАГНОСТИКА ОБЩЕГО КОМПОНЕНТА FIBERSENSE')
    all_lines=['ДИАГНОСТИКА ОБЩЕГО КОМПОНЕНТА FIBERSENSE',
               'Статистическое совпадение сигналов НЕ доказывает аппаратный шум.', '']
    for filename in ['4.csv','5.csv']:
        csv_path=BASE.parent/filename
        if not csv_path.is_file():
            msg=f'ПРОПУЩЕН: {filename}. Ожидается в {BASE.parent}'
            print(msg);all_lines.append(msg)
            continue
        report=inspect_file(csv_path)
        all_lines.extend(report+[''])
    result=OUT/'common_mode_report_ru.txt'
    result.write_text('\n'.join(all_lines)+'\n',encoding='utf-8')
    print('ОТЧЁТ:',result)
    print('ГОТОВО.')

if __name__=='__main__': main()
