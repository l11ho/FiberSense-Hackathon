"""Анализ данных распределённых волоконно-оптических датчиков из CSV.
Запуск: python analysis.py --data-dir PATH --out-dir PATH.
Зависимости: NumPy, SciPy. Метки реальных отказов неизвестны.
"""
from __future__ import annotations
import csv, json, argparse, math
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from scipy import ndimage, stats

SENTINEL = -300.0
MIN_DELTA = 3.0                 # heuristic in raw Y units, not a certified safety threshold
ROBUST_Z = 5.0                  # robust noise multiple
MAX_CORR_POINTS = 400          # optional selection for downstream dashboard

def load_csv(path):
    ts, rows, ids, channels, invalid = [], [], [], [], []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for i, r in enumerate(csv.DictReader(f)):
            try:
                arr = np.array(json.loads(r['points']), dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 2: raise ValueError('поле points должно иметь форму N×2')
                ts.append(datetime.fromisoformat(r['timestamp']))
                rows.append(arr)
                ids.append(r['dts_id']); channels.append(r['channel_id'])
            except Exception as exc:
                invalid.append({'row':i+2, 'reason':str(exc)})
    if not rows: raise ValueError(f'Нет корректных данных в {path}')
    # берём самую длинную сетку X; укороченные записи могут обрываться в конце
    grid = rows[int(np.argmax([len(x) for x in rows]))][:, 0]
    y = np.full((len(rows),len(grid)),np.nan,dtype=np.float32)
    mismatch = []
    for i,arr in enumerate(rows):
        if len(arr)>len(grid) or not np.allclose(arr[:,0],grid[:len(arr)],rtol=1e-7,atol=1e-7):
            mismatch.append(i)
            if len(arr) and len(grid)>1:
                idx = np.searchsorted(grid,arr[:,0]); valid = (idx<len(grid)) & np.isclose(grid[np.minimum(idx,len(grid)-1)],arr[:,0],atol=1e-5)
                y[i,idx[valid]] = arr[valid,1]
        else:
            y[i,:len(arr)] = arr[:,1]
    sentinel_mask = y <= -200
    y[sentinel_mask]=np.nan
    return {'x':grid,'y':y,'ts':ts,'raw_lens':[len(r) for r in rows], 'ids':ids,
            'channels':channels,'parse_errors':invalid,'x_mismatch':mismatch,
            'sentinel_count':int(np.sum(sentinel_mask)), 'sentinel_by_col':np.sum(sentinel_mask,axis=0)}

def minutes_since_start(ts):
    return np.array([(t-ts[0]).total_seconds()/60 for t in ts],dtype=float)

def safe_num(v, n=4):
    return None if not np.isfinite(v) else round(float(v),n)

def contiguous_intervals(mask):
    d = np.diff(np.concatenate([[0],np.asarray(mask,int),[0]]))
    return list(zip(np.flatnonzero(d==1).tolist(), np.flatnonzero(d==-1).tolist()))

def fmt_ts(t): return t.isoformat(sep=' ',timespec='seconds')

def summarize_data(d):
    x,y,ts = d['x'],d['y'],d['ts']
    good=np.any(np.isfinite(y),axis=0)
    dt = np.diff([t.timestamp() for t in ts])
    q=np.nanquantile(y,[0,.01,.05,.25,.5,.75,.95,.99,1]); other=Counter(np.rint(dt).astype(int))
    med_dt=float(np.median(dt)); gaps=np.flatnonzero(dt>1.5*med_dt)
    signals=y[:,good]
    df = {
        'records': len(ts), 'channel_ids': sorted(set(d['channels'])), 'distinct_dts_ids':len(set(d['ids'])),
        'start':fmt_ts(ts[0]),'end':fmt_ts(ts[-1]),'timezone': ('UTC'+str(ts[0].utcoffset()) if ts[0].tzinfo else 'not specified'),
        'observed_duration_hours':round((ts[-1]-ts[0]).total_seconds()/3600,3),
        'raw_positions':len(x),'valid_positions':int(good.sum()),
        'valid_cells':int(np.sum(np.isfinite(y))),'missing_or_invalid_cells':int(np.sum(~np.isfinite(y))),
        'sentinel_minus_300_cells':d['sentinel_count'],
        'truncated_records':int(np.sum(np.asarray(d['raw_lens'])<len(x))),
        'raw_length_histogram':{str(int(k)):int(v) for k,v in Counter(d['raw_lens']).most_common()},
        'position_min':safe_num(x[0]),'position_max':safe_num(x[-1]),
        'position_step':safe_num(np.nanmedian(np.diff(x)),6),
        'time_step_median_sec':safe_num(med_dt,2),
        'time_step_histogram_sec':{str(int(k)):int(v) for k,v in other.most_common(12)},
        'time_gaps_over_1p5median':len(gaps), 'nonpositive_dt_count':int(np.sum(dt<=0)),
        'gap_list': [{'from':fmt_ts(ts[i]), 'to':fmt_ts(ts[i+1]), 'sec':round(float(dt[i]),1)} for i in gaps[:50]],
        'x_grid_inconsistent_rows':len(d['x_mismatch']),
        'parse_errors':len(d['parse_errors']),
        'temp_quantiles':{str(k):safe_num(v) for k,v in zip(['min','p01','p05','p25','p50','p75','p95','p99','max'],q)},
        'temporal_mean_first':safe_num(np.nanmean(signals[:min(10,len(signals))])),
        'temporal_mean_last':safe_num(np.nanmean(signals[-min(10,len(signals)):])),
    }
    sentinel_cols=np.flatnonzero(d['sentinel_by_col'] == len(ts))
    if len(sentinel_cols):
        df['permanently_invalid_x_range']=[safe_num(x[sentinel_cols[0]]),safe_num(x[sentinel_cols[-1]])]
        df['permanently_invalid_positions']=len(sentinel_cols)
    return df

def calculate_anomalies(d, name, min_delta=MIN_DELTA, robust_z=ROBUST_Z):
    """Positive and negative anomalies beyond site median, common-mode-adjusted;
    events: 4-connectivity in time x position. Filter: >= 3 pixels; wide/long events higher confidence.
    """
    y=d['y']; x=d['x']; ts=d['ts']
    keep=np.any(np.isfinite(y),axis=0)
    yf=y[:,keep]; xf=x[keep]
    base=np.nanmedian(yf,axis=0)
    residual0=yf-base[None,:]
    # убираем общую составляющую изменения температуры по всей линии
    common=np.nanmedian(residual0,axis=1)
    residual = residual0-common[:,None]
    sigma=1.4826*np.nanmedian(np.abs(residual-np.nanmedian(residual,axis=0)),axis=0)
    threshold = np.maximum(min_delta,robust_z*sigma)
    max_res=np.nanmax(residual,axis=1)
    min_res=np.nanmin(residual,axis=1)
    sx = np.median(np.diff(xf))
    results=[]
    structure=ndimage.generate_binary_structure(2,1)
    for sign,mask in [('hot',(residual>threshold[None,:])),('cold',(residual < -threshold[None,:]))]:
        mask &= np.isfinite(residual)
        components, n = ndimage.label(mask, structure)
        objs = ndimage.find_objects(components)
        for c,sl in enumerate(objs,1):
            if sl is None: continue
            sub_label=(components[sl] == c)
            npoints=int(np.sum(sub_label))
            if npoints < 3: continue
            time_ids=np.flatnonzero(np.any(sub_label,axis=1))+sl[0].start
            space_ids=np.flatnonzero(np.any(sub_label,axis=0))+sl[1].start
            ny=len(time_ids); nx=len(space_ids)
            # исключаем одиночные выбросы малого размера, сохраняя протяжённые события
            if ny==1 and nx<3: continue
            rr,cc = np.where(sub_label)
            tt=rr+sl[0].start; xx=cc+sl[1].start
            pv=residual[tt,xx]
            peak_idx=np.argmax(pv) if sign=='hot' else np.argmin(pv)
            it=tt[peak_idx]; ix=xx[peak_idx]
            # длительность наблюдения между первым и последним измерением; для одного кадра равна нулю
            duration=(ts[time_ids[-1]]-ts[time_ids[0]]).total_seconds()
            max_magnitude=float(np.max(np.abs(pv)))
            eventscore=max_magnitude * np.sqrt(min(npoints,5000))
            result={
                'file':name,'type':'повышение' if sign=='hot' else 'понижение',
                'first_seen':fmt_ts(ts[time_ids[0]]),'last_seen':fmt_ts(ts[time_ids[-1]]),
                'duration_observed_sec':round(duration,1),'frames':ny,
                'x_start':safe_num(xf[space_ids[0]],3),'x_end':safe_num(xf[space_ids[-1]],3),
                'width_x_units':safe_num(xf[space_ids[-1]]-xf[space_ids[0]]+sx,3),'positions':nx,
                'peak_x':safe_num(xf[ix],3),'peak_time':fmt_ts(ts[it]),
                'peak_y':safe_num(yf[it,ix],3), 'reference_median_y':safe_num(base[ix],3),
                'peak_delta_y':safe_num(pv[peak_idx],3),
                'max_abs_delta_y':round(max_magnitude,3),
                'affected_cells':npoints,'score':round(eventscore,2),
                'method':'Медиана по X + пространственная медиана; порог 5*MAD и >=3 единиц Y',
                'assessment':'требует проверки'
            }
            results.append(result)
    results.sort(key=lambda z:z['score'],reverse=True)
    return results, {'keep':keep,'x':xf,'y':yf,'baseline':base,'common':common,'residual':residual,
                     'threshold':threshold,'sigma':sigma,'max_res':max_res,'min_res':min_res}

def pairwise_corr(a,b=None,method='pearson'):
    if b is None: b=a
    k=a.shape[1]; l=b.shape[1]; out=np.eye(k,l)
    for i in range(k):
        for j in range(l):
            both=np.isfinite(a[:,i])&np.isfinite(b[:,j])
            if np.sum(both)>=10 and np.std(a[both,i])>1e-10 and np.std(b[both,j])>1e-10:
                if method=='pearson': out[i,j]=np.corrcoef(a[both,i],b[both,j])[0,1]
                else: out[i,j]=stats.spearmanr(a[both,i],b[both,j]).statistic
            elif i!=j: out[i,j]=np.nan
    return out

def region_stats(d, regions=10):
    y=d['y']; x=d['x'];ts=d['ts']; valid=np.flatnonzero(np.any(np.isfinite(y),axis=0));
    blocks=np.array_split(valid,regions)
    # неполные строки могут оставить пустые зоны; сохраняем NaN и используем только общие наблюдения
    means=np.column_stack([np.where(np.any(np.isfinite(y[:,idx]),axis=1),
                         np.nansum(y[:,idx],axis=1)/np.maximum(np.sum(np.isfinite(y[:,idx]),axis=1),1),
                         np.nan) for idx in blocks]);
    region_names=[f'R{i+1} ({x[idx[0]]:.1f}-{x[idx[-1]]:.1f})' for i,idx in enumerate(blocks)]
    r=pairwise_corr(means)
    mins=minutes_since_start(ts)
    b=np.stack([np.ones_like(mins),mins],axis=1)
    detr=np.full_like(means,np.nan)
    for j in range(regions):
        mask=np.isfinite(means[:,j]);coeff=np.linalg.lstsq(b[mask],means[mask,j],rcond=None)[0]
        detr[mask,j]=means[mask,j]-b[mask]@coeff
    rd=pairwise_corr(detr)
    delta=np.diff(means,axis=0)
    rdif=pairwise_corr(delta)
    r2=pairwise_corr(means,method='spearman')
    return {'means':means,'regions':region_names,'corr_raw':r,'corr_detr':rd,
            'corr_diff':rdif,'corr_spearman':r2,'detrended':detr}

def top_pairs(m,names,n=6):
    upper=np.triu_indices(len(names),1)
    ranked=np.argsort(-np.abs(m[upper]))[:n]
    return [{'area_1':names[upper[0][i]],'area_2':names[upper[1][i]],'r':safe_num(m[upper[0][i],upper[1][i]],3)} for i in ranked]

def detect_sampling_runs(d):
    """Непрерывные участки с периодичностью около 10/20 секунд."""
    ts=d['ts']; dt=np.diff([t.timestamp() for t in ts]);
    typ=(np.where(dt<15,'~10s',np.where(dt<26,'~20s','gap')) if np.median(dt)<30 else np.where(dt<180,'~120s','gap'))
    changes=np.flatnonzero(typ[1:]!=typ[:-1])+1
    bounds=[0,*changes,len(typ)]; runs=[]
    for st,en in zip(bounds[:-1],bounds[1:]):
        if en-st >=10:
            runs.append({'from':fmt_ts(ts[st]),'to':fmt_ts(ts[en]),'interval_type':str(typ[st]),'count_intervals':int(en-st)})
    return sorted(runs,key=lambda v:v['count_intervals'],reverse=True)[:12]

def main(data_dir, out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    summaries={}; data={}; deriv={}; corrs={}; all_events=[]
    for name in ('4.csv','5.csv'):
        print('ЗАГРУЗКА',name,flush=True)
        d=load_csv(Path(data_dir)/name)
        s=summarize_data(d)
        events,drv=calculate_anomalies(d,name)
        reg=region_stats(d)
        s['sampling_runs']=detect_sampling_runs(d)
        s['hot_event_count']=sum(e['type']=='повышение' for e in events)
        s['cold_event_count']=sum(e['type']=='понижение' for e in events)
        s['largest_events']=events[:15]
        s['top_correlations_raw']=top_pairs(reg['corr_raw'],reg['regions'])
        s['top_correlations_detrended']=top_pairs(reg['corr_detr'],reg['regions'])
        s['top_correlations_first_diff']=top_pairs(reg['corr_diff'],reg['regions'])
        # Сохраняем полные матрицы корреляций, а не только лучшие пары.
        for k,lab in [('corr_raw','pearson_raw'),('corr_detr','pearson_detrended'),('corr_diff','pearson_first_difference'),('corr_spearman','spearman_raw')]:
            with open(out_dir/f'correlation_{name[0]}_{lab}.csv','w',encoding='utf-8-sig',newline='') as cm:
                writer=csv.writer(cm);writer.writerow(['region']+reg['regions'])
                for region,vals in zip(reg['regions'],reg[k]):writer.writerow([region]+[round(float(v),6) if np.isfinite(v) else '' for v in vals])
        summaries[name]=s; data[name]=d; deriv[name]=drv; corrs[name]=reg; all_events+=events
        print('СВОДКА',name,json.dumps({k:v for k,v in s.items() if k in ['records','hot_event_count','cold_event_count','time_step_median_sec']},ensure_ascii=False),'top',[(e['type'],e['peak_x'],e['peak_delta_y'],e['frames'],e['positions']) for e in events[:8]],flush=True)
    with open(out_dir/'summary.json','w',encoding='utf-8') as f: json.dump(summaries,f,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
    fields=['file','type','first_seen','last_seen','duration_observed_sec','frames','x_start','x_end','width_x_units','positions','peak_x','peak_time','peak_y','reference_median_y','peak_delta_y','max_abs_delta_y','affected_cells','score','method','assessment']
    with open(out_dir/'anomaly_events.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_events)
    return summaries,data,deriv,corrs,all_events

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent));a.add_argument('--out-dir',default=str(Path(__file__).resolve().parent))
    args=a.parse_args();main(args.data_dir,args.out_dir)
