"""Построение графиков на основе двух исходных CSV-файлов."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from analysis import main, minutes_since_start
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':12,
                     'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':175,'figure.facecolor':'white'})

def ticks_as_time(ax,ts, n=6):
    ids=np.unique(np.linspace(0,len(ts)-1,n,dtype=int))
    ax.set_yticks(ids);ax.set_yticklabels([ts[i].strftime('%H:%M') for i in ids]); ax.set_ylabel('Время (часы из файла)')

def save(fig,path):
    fig.savefig(path,bbox_inches='tight',facecolor='white',dpi=175);plt.close(fig)

def overview(d,drv,name,p):
    ts=d['ts'];y=drv['y'];x=drv['x'];base=drv['baseline'];resid=drv['residual']
    fig=plt.figure(figsize=(13,9.5),constrained_layout=True)
    gs=fig.add_gridspec(2,2,height_ratios=[1.3,0.9])
    ax=fig.add_subplot(gs[0,0]); lo,hi=np.nanpercentile(y,[1,99.8]);hi=max(hi,np.nanmax(y)-1)
    im=ax.imshow(y,aspect='auto',origin='lower',extent=[x[0],x[-1],0,len(ts)-1],interpolation='nearest',cmap='inferno',vmin=lo,vmax=hi)
    fig.colorbar(im,ax=ax,label='Y (единица не подтверждена)',shrink=.78)
    ax.set(xlabel='Координата X',title=f'{name}: исходная тепловая карта');ticks_as_time(ax,ts)
    ax=fig.add_subplot(gs[0,1]); im=ax.imshow(resid,aspect='auto',origin='lower',extent=[x[0],x[-1],0,len(ts)-1],interpolation='nearest',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=14))
    fig.colorbar(im,ax=ax,label='Отклонение Y от фона',shrink=.78)
    ax.set(xlabel='Координата X',title='Отклонение от локального фона');ticks_as_time(ax,ts)
    ax=fig.add_subplot(gs[1,0]);
    mins=minutes_since_start(ts)/60;ax.plot(mins,np.nanmedian(y,axis=1),label='Медиана всей линии',lw=1.3)
    ax.plot(mins,np.nanquantile(y,.99,axis=1),label='99-й процентиль',lw=1.1)
    ax.plot(mins,np.nanmax(y,axis=1),label='Максимум',lw=.9,alpha=.7)
    ax.set(title='Динамика профиля по времени',xlabel='Часы с начала записи',ylabel='Значение Y');ax.legend(fontsize=8);ax.grid(alpha=.18)
    ax=fig.add_subplot(gs[1,1]);ax.plot(x,base,alpha=.9,lw=1.3,label='Временная медиана')
    for idx,label in [(0,'Первый замер'),(len(y)//2,'Середина записи'),(len(y)-1,'Последний замер')]:
        ax.plot(x,y[idx],label=label,lw=.85,alpha=.75)
    ax.set(title='Профиль Y по координате',xlabel='Координата X',ylabel='Значение Y');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.18)
    fig.suptitle('РАСПРЕДЕЛЁННЫЕ ВОЛОКОННЫЕ ДАТЧИКИ — ОБЗОР',fontsize=17,fontweight='bold')
    save(fig,p)

def correlation_panels(reg,name,p):
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    for ax,k,title in zip(axes,['corr_raw','corr_detr','corr_diff'],
                          ['Исходные данные','Без линейного тренда','Корреляция первых разностей']):
        m=reg[k];img=ax.imshow(m,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_title(title,fontsize=10);ax.set_xticks(range(10),[f'R{i}' for i in range(1,11)],rotation=45)
        ax.set_yticks(range(10),[f'R{i}' for i in range(1,11)])
        for i in range(10):
            for j in range(10):
                ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=6.4,color='white' if abs(m[i,j])>.55 else '#17212e')
    fig.colorbar(img,ax=axes,label='Коэффициент r',shrink=.85,pad=.02)
    fig.suptitle(f'{name}: корреляции 10 зон по координате X',fontsize=15,fontweight='bold')
    save(fig,p)

def event_profiles(d,drv,events,name,p):
    x=drv['x'];y=drv['y'];resid=drv['residual'];ts=d['ts'];base=drv['baseline']
    fig,ax=plt.subplots(2,1,figsize=(12,6.3),constrained_layout=True,sharex=True)
    mins=minutes_since_start(ts)/60
    for e in events[:4]:
        idx=np.argmin(abs(x-e['peak_x']))
        ax[0].plot(mins,y[:,idx],lw=1.0,label=f'X={x[idx]:.1f}')
        ax[1].plot(mins,resid[:,idx],lw=1.0,label=f'X={x[idx]:.1f}')
    ax[0].set(ylabel='Y (единица не подтверждена)',title=f'{name}: Y в центрах 4 заметных событий')
    ax[1].set(xlabel='Часы с начала записи',ylabel='Отклонение Y',title='Отклонение от локального и общего фона')
    ax[1].axhline(3,ls='--',c='black',alpha=.35,lw=1,label='Минимальный порог +3')
    for a in ax: a.grid(alpha=.22);a.legend(fontsize=8,ncol=3)
    save(fig,p)

def quality(d,name,p):
    ts=d['ts'];x=d['x'];y=d['y'];mins=minutes_since_start(ts)/60
    dt=np.diff(mins)*60
    valid=np.sum(np.isfinite(y),axis=1)
    fig,axes=plt.subplots(2,1,figsize=(12,5.6),constrained_layout=True,sharex=True)
    axes[0].plot(mins[1:],dt,linewidth=.6);axes[0].set(ylabel='Секунды',title=f'{name}: интервал между измерениями');axes[0].grid(alpha=.18)
    axes[1].plot(mins,valid,linewidth=.75);axes[1].set(ylabel='Корректных позиций X',xlabel='Часы с начала записи',title='Число корректных координат в измерении');axes[1].grid(alpha=.18)
    save(fig,p)

def all_plots(summaries,data,deriv,corrs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ('4.csv','5.csv'):
        suffix=name[0]
        overview(data[name],deriv[name],name,out/f'overview_{suffix}.png')
        correlation_panels(corrs[name],name,out/f'correlation_{suffix}.png')
        event_profiles(data[name],deriv[name],summaries[name]['largest_events'],name,out/f'events_{suffix}.png')
        quality(data[name],name,out/f'quality_{suffix}.png')

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent));a.add_argument('--out-dir',default=str(Path(__file__).resolve().parent));args=a.parse_args()
    summaries,data,deriv,corrs,_ = main(args.data_dir,args.out_dir)
    all_plots(summaries,data,deriv,corrs,Path(args.out_dir)/'figures')
