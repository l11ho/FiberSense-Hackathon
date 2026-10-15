"""Генерирует автономный HTML-демонстратор без доступа к сети."""
from pathlib import Path
import argparse,html,json, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from analysis import main,minutes_since_start

def make_fig(d,drv,reg,name):
    x=drv['x']; t=d['ts'];y=drv['y'];r=drv['residual']
    tidx=np.unique(np.linspace(0,len(t)-1,min(len(t),280),dtype=int))
    xidx=np.unique(np.linspace(0,len(x)-1,min(len(x),450),dtype=int))
    tm=minutes_since_start(t)/60
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Значения Y: координата и время', 'Отклонение Y от фона', 'Статистика во времени', 'Корреляции между 10 зонами'),
                      vertical_spacing=.16,horizontal_spacing=.14,row_heights=[.57,.43])
    fig.add_trace(go.Heatmap(x=x[xidx],y=tm[tidx],z=np.round(y[np.ix_(tidx,xidx)],2),colorscale='Inferno',
                 colorbar=dict(x=.445,y=.79,len=.41,thickness=11,title='Y'),hoverongaps=False,name='Y'),1,1)
    fig.add_trace(go.Heatmap(x=x[xidx],y=tm[tidx],z=np.round(r[np.ix_(tidx,xidx)],2),colorscale='RdBu_r',zmin=-5,zmax=14,
                 colorbar=dict(x=1.02,y=.79,len=.41,thickness=11,title='ΔY'),hoverongaps=False,name='Отклонение'),1,2)
    for color,data,label in [('#1384a6',np.nanmedian(y,axis=1),'Медиана всей линии'),('#f39c12',np.nanquantile(y,.99,axis=1),'99-й процентиль'),('#d0444d',np.nanmax(y,axis=1),'Максимум измерения')]:
        fig.add_trace(go.Scatter(x=tm,y=np.round(data,3),name=label,mode='lines',line=dict(color=color,width=1.5)),2,1)
    fig.add_trace(go.Heatmap(z=np.round(reg['corr_detr'],3),x=list(range(1,11)),y=list(range(1,11)),
                     zmin=-1,zmax=1,colorscale='RdBu_r',colorbar=dict(x=1.02,y=.19,len=.32,thickness=11,title='r'),name='Корреляции'),2,2)
    fig.update_xaxes(title_text='Координата X (единица не подтверждена)',row=1,col=1)
    fig.update_xaxes(title_text='Координата X (единица не подтверждена)',row=1,col=2)
    fig.update_yaxes(title_text='Часы с начала записи',row=1,col=1)
    fig.update_yaxes(title_text='Часы с начала записи',row=1,col=2)
    fig.update_xaxes(title_text='Часы с начала записи',row=2,col=1)
    fig.update_yaxes(title_text='Y',row=2,col=1)
    fig.update_xaxes(title_text='Зоны R1–R10',dtick=1,row=2,col=2)
    fig.update_yaxes(title_text='Зоны R1–R10',dtick=1,row=2,col=2)
    fig.update_layout(title=f'{name} · пространственно-временной анализ',height=870,plot_bgcolor='white',paper_bgcolor='white',
                      font=dict(family='Arial, sans-serif',size=11,color='#253547'),margin=dict(l=65,r=80,t=96,b=60),
                      legend=dict(orientation='h',y=-.12),hovermode='closest')
    return fig

def create(data_dir,out_dir,computed=None):
    summ,data,drvs,corrs,_=computed if computed is not None else main(data_dir,out_dir)
    figs={name:make_fig(data[name],drvs[name],corrs[name],name) for name in ('4.csv','5.csv')}
    chunks=[]
    for i,name in enumerate(('4.csv','5.csv')):
        figure_html=plot(figs[name],output_type='div',include_plotlyjs=(True if i==0 else False),config={'displaylogo':False,'responsive':True})
        s=summ[name];evt=s['largest_events'][:12]
        meta=''.join(f'<div class="metric"><b>{v}</b><small>{k}</small></div>' for k,v in [
            ('Измерений',s['records']),('Корректных X',s['valid_positions']),
            ('Медиана Y',round(s['temp_quantiles']['p50'],2)),
            ('Максимум Y',round(s['temp_quantiles']['max'],2)),
            ('Кластеры повышения Y',s['hot_event_count'])])
        rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [
                 e['first_seen'],e['last_seen'],e['x_start'],e['x_end'],e['peak_x'],e['peak_delta_y'],e['frames']])+'</tr>' for e in evt)
        table='<h3>12 наиболее заметных кластеров по рейтингу алгоритма</h3><div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in ['Начало','Конец','X начало','X конец','Пик X','Отклонение ΔY','Измерений'])+'</tr></thead><tbody>'+rows+'</tbody></table></div>'
        note=f'<p class="note">Часовой пояс: {html.escape("не указан" if s["timezone"] == "not specified" else s["timezone"])}. Период: {html.escape(s["start"])} — {html.escape(s["end"])}. Статистические сигналы, физические дефекты не подтверждены.</p>'
        chunks.append('<section class="tab" id="tab'+str(i)+'"'+(' style="display:none"' if i else '')+'><div class="metrics">'+meta+'</div>'+note+figure_html+table+'</section>')
    whole='''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FiberSense · Hackathon Dashboard</title><style>
    *{box-sizing:border-box}body{font-family:Arial,'Segoe UI',sans-serif;background:#f4f7fa;color:#253547;margin:0}header{background:#102741;color:white;padding:24px max(22px,calc((100vw - 1370px)/2)) 20px}h1{margin:0 0 9px;font-size:25px}header p{opacity:.86;max-width:900px;line-height:1.5;margin:0}main{max-width:1400px;margin:16px auto;padding:0 16px 40px}.tabs{display:flex;gap:12px;margin-bottom:16px}.tabs button{padding:11px 24px;cursor:pointer;border:1px solid #bfd0d9;background:white;border-radius:8px;font-size:15px;font-weight:700;color:#234d60}.tabs button.active{color:white;background:#176980;border-color:#176980}.tab{background:white;border:1px solid #e1e8ee;border-radius:13px;padding:15px 20px 26px}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:8px 0 12px}.metric{background:#f0f6f8;padding:13px;border-radius:9px;min-height:83px}.metric b{display:block;font-size:24px;color:#102741}.metric small{display:block;margin-top:5px;color:#526c7e;font-size:12px}.note{font-size:12px;color:#637383;margin:8px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th{background:#102741;color:white;text-align:left;padding:12px 10px}td{padding:9px 10px;border-bottom:1px solid #d8e5ec;font-size:13px}tr:nth-child(even){background:#f5f9fc}footer{text-align:center;color:#637383;font-size:12px;margin:30px 0}@media(max-width:720px){.metrics{grid-template-columns:repeat(2,1fr)}.tab{padding:8px}.metric b{font-size:18px}}
    </style></head><body><header><h1>FiberSense — Анализ данных распределённых волоконно-оптических датчиков</h1><p>Автономная демонстрация на двух CSV из задания хакатона. Графики интерактивны: масштабирование, подсказки и сохранение PNG. Для отображения выполнено прореживание. Единицы X/Y не подтверждены. Результаты не применимы для принятия решений по безопасности эксплуатации.</p></header><main><div class="tabs"><button class="active" onclick="tab(0,this)">4.csv</button><button onclick="tab(1,this)">5.csv</button></div>
    '''+''.join(chunks)+'''<footer>Без подключения к интернету · Сигналы — статистические гипотезы, а не подтверждённые повреждения кабеля.</footer></main><script>function tab(which,btn){document.querySelectorAll('.tab').forEach((node,i)=>node.style.display=i===which?'block':'none');document.querySelectorAll('.tabs button').forEach(node=>node.classList.remove('active'));btn.classList.add('active');setTimeout(()=>document.querySelectorAll('#tab'+which+' .plotly-graph-div').forEach(p=>Plotly.Plots.resize(p)),100)}</script></body></html>'''
    path=Path(out_dir)/'dashboard_demo.html';path.write_text(whole,encoding='utf-8');print('HTML CREATED',path,path.stat().st_size)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent));ap.add_argument('--out-dir',default=str(Path(__file__).resolve().parent));a=ap.parse_args();create(a.data_dir,a.out_dir)
