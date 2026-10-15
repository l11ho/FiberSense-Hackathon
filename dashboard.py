"""Интерактивная панель анализа данных распределённых волоконно-оптических датчиков.
Запуск: pip install -r requirements.txt && streamlit run dashboard.py
CSV можно загрузить через интерфейс либо разместить рядом с проектом.
"""
from pathlib import Path
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from analysis import load_csv, summarize_data, calculate_anomalies, region_stats

st.set_page_config(page_title='FiberSense | DTS Data Explorer',layout='wide',page_icon='🌡️')
st.title('FiberSense · Distributed Fiber Sensing')
st.caption('Структура данных · аномалии · корреляции | Единицы X и Y требуют подтверждения')

@st.cache_resource(show_spinner='Чтение и анализ CSV...')
def load_cached(path):
    d=load_csv(path)
    return d,summarize_data(d),region_stats(d)

with st.sidebar:
    st.header('Исходные данные')
    uploaded=st.file_uploader('Загрузите CSV с полем points вида [X, Y]',type=['csv'])
    if uploaded is not None:
        tmp=Path(tempfile.gettempdir())/'fibersense_demo';tmp.mkdir(exist_ok=True)
        target=tmp/(Path(uploaded.name).name)
        # user-uploaded file can be very large; do not convert bytes to data URL
        if not target.exists() or target.stat().st_size!=uploaded.size:
            target.write_bytes(uploaded.getbuffer())
        path=target
    else:
        candidates=[]
        for directory in [Path(__file__).resolve().parent,Path(__file__).resolve().parent.parent,Path.cwd()]:
            candidates += [p for p in [directory/'4.csv',directory/'5.csv'] if p.exists()]
        candidates=list(dict.fromkeys(candidates))
        if not candidates:
            st.info('Для начала загрузите CSV-файл.')
            st.stop()
        selected=st.selectbox('Выберите локальный файл',candidates,format_func=lambda p:p.name)
        path=selected
    st.divider()
    st.header('Настройка детектора')
    min_delta=st.slider('Минимальное отклонение Y',1.0,8.0,3.0,0.5)
    robust_z=st.slider('Коэффициент порога × MAD',3.0,8.0,5.0,0.5)
    st.caption('Статистические пороги, не нормативы безопасной эксплуатации кабеля.')

d,s,c=load_cached(str(path))
with st.spinner('Поиск аномальных областей...'):
    events,drv=calculate_anomalies(d,path.name,min_delta=min_delta,robust_z=robust_z)

cols=st.columns(5)
for col,(lab,val) in zip(cols,[('Измерений',s['records']),('Позиции X',s['valid_positions']),('Медиана Y',f"{s['temp_quantiles']['p50']:.2f}"),('Максимум Y',f"{s['temp_quantiles']['max']:.2f}"),('Группы событий',len(events))]):
    col.metric(lab,val)

T1,T2,T3,T4=st.tabs(['🌡️ Тепловая карта','📈 Координата и время','⚠️ Аномальные события','🔗 Корреляции'])

@st.cache_data
# Прореживание только для визуализации, а не для детектора
def sample_plot(x,y,ts,max_t=400,max_x=600):
    ir=np.unique(np.linspace(0,len(ts)-1,min(max_t,len(ts)),dtype=int))
    ic=np.unique(np.linspace(0,len(x)-1,min(max_x,len(x)),dtype=int))
    return x[ic],y[np.ix_(ir,ic)], [ts[i].strftime('%H:%M:%S') for i in ir]

with T1:
    left,right=st.columns(2)
    with left:
        xz,yz,tz=sample_plot(drv['x'],drv['y'],d['ts'])
        fig=go.Figure(go.Heatmap(x=xz,y=tz,z=yz,colorscale='Inferno',colorbar=dict(title='Y'),hoverongaps=False))
        fig.update_layout(title='Исходное значение Y',height=560,xaxis_title='Координата X',yaxis_title='Время в файле')
        st.plotly_chart(fig,use_container_width=True)
    with right:
        xz,yz,tz=sample_plot(drv['x'],drv['residual'],d['ts'])
        fig=go.Figure(go.Heatmap(x=xz,y=tz,z=yz,colorscale='RdBu_r',zmin=-5,zmax=14,colorbar=dict(title='ΔY'),hoverongaps=False))
        fig.update_layout(title='Отклонение от фонового уровня',height=560,xaxis_title='Координата X',yaxis_title='Время в файле')
        st.plotly_chart(fig,use_container_width=True)
    st.caption('Для ускорения отображения карта прорежена, но алгоритм работает на исходных измерениях.')

with T2:
    default_x=float(events[0]['peak_x']) if events else float(drv['x'][len(drv['x'])//2])
    position=st.number_input('Координата X (будет выбрана ближайшая позиция)',float(drv['x'][0]),float(drv['x'][-1]),float(default_x))
    j=int(np.argmin(np.abs(drv['x']-position)))
    st.caption(f'Выбранная позиция: X={drv["x"][j]:.3f}; фоновый уровень (медиана)={drv["baseline"][j]:.2f}')
    fig=go.Figure()
    fig.add_scatter(x=d['ts'],y=drv['y'][:,j],mode='lines',name='Y в точке X')
    fig.add_hline(y=float(drv['baseline'][j]),line_dash='dash',annotation_text='Историческая медиана')
    fig.update_layout(height=440,title='Динамика значения в выбранной точке',xaxis_title='Время',yaxis_title='Y')
    st.plotly_chart(fig,use_container_width=True)
    ix=st.slider('Выберите измерение для просмотра профиля',min_value=0,max_value=len(d['ts'])-1,value=len(d['ts'])//2)
    fig2=go.Figure()
    fig2.add_scatter(x=drv['x'],y=drv['y'][ix],mode='lines',name=f'Profile {ix}')
    fig2.add_scatter(x=drv['x'],y=drv['baseline'],mode='lines',name='Медианный фон',line=dict(dash='dash'))
    fig2.update_layout(height=400,title=f'Профиль в момент {d["ts"][ix]}',xaxis_title='X',yaxis_title='Y')
    st.plotly_chart(fig2,use_container_width=True)

with T3:
    st.markdown('Связные кластеры по координате и времени после подавления шума. Это **статистические предупреждения**, а не подтверждённые повреждения кабеля.')
    view=[{k:e[k] for k in ['type','first_seen','last_seen','x_start','x_end','peak_x','peak_y','peak_delta_y','frames','affected_cells','score']} for e in events]
    # Русские заголовки только для отображения; машинный CSV сохранён совместимым.
    labels={'type':'Тип','first_seen':'Начало','last_seen':'Конец','x_start':'X начало','x_end':'X конец','peak_x':'Пиковая позиция X','peak_y':'Максимум Y','peak_delta_y':'Пиковое отклонение Y','frames':'Число замеров','affected_cells':'Ячеек','score':'Баллы'}
    type_ru={'повышение':'Повышение Y','понижение':'Понижение Y'}
    view_ru=[{labels.get(k,k):(type_ru.get(v,v) if k=='type' else v) for k,v in row.items()} for row in view]
    st.dataframe(view_ru,use_container_width=True,hide_index=True,height=500)
    import csv,io
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(view_ru[0]) if view_ru else ['Тип'])
    writer.writeheader();writer.writerows(view_ru)
    st.download_button('Скачать CSV событий',data=buffer.getvalue().encode('utf-8-sig'),file_name=f'events_{path.stem}.csv',mime='text/csv')

with T4:
    mode=st.radio('Метод корреляции',options=['corr_raw','corr_detr','corr_diff','corr_spearman'],horizontal=True,
                  format_func=lambda v:{'corr_raw':'Пирсон: исходные ряды','corr_detr':'Пирсон: без тренда','corr_diff':'Пирсон: первые разности','corr_spearman':'Спирмен'}[v])
    R=c[mode]
    fig=go.Figure(go.Heatmap(z=R,x=c['regions'],y=c['regions'],zmin=-1,zmax=1,colorscale='RdBu_r',text=np.round(R,2),texttemplate='%{text}',hoverongaps=False))
    fig.update_layout(height=700,title='Матрица корреляций: 10 зон',xaxis_tickangle=-55)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Зоны R1–R10 последовательно разделяют корректный диапазон X. Корреляция Пирсона не доказывает причинно-следственную связь.')

st.divider()
st.caption(f"File: {path.name} · {s['start']} → {s['end']} · Часовой пояс: {s['timezone']} · единицы X/Y не подтверждены")
