"""Воссоздание всех артефактов за один проход исходных данных."""
from pathlib import Path
import subprocess,sys
from analysis import main
from visuals import all_plots
from make_demo_html import create
root=Path(__file__).resolve().parent
print('Шаг 1 / 4: расчёты...',flush=True)
res=main(root.parent,root)
from localize_events import make
make(root)
print('Шаг 2 / 4: графики на русском...',flush=True)
all_plots(res[0],res[1],res[2],res[3],root/'figures')
print('Шаг 3 / 4: интерактивный HTML...',flush=True)
create(root.parent,root,computed=res)
print('Шаг 4 / 4: отчёт PDF...',flush=True)
import make_report
print('Все файлы готовы.',flush=True)
