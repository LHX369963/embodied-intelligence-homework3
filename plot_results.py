"""从实际结果绘制报告图；PDF 中间图转换为 Typst 使用的 SVG。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
import json
import subprocess

# --- Style Config ---
FONT_PATH = Path(__file__).resolve().parent / 'assets' / 'Times_New_Roman+SimSun.ttf'
font_main = font_manager.FontProperties(fname=FONT_PATH, size=12)
font_legend = font_manager.FontProperties(fname=FONT_PATH, size=10)
plt.rcParams.update({
    'lines.linewidth': 1, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'figure.figsize': (5, 3),
})
COLORS = ['#3e7ad9', '#8cb369', '#f4e285', '#f4a259', '#5b8e7d', '#a3b18a']

ROOT = Path(__file__).resolve().parent
pretraining = json.loads((ROOT / 'pretraining.json').read_text())
finetuning = json.loads((ROOT / 'finetuning.json').read_text())

fig, ax = plt.subplots()
history = pretraining['history']
ax.plot([r['step'] for r in history], [r['train_loss'] for r in history],
        color=COLORS[0], label='训练：最近250步均值')
ax.plot([0] + [r['step'] for r in history],
        [pretraining['initial_validation']['loss']] + [r['validation_loss'] for r in history],
        color=COLORS[1], marker='o', markersize=3, label='验证：完整评估块')
ax.set_xlabel('参数更新步数', fontproperties=font_main)
ax.set_ylabel('词元交叉熵', fontproperties=font_main)
ax.legend(prop=font_legend, loc='upper right', frameon=False)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_main)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(ROOT / 'assets/pretraining.pdf', bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

fig, ax = plt.subplots()
for name, label, color in [('frozen', '冻结主干', COLORS[3]),
                           ('scratch', '随机初始化', COLORS[0]),
                           ('full', '全参数微调', COLORS[1])]:
    rows = finetuning['variants'][name]['history']
    ax.plot([r['epoch'] for r in rows], [r['loss'] for r in rows],
            marker='o', markersize=3, color=color, label=label)
ax.set_xlabel('监督训练轮次', fontproperties=font_main)
ax.set_ylabel('验证交叉熵', fontproperties=font_main)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.legend(prop=font_legend, loc='lower center', bbox_to_anchor=(0.5, 1.01),
          ncol=3, columnspacing=0.8, handlelength=1.4, frameon=False)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_main)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(ROOT / 'assets/finetuning.pdf', bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

fig, ax = plt.subplots()
matrix = finetuning['variants']['full']['test']['confusion_matrix']
ax.imshow(matrix, cmap='Blues', vmin=0, vmax=533)
ax.set_xticks([0, 1], ['负面', '正面'], fontproperties=font_main)
ax.set_yticks([0, 1], ['负面', '正面'], fontproperties=font_main)
ax.set_xlabel('预测类别', fontproperties=font_main)
ax.set_ylabel('真实类别', fontproperties=font_main)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                fontproperties=font_main, color='white' if matrix[i][j] > 266 else '#222222')
plt.tight_layout()
plt.savefig(ROOT / 'assets/confusion.pdf', bbox_inches='tight', pad_inches=0.03)
plt.close(fig)

# Typst 以 SVG 嵌入矢量图，转换后只保留这一份图件。
for name in ('pretraining', 'finetuning', 'confusion'):
    source = ROOT / 'assets' / f'{name}.pdf'
    subprocess.run(['pdftocairo', '-svg', str(source), str(source.with_suffix('.svg'))], check=True)
    source.unlink()
