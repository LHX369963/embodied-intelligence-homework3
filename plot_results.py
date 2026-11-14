"""从实测 JSON 结果绘制实验报告矢量图，经 PDF 转换为 Typst 嵌入的 SVG。"""
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
FONT_PATH = ROOT / 'assets' / 'Times_New_Roman+SimSun.ttf'
font_main = font_manager.FontProperties(fname=FONT_PATH, size=10.0)
font_small = font_manager.FontProperties(fname=FONT_PATH, size=8.8)
font_title = font_manager.FontProperties(fname=FONT_PATH, size=10.8)
font_legend = font_manager.FontProperties(fname=FONT_PATH, size=9.2)

plt.rcParams.update({
    'lines.linewidth': 1.3,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})
COLORS = {
    'train': '#3e7ad9',
    'val': '#2b9348',
    'frozen': '#e07a5f',
    'scratch': '#3d5a80',
    'full': '#2b9348',
    'm4': '#8d99ae',
    'm515': '#2b6cb0',
    'm515_long': '#1b4332',
}


def style_axis(ax, small_x=False):
    x_prop = font_small if small_x else font_main
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(x_prop)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(font_main)
    ax.spines[['top', 'right']].set_visible(False)


def export_svg(fig, stem: str):
    pdf_path = ROOT / 'assets' / f'{stem}.pdf'
    svg_path = ROOT / 'assets' / f'{stem}.svg'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    subprocess.run(['pdftocairo', '-svg', str(pdf_path), str(svg_path)], check=True)
    pdf_path.unlink()


def plot_benchmark(bench: dict):
    models = bench.get('models', {})
    if '4m' not in models or '515m' not in models:
        return
    m4 = models['4m']
    m5 = models['515m']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.2))

    # (a) WikiText 跨词表 BPB 对比
    lm_labels = ['4.24M\n(窗口128)', '515M\n(窗口128)', '515M\n(窗口512)']
    bpb_vals = [
        m4['wikitext_test']['ctx_128']['bits_per_byte'],
        m5['wikitext_test']['ctx_128']['bits_per_byte'],
        m5['wikitext_test']['ctx_512']['bits_per_byte'],
    ]
    bars = ax1.bar(lm_labels, bpb_vals, width=0.52,
                   color=[COLORS['m4'], COLORS['m515'], COLORS['m515_long']])
    ax1.set_ylabel('每字节交叉熵 (BPB, 越低越优)', fontproperties=font_main)
    ax1.set_title('(a) 同测试集语言建模压缩率', fontproperties=font_title, pad=8)
    ax1.set_ylim(0, max(bpb_vals) * 1.22)
    for b, v in zip(bars, bpb_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.3f}',
                 ha='center', va='bottom', fontproperties=font_small)
    style_axis(ax1, small_x=True)

    # (b) BLiMP 语法最小对零样本准确率
    task_keys = list(m4['blimp_zero_shot']['tasks'].keys())
    short_names = ['主谓一致', '限定词', '代词指代', '过去分词', '疑问空位', '总平均']
    v4_blimp = [m4['blimp_zero_shot']['tasks'][k]['accuracy'] * 100 for k in task_keys] + [
        m4['blimp_zero_shot']['macro_accuracy'] * 100
    ]
    v5_blimp = [m5['blimp_zero_shot']['tasks'][k]['accuracy'] * 100 for k in task_keys] + [
        m5['blimp_zero_shot']['macro_accuracy'] * 100
    ]
    x = np.arange(len(short_names))
    w = 0.36
    ax2.bar(x - w / 2, v4_blimp, width=w, color=COLORS['m4'], label='4.24M 原型')
    ax2.bar(x + w / 2, v5_blimp, width=w, color=COLORS['m515'], label='515M 主力')
    ax2.axhline(50.0, color='#999999', linestyle='--', linewidth=0.8)
    ax2.set_xticks(x, short_names, rotation=22)
    ax2.set_ylabel('零样本语法准确率 / %', fontproperties=font_main)
    ax2.set_title('(b) BLiMP 语言学最小对评测', fontproperties=font_title, pad=8)
    ax2.set_ylim(35, 100)
    ax2.legend(prop=font_legend, loc='upper left', frameon=False)
    style_axis(ax2, small_x=True)

    # (c) Rotten Tomatoes 零样本/少样本与三组监督全谱系对比
    stage_names = ['零样本', '4-Shot', '冻结主干', '随机初值', '全参微调']
    v4_sent = [
        m4['sentiment_icl']['zero_shot']['accuracy'] * 100,
        m4['sentiment_icl']['four_shot']['accuracy'] * 100,
        m4['sentiment_supervised']['frozen']['accuracy'] * 100,
        m4['sentiment_supervised']['scratch']['accuracy'] * 100,
        m4['sentiment_supervised']['full']['accuracy'] * 100,
    ]
    v5_sent = [
        m5['sentiment_icl']['zero_shot']['accuracy'] * 100,
        m5['sentiment_icl']['four_shot']['accuracy'] * 100,
        m5['sentiment_supervised']['frozen']['accuracy'] * 100,
        m5['sentiment_supervised']['scratch']['accuracy'] * 100,
        m5['sentiment_supervised']['full']['accuracy'] * 100,
    ]
    xs = np.arange(len(stage_names))
    ax3.bar(xs - w / 2, v4_sent, width=w, color=COLORS['m4'], label='4.24M 原型')
    ax3.bar(xs + w / 2, v5_sent, width=w, color=COLORS['m515'], label='515M 主力')
    ax3.axhline(50.0, color='#999999', linestyle='--', linewidth=0.8)
    ax3.set_xticks(xs, stage_names, rotation=18)
    ax3.set_ylabel('测试集分类准确率 / %', fontproperties=font_main)
    ax3.set_title('(c) 影评情感判别全谱系对比', fontproperties=font_title, pad=8)
    ax3.set_ylim(45, 96)
    style_axis(ax3, small_x=True)

    fig.tight_layout(w_pad=1.4)
    export_svg(fig, 'benchmark')


def main():
    pretraining = json.loads((ROOT / 'pretraining.json').read_text())
    finetuning = json.loads((ROOT / 'finetuning.json').read_text())

    # 图 1：预训练交叉熵与验证困惑度双面板
    history = pretraining['history']
    steps = [r['step'] for r in history]
    train_losses = [r['train_loss'] for r in history]
    val_steps = [0] + steps
    val_losses = [pretraining['initial_validation']['loss']] + [r['validation_loss'] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    ax1.plot(steps, train_losses, color=COLORS['train'], label='训练批次均值')
    ax1.plot(val_steps, val_losses, color=COLORS['val'], marker='o', markersize=3.2, label='验证集完整评估')
    ax1.set_xlabel('参数更新步数', fontproperties=font_main)
    ax1.set_ylabel('词元交叉熵损失', fontproperties=font_main)
    ax1.set_title('(a) 训练与验证交叉熵变化', fontproperties=font_title, pad=6)
    ax1.legend(prop=font_legend, loc='upper right', frameon=False)
    style_axis(ax1)

    ax2.plot(steps, [r['validation_perplexity'] for r in history],
             color=COLORS['val'], marker='s', markersize=3.2)
    ax2.set_xlabel('参数更新步数', fontproperties=font_main)
    ax2.set_ylabel('验证集词元困惑度 (PPL)', fontproperties=font_main)
    ax2.set_title('(b) 预训练期间验证困惑度收敛过程', fontproperties=font_title, pad=6)
    style_axis(ax2)

    fig.tight_layout(w_pad=1.8)
    export_svg(fig, 'pretraining')

    # 图 2：三组监督微调对照的验证损失与验证准确率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    specs = [
        ('frozen', '冻结预训练主干', COLORS['frozen'], '^'),
        ('scratch', '随机初始化全参数', COLORS['scratch'], 's'),
        ('full', '预训练全参数微调', COLORS['full'], 'o'),
    ]
    for key, label, color, marker in specs:
        rows = finetuning['variants'][key]['history']
        epochs = [r['epoch'] for r in rows]
        ax1.plot(epochs, [r['loss'] for r in rows], color=color, marker=marker, markersize=3.8, label=label)
        ax2.plot(epochs, [r['accuracy'] * 100.0 for r in rows], color=color, marker=marker, markersize=3.8, label=label)

    ax1.set_xlabel('监督训练轮次', fontproperties=font_main)
    ax1.set_ylabel('验证集交叉熵损失', fontproperties=font_main)
    ax1.set_title('(a) 逐轮验证交叉熵', fontproperties=font_title, pad=6)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(ax1)

    ax2.set_xlabel('监督训练轮次', fontproperties=font_main)
    ax2.set_ylabel('验证集准确率 / %', fontproperties=font_main)
    ax2.set_title('(b) 逐轮验证准确率', fontproperties=font_title, pad=6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(ax2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, prop=font_legend, loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, columnspacing=1.5)
    fig.tight_layout(w_pad=1.8)
    export_svg(fig, 'finetuning')

    # 图 3：三组对照模型的测试集混淆矩阵并列对比
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    titles = [
        ('frozen', '(a) 冻结预训练主干'),
        ('scratch', '(b) 随机初始化全参数'),
        ('full', '(c) 预训练全参数微调'),
    ]
    for idx, (key, subtitle) in enumerate(titles):
        ax = axes[idx]
        matrix = finetuning['variants'][key]['test']['confusion_matrix']
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=533)
        ax.set_xticks([0, 1], ['负面', '正面'], fontproperties=font_main)
        ax.set_yticks([0, 1], ['负面', '正面'], fontproperties=font_main)
        ax.set_xlabel('预测类别', fontproperties=font_main)
        if idx == 0:
            ax.set_ylabel('真实类别', fontproperties=font_main)
        ax.set_title(subtitle, fontproperties=font_title, pad=6)
        for i in range(2):
            for j in range(2):
                val = matrix[i][j]
                ax.text(
                    j, i, str(val), ha='center', va='center',
                    fontproperties=font_main,
                    color='white' if val > 270 else '#1d2d44',
                )
    fig.tight_layout(w_pad=1.2)
    export_svg(fig, 'confusion')

    bench_path = ROOT / 'benchmark.json'
    if bench_path.exists():
        plot_benchmark(json.loads(bench_path.read_text()))


if __name__ == '__main__':
    main()
