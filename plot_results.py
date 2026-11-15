"""从实测 JSON 结果绘制三阶段（4.24M GPT / 515M GPT-2 / 516M Qwen3）实验报告矢量图，经 PDF 转为 SVG。"""
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
font_small = font_manager.FontProperties(fname=FONT_PATH, size=8.6)
font_title = font_manager.FontProperties(fname=FONT_PATH, size=10.6)
font_legend = font_manager.FontProperties(fname=FONT_PATH, size=8.9)

plt.rcParams.update({
    'lines.linewidth': 1.35,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})
COLORS = {
    'gpt2_train': '#94a3b8',
    'gpt2_val': '#2b6cb0',
    'qwen_train': '#74c69d',
    'qwen_val': '#1b7f3b',
    'frozen': '#e07a5f',
    'scratch': '#3d5a80',
    'full': '#1b7f3b',
    'm4': '#94a3b8',
    'm515': '#3b82f6',
    'qwen3': '#1b7f3b',
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
    if not all(k in models for k in ('4m', '515m', 'qwen3')):
        return
    m4 = models['4m']
    m5 = models['515m']
    mq = models['qwen3']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.25))

    # (a) WikiText 跨词表与同参数架构 BPB 对比
    lm_labels = ['4.24M\nGPT', '515M\nGPT-2', '516M\nQwen3']
    bpb_vals = [
        m4['wikitext_test']['ctx_128']['bits_per_byte'],
        m5['wikitext_test']['ctx_512']['bits_per_byte'],
        mq['wikitext_test']['ctx_512']['bits_per_byte'],
    ]
    bars = ax1.bar(lm_labels, bpb_vals, width=0.50,
                   color=[COLORS['m4'], COLORS['m515'], COLORS['qwen3']])
    ax1.axhline(1.0, color='#999999', linestyle='--', linewidth=0.75)
    ax1.set_ylabel('每字节交叉熵 (BPB, 越低越优)', fontproperties=font_main)
    ax1.set_title('(a) 同测试集语言建模压缩率', fontproperties=font_title, pad=8)
    ax1.set_ylim(0, max(bpb_vals) * 1.22)
    for b, v in zip(bars, bpb_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:.4f}',
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
    vq_blimp = [mq['blimp_zero_shot']['tasks'][k]['accuracy'] * 100 for k in task_keys] + [
        mq['blimp_zero_shot']['macro_accuracy'] * 100
    ]
    x = np.arange(len(short_names))
    w = 0.26
    ax2.bar(x - w, v4_blimp, width=w, color=COLORS['m4'], label='4.24M GPT')
    ax2.bar(x, v5_blimp, width=w, color=COLORS['m515'], label='515M GPT-2')
    ax2.bar(x + w, vq_blimp, width=w, color=COLORS['qwen3'], label='516M Qwen3')
    ax2.axhline(50.0, color='#999999', linestyle='--', linewidth=0.75)
    ax2.set_xticks(x, short_names, rotation=22)
    ax2.set_ylabel('零样本语法准确率 / %', fontproperties=font_main)
    ax2.set_title('(b) BLiMP 语言学最小对评测', fontproperties=font_title, pad=8)
    ax2.set_ylim(35, 102)
    ax2.legend(prop=font_legend, loc='lower right', frameon=True, framealpha=0.88, edgecolor='none')
    style_axis(ax2, small_x=True)

    # (c) Rotten Tomatoes 零样本与三组监督对比
    stage_names = ['零样本', '冻结主干', '随机初值', '全参微调']
    v4_sent = [
        m4['sentiment_icl']['zero_shot']['accuracy'] * 100,
        m4['sentiment_supervised']['frozen']['accuracy'] * 100,
        m4['sentiment_supervised']['scratch']['accuracy'] * 100,
        m4['sentiment_supervised']['full']['accuracy'] * 100,
    ]
    v5_sent = [
        m5['sentiment_icl']['zero_shot']['accuracy'] * 100,
        m5['sentiment_supervised']['frozen']['accuracy'] * 100,
        m5['sentiment_supervised']['scratch']['accuracy'] * 100,
        m5['sentiment_supervised']['full']['accuracy'] * 100,
    ]
    vq_sent = [
        mq['sentiment_icl']['zero_shot']['accuracy'] * 100,
        mq['sentiment_supervised']['frozen']['accuracy'] * 100,
        mq['sentiment_supervised']['scratch']['accuracy'] * 100,
        mq['sentiment_supervised']['full']['accuracy'] * 100,
    ]
    xs = np.arange(len(stage_names))
    ax3.bar(xs - w, v4_sent, width=w, color=COLORS['m4'], label='4.24M GPT')
    ax3.bar(xs, v5_sent, width=w, color=COLORS['m515'], label='515M GPT-2')
    ax3.bar(xs + w, vq_sent, width=w, color=COLORS['qwen3'], label='516M Qwen3')
    ax3.axhline(50.0, color='#999999', linestyle='--', linewidth=0.75)
    ax3.set_xticks(xs, stage_names, rotation=14)
    ax3.set_ylabel('测试集分类准确率 / %', fontproperties=font_main)
    ax3.set_title('(c) 影评情感判别全谱系对比', fontproperties=font_title, pad=8)
    ax3.set_ylim(42, 92)
    style_axis(ax3, small_x=True)

    fig.tight_layout(w_pad=1.35)
    export_svg(fig, 'benchmark')


def main():
    pretraining = json.loads((ROOT / 'pretraining.json').read_text())
    finetuning = json.loads((ROOT / 'finetuning.json').read_text())
    bench = json.loads((ROOT / 'benchmark.json').read_text())

    # 图 1：515M GPT-2 (Cosine) 与 516M Qwen3 (WSD) 等词元预训练收敛同台对比
    q_hist = pretraining['history']
    q_steps = [r['step'] for r in q_hist]
    g_summary = bench['models']['515m'].get('pretrain_summary', {})
    g_hist = g_summary.get('history', [])
    g_steps = [r['step'] for r in g_hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.05))
    if g_hist:
        ax1.plot(g_steps, [r['train_loss'] for r in g_hist], color=COLORS['gpt2_train'],
                 linestyle='--', linewidth=1.1, label='515M GPT-2 训练损失')
        ax1.plot(g_steps, [r['validation_loss'] for r in g_hist], color=COLORS['gpt2_val'],
                 marker='^', markersize=3.0, label='515M GPT-2 验证损失')
    ax1.plot(q_steps, [r['train_loss'] for r in q_hist], color=COLORS['qwen_train'],
             linestyle='--', linewidth=1.1, label='516M Qwen3 训练损失')
    ax1.plot(q_steps, [r['validation_loss'] for r in q_hist], color=COLORS['qwen_val'],
             marker='o', markersize=3.2, label='516M Qwen3 验证损失')
    ax1.axvline(4000, color='#888888', linestyle=':', linewidth=0.85)
    ax1.set_xlabel('参数更新步数', fontproperties=font_main)
    ax1.set_ylabel('词元交叉熵损失', fontproperties=font_main)
    ax1.set_title('(a) 等词元预训练交叉熵对比', fontproperties=font_title, pad=6)
    ax1.legend(prop=font_legend, loc='upper right', frameon=False)
    style_axis(ax1)

    if g_hist:
        ax2.plot(g_steps, [r['validation_perplexity'] for r in g_hist],
                 color=COLORS['gpt2_val'], marker='^', markersize=3.0, label='515M GPT-2 (Cosine)')
    ax2.plot(q_steps, [r['validation_perplexity'] for r in q_hist],
             color=COLORS['qwen_val'], marker='s', markersize=3.2, label='516M Qwen3 (WSD)')
    ax2.axvline(4000, color='#888888', linestyle=':', linewidth=0.85)
    ax2.text(4040, 260, 'WSD 退火起点', fontproperties=font_small, color='#555555')
    ax2.set_xlabel('参数更新步数', fontproperties=font_main)
    ax2.set_ylabel('验证集词元困惑度 (PPL)', fontproperties=font_main)
    ax2.set_title('(b) 验证集困惑度收敛轨迹对比', fontproperties=font_title, pad=6)
    ax2.legend(prop=font_legend, loc='upper right', frameon=False)
    style_axis(ax2)

    fig.tight_layout(w_pad=1.6)
    export_svg(fig, 'pretraining')

    # 图 2：516M Qwen3 三组监督微调对照的验证损失与验证准确率（含第 0 轮初始状态）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.05))
    specs = [
        ('frozen', '冻结预训练主干', COLORS['frozen'], '^'),
        ('scratch', '随机初始化全参数', COLORS['scratch'], 's'),
        ('full', '预训练全参数微调', COLORS['full'], 'o'),
    ]
    for key, label, color, marker in specs:
        var = finetuning['variants'][key]
        init_v = var['initial_validation']
        rows = var['history']
        epochs = [0] + [r['epoch'] for r in rows]
        losses = [init_v['loss']] + [r['loss'] for r in rows]
        accs = [init_v['accuracy'] * 100.0] + [r['accuracy'] * 100.0 for r in rows]
        ax1.plot(epochs, losses, color=color, marker=marker, markersize=3.6, label=label)
        ax2.plot(epochs, accs, color=color, marker=marker, markersize=3.6, label=label)

    ax1.set_xlabel('监督训练轮次（0 表示训练前初值）', fontproperties=font_main)
    ax1.set_ylabel('验证集交叉熵损失', fontproperties=font_main)
    ax1.set_title('(a) 逐轮验证交叉熵', fontproperties=font_title, pad=6)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(ax1)

    ax2.set_xlabel('监督训练轮次（0 表示训练前初值）', fontproperties=font_main)
    ax2.set_ylabel('验证集准确率 / %', fontproperties=font_main)
    ax2.set_title('(b) 逐轮验证准确率', fontproperties=font_title, pad=6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    style_axis(ax2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, prop=font_legend, loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, columnspacing=1.5)
    fig.tight_layout(w_pad=1.7)
    export_svg(fig, 'finetuning')

    # 图 3：516M Qwen3 三组对照模型的测试集混淆矩阵并列对比
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

    plot_benchmark(bench)


if __name__ == '__main__':
    main()
