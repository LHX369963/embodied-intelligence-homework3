# 作业三：从零构建大语言模型与下游情感分类微调

- 代码仓库：<https://github.com/LHX369963/embodied-intelligence-homework3>
- 模型权重（GitHub Release）：<https://github.com/LHX369963/embodied-intelligence-homework3/releases/tag/submission>

## 一、快速运行与推理演示

`inference.py` 支持自动下载模型权重与国内镜像源切换。为便于直接展示最优模型效果并节约下载与测试时间，Release 页面仅提供 516M Qwen3 架构的模型权重（`bfloat16` 格式，单文件约 985 MiB）。克隆仓库并安装依赖后，首次执行命令会自动将所需权重下载至 `checkpoints/` 目录，支持 CPU、CUDA、ROCm 与 Apple MPS 运行。

```bash
git clone https://github.com/LHX369963/embodied-intelligence-homework3.git
cd embodied-intelligence-homework3
python -m pip install -r requirements.txt

# 1. 一键演示（自动下载 pretrained.pt 与 full.pt，展示英文文本续写与影评情感分类）
python inference.py demo

# 2. 自定义提示词续写（可通过 --device auto 自动选择硬件加速）
python inference.py generate "The history of artificial intelligence" --new-tokens 80

# 3. 自定义影评情感分类（可通过 --variant 切换 full / frozen / scratch）
python inference.py classify "A deeply moving, visually stunning film."
```

若需要在离线环境中测试，可提前执行 `python inference.py download --all` 下载全部四份权重，或从 Release 页面下载 `.pt` 文件放入 `checkpoints/` 目录。网络访问 GitHub 较慢时可指定镜像前缀：

```bash
GITHUB_PROXY=https://ghfast.top python inference.py demo
```

## 二、模型配置与实验结果

项目基于 PyTorch 基础模块从头实现三套递进的语言模型，对比模型规模扩展（4.24M 到 515M）与同参数量架构升级（515M GPT-2 到 516M Qwen3）的效果。

### 1. 三套模型结构配置

| 配置项 | 小型 GPT | 标准 GPT-2 | 改进型 Qwen3 |
| --- | --- | --- | --- |
| 词表容量 | 4,096 | 32,768 | 32,768 |
| 上下文窗口 / 层数 | 128 / 4 层 | 512 / 24 层 | 512 / 28 层 |
| 隐藏维 / 注意力头 | 256 / 4 头 MHA | 1,280 / 20 头 MHA | 1,024 / 16Q:8KV GQA |
| 位置编码 | 可学习绝对位置 | 可学习绝对位置 | RoPE 旋转位置编码 |
| 注意力改进 | 无 | 无 | QK-Norm + Sigmoid 门控 |
| 归一化与前馈层 | LayerNorm + GELU | LayerNorm + GELU | RMSNorm + SwiGLU |
| 学习率调度 | Cosine | Cosine（3e-4） | WSD（6e-4） |
| 总参数量 | 4.24M | 514.86M | 516.23M |

### 2. 516M Qwen3 影评分类微调对比（Rotten Tomatoes 测试集）

| 训练方式 | 可训练参数量 | 测试损失 | 测试准确率 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| 随机初始化全参（`scratch`） | 516.23M | 0.5112 | 75.42% | 0.8336 |
| 冻结预训练主干（`frozen`） | 2,050 | 0.4514 | 79.92% | 0.8703 |
| 预训练全参微调（`full`） | 516.23M | 0.3960 | **82.08%** | **0.9021** |

### 3. 三套模型横向评测对比

| 评测指标 | 4.24M GPT | 515M GPT-2 | 516M Qwen3 |
| --- | ---: | ---: | ---: |
| WikiText 词级困惑度（Word-PPL） | 465.32 | 47.47 | **35.84** |
| WikiText 每字节比特数（BPB） | 1.6601 | 1.0432 | **0.9673** |
| BLiMP 语法最小对准确率 | 65.54% | 84.20% | **86.30%** |
| 影评零样本分类准确率 | 49.34% | 58.82% | **72.70%** |
| 影评冻结主干准确率（`frozen`） | 60.60% | 74.77% | **79.92%** |
| 影评全参微调准确率（`full`） | 71.67% | 79.74% | **82.08%** |

## 三、完整训练复现命令

```bash
python prepare.py                             # 下载公开语料并训练 32,768 词表字节级 BPE 分词器
python experiment.py pretrain --steps 5000    # 运行 516M Qwen3 模型 WSD 混合精度预训练
python experiment.py finetune --epochs 10     # 运行三种监督微调方式并评估测试集
python benchmark.py --scale qwen3             # 运行多任务横向基准评测
python plot_results.py                        # 生成 assets/ 目录下的实验结果矢量图
```

## 四、仓库文件说明

- `model.py`：GPT-2 与 Qwen3 语言模型、增量 KV 缓存生成及情感分类头实现。
- `prepare.py`：下载 WikiText-103、IMDb、Rotten Tomatoes 语料并训练字节级 BPE 分词器。
- `experiment.py`：预训练与三种监督微调方式的训练及评测流程。
- `inference.py`：权重自动下载、文本续写与情感分类命令行工具。
- `benchmark.py`：三套模型在 WikiText、BLiMP 与 Rotten Tomatoes 上的横向评测脚本。
- `plot_results.py`：训练曲线、混淆矩阵与对比柱状图绘制脚本。
- `tokenizer.json`、`data_info.json`、`pretraining.json`、`finetuning.json`、`benchmark.json`：分词器文件与实测数据记录。

## 五、AI 使用说明

- **实验设计与报告撰写**：先用小型模型跑通流程、再扩充为百兆参数标准模型、最后升级为同规模现代架构的递进对比思路，以及从候选架构中选定 Qwen3 均由作者人工提出与决策；课程实验报告正文由作者人工撰写。
- **代码实现、图表绘制、文档撰写与训练执行**：本仓库中的全部 Python 代码、实验结果图片的绘制、训练与评测过程的执行，以及本仓库说明文档（`README.md`）均由 AI 编写与完成。
