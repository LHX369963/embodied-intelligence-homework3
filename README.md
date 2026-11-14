# 作业三：从零构建 515M 参数 GPT 语言模型与监督微调

- 源码与实测结果仓库：<https://github.com/LHX369963/embodied-intelligence-homework3>
- 模型权重发布页（GitHub Release）：<https://github.com/LHX369963/embodied-intelligence-homework3/releases/tag/submission>

## 一、快速跑通（教师与助教验收入口）

`inference.py` 内置按需自动下载与国内镜像加速切换逻辑。克隆仓库并安装依赖后，首次执行推理命令会自动将所需权重（`bfloat16` 存储，单文件约 `982 MiB`）下载至 `checkpoints/` 目录并显示实时进度，支持纯 CPU、NVIDIA CUDA、AMD ROCm 与 Apple MPS 直接运行。

```bash
git clone https://github.com/LHX369963/embodied-intelligence-homework3.git
cd embodied-intelligence-homework3
python -m pip install -r requirements.txt

# 1. 一键完整演示（自动拉取 pretrained.pt 与 full.pt，依次展示模型配置、英文续写与四条影评情感分类）
python inference.py demo

# 2. 自定义提示词自由续写（可通过 --device auto 自动调用本机硬件加速）
python inference.py generate "The history of artificial intelligence" --new-tokens 80

# 3. 自定义影评情感分类（输出预测类别与正负面概率 JSON，可通过 --variant 切换 full / frozen / scratch）
python inference.py classify "A deeply moving, visually stunning film with unforgettable performances."
```

若测试机器处于离线环境，可在联网设备上执行 `python inference.py download --all` 预先拉取全部四份权重，或直接从上述 Release 页面下载 `.pt` 文件放入 `checkpoints/` 目录。若校园网直连 GitHub 较慢，可指定镜像前缀运行：

```bash
GITHUB_PROXY=https://ghfast.top python inference.py demo
```

## 二、模型架构与实验设计

本项目仅使用 PyTorch 基础张量与神经网络模块，从随机初始化实现完整的仅解码器（Decoder-only）GPT 语言模型与下游分类迁移框架，并开展两阶段规模扩展横向对比：

| 配置项 | 阶段一轻量原型（4.24M） | 阶段二主力模型（514.86M） |
| --- | --- | --- |
| 参数总量 | `4,240,896` | `514,859,520` |
| 子词分词词表 | 字节级 BPE，`4,096` | 字节级 BPE，`32,768` |
| 最大上下文长度 | `128` | `512` |
| 解码器层数 / 隐藏维度 / 注意力头数 | `4` 层 / `256` 维 / `4` 头（每头 `64` 维） | `24` 层 / `1280` 维 / `20` 头（每头 `64` 维） |
| 前馈网络与归一化 | `256 -> 1024 -> 256`（GELU），前置层归一化 | `1280 -> 5120 -> 1280`（GELU），前置层归一化 |
| 权重共享与推理加速 | 输入词嵌入与语言模型输出层共享权重 | 共享词嵌入权重 + 逐层键值缓存（KV-Cache）增量解码 |
| 预训练语料与词元总量 | WikiText-2（`3,083,246` 词元） | WikiText-103 + IMDb 无标签影评（`128,993,363` 词元） |

- **预训练机制**：将序列联合概率按链式法则分解为逐位置条件概率乘积，输入与目标错开一位优化下一词元交叉熵。采用解耦权重衰减 AdamW（二维权重矩阵衰减系数 `0.1`，一维偏置与层归一化参数衰减系数 `0.0`）、线性预热与余弦退火学习率计划、`bfloat16` 混合精度、`torch.compile` 图算子融合与微批次梯度累积（`8 × 8 = 64` 有效批大小，单步 `32,768` 词元）。
- **监督微调机制**：保留预训练解码器主干，提取序列末尾 `<eos>` 位置的隐藏状态接入二分类线性层（遵循 GPT-1 判别任务迁移结构），在 Rotten Tomatoes 影评数据集上对比三种训练方式：**冻结预训练主干仅训练分类头（`frozen`）**、**随机初始化全参数训练（`scratch`）**、**加载预训练主干全参数微调（`full`，主干学习率 `2e-5`、分类头学习率 `2e-4` 分层预热余弦退火）**。

## 三、实测结果与跨规模公平基准评测（Benchmark）

### 1. 515M 主力模型预训练与监督微调结果

515M 模型在单张 AMD Instinct MI100（32 GB）上完成 `5,000` 步预训练（累计采样 `163,840,000` 词元，约合混合语料的 `1.27` 倍，峰值显存 `20.07 GB`）。在官方 WikiText 测试集上，词元困惑度由随机初值的 `41414.63` 降至 `30.57`，每字节交叉熵由 `3.2429 bpb` 降至 `1.0432 bpb`。

在 Rotten Tomatoes 官方测试集（`1,066` 条，正负各 `533` 条，512 窗口下截断样本数为 `0`）上的三组对照结果如下：

| 监督训练方式 | 可训练参数量 | 停止轮次 | 选定轮次 | 测试准确率 | 宏 F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 随机初始化全参数训练（`scratch`） | `514,862,082` | 5 | 2 | `71.86%` | `0.7085` | `0.8445` |
| 冻结预训练主干（`frozen`） | `2,562` | 10 | 9 | `74.95%` | `0.7488` | `0.8259` |
| 预训练全参数微调（`full`） | `514,862,082` | 5 | 2 | `79.74%` | `0.7947` | `0.8975` |

仅训练 `2,562` 个线性分类头参数的冻结主干组即超越随机初始化全参数组 `+3.09` 个百分点；全参数微调组进一步提升至 `79.74%`（ROC-AUC `0.8975`），领先随机初始化组 `+7.88` 个百分点，消除了 4.24M 原型阶段因语料单薄与步长过大产生的负迁移现象。

### 2. 4.24M 与 515M 跨规模横向基准对比

为消除 `4096` 与 `32768` 词表粒度差异，`benchmark.py` 在同一批官方 WikiText 测试文章（`1,287,656` 字节 / `241,211` 英文单词）、公开的 BLiMP 英语语法最小对评测集（5 类核心语法现象共 `5,000` 对句子）以及零样本/少样本上下文学习任务上进行统一尺度对比：

| 评测维度与指标 | 4.24M 原型 | 515M（窗口 128） | 515M（完整窗口 512） | 变化幅度 |
| --- | ---: | ---: | ---: | --- |
| WikiText 测试集每字节比特数（BPB，越低越优） | `1.6601` | `1.1061` | `1.0432` | 降低 `37.2%` |
| WikiText 测试集词级困惑度（Word-PPL，越低越优） | `465.32` | `59.90` | `47.47` | 降低 `89.8%` |
| BLiMP 5000 组语法最小对零样本准确率 | `65.54%` | — | `84.20%` | `+18.66` 个百分点 |
| 影评零样本提示分类（Zero-Shot）准确率 / AUC | `49.34%` / `0.4950` | — | `58.82%` / `0.6480` | `+9.48` 个百分点 |
| 影评少样本上下文学习（4-Shot ICL）准确率 / AUC | `49.34%` / `0.4816` | — | `59.47%` / `0.6830` | `+10.13` 个百分点 |
| 监督全参微调测试准确率（含否定转折难例切片） | `71.67%`（`66.76%`） | — | `79.74%`（`75.72%`） | 整体 `+8.07` / 难例 `+8.96` 个百分点 |

## 四、从零完整复现流程

```bash
python test_model.py                          # 运行因果掩码、KV 缓存等价性、共享权重与早停单元测试
python prepare.py                             # 下载公开 Parquet 语料，拟合 32768 字节级 BPE 分词器并核查数据重合度
python experiment.py pretrain --steps 5000    # 执行 515M GPT 混合精度预训练与测试集评估
python experiment.py finetune --epochs 10     # 执行冻结主干、随机初始化与全参数微调三组对照评估
python benchmark.py --scale 515m              # 运行跨词表 BPB、BLiMP 语法、零样本/少样本与难例切片基准评测
```

## 五、文件结构与公开数据来源

- 核心实现：`model.py`（模型结构与 KV 缓存）、`prepare.py`（语料拉取与分词器训练）、`experiment.py`（预训练与三组微调）、`inference.py`（自动拉取权重与命令行推理）、`benchmark.py`（横向基准评测）、`test_model.py`（底层单元测试）、`plot_results.py`（矢量图生成）。
- 实验记录：`tokenizer.json`、`data_info.json`、`data_audit.json`、`environment.json`、`model_tests.json`、`pretraining.json`、`finetuning.json`、`benchmark.json`。
- 公开数据集与参考来源：
  - WikiText-103：<https://huggingface.co/datasets/Salesforce/wikitext>（`wikitext-103-raw-v1`）
  - IMDb 无标签语料：<https://huggingface.co/datasets/stanfordnlp/imdb>（`unsupervised` 划分）
  - Rotten Tomatoes 影评分类：<https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes>
  - BLiMP 语言学最小对基准：<https://huggingface.co/datasets/nyu-mll/blimp>
  - 教学参考实现：<https://github.com/rasbt/LLMs-from-scratch>
