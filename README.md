# 作业三：从零构建大模型

源码与实验结果：<https://github.com/LHX369963/embodied-intelligence-homework3>

模型权重发布页：<https://github.com/LHX369963/embodied-intelligence-homework3/releases/tag/submission>

## 一、知识点与实验设计

自回归语言模型把序列概率分解为各位置条件概率的乘积，以前文预测下一个词元。预训练的标签来自原文右移一位，监督信号由文本自身构成。文本先经过字节级 BPE 分词，再查表获得词嵌入，与可学习的位置嵌入相加。

模型由四个仅解码器 Transformer 块组成。每块采用前置层归一化、多头因果自注意力、残差连接和 GELU 前馈网络。注意力计算为 softmax(QKᵀ/√dₖ + M)V，其中未来位置的掩码值为负无穷；每个预测只能利用当前位置及其之前的输入。输出层与输入词嵌入共享权重。

预训练最小化下一词元交叉熵，以 exp(平均交叉熵) 定义词元级困惑度。分类微调保留解码器主干，以序列末尾 EOS 的表征接二分类线性层，监督目标为影评正负情感。比较预训练主干全参数微调、冻结主干仅训练分类头，以及随机主干全参数训练。验证集负责选模，测试集用于最终评价。

## 二、顺序执行

1. 知识点梳理与方案确认。
2. 实现模型、分词器和核心单元测试。
3. 下载 WikiText-2 原始文本，训练分词器并从随机参数预训练，记录学习曲线和文本生成。
4. 构建 Rotten Tomatoes 影评监督样本，完成微调和对照，记录准确率、宏 F1、混淆矩阵和案例。
5. 整理报告、核心代码附录和推理入口。

## 数据与参考

- 课程作业原文：`../../考试与小测/作业三(2026).pdf`。
- 参考实现与教学思路：https://github.com/rasbt/LLMs-from-scratch 。本项目自行实现小型 GPT，使用 PyTorch 张量与优化器接口，不加载外部预训练参数。
- 预训练数据：https://huggingface.co/datasets/Salesforce/wikitext ，配置 `wikitext-2-raw-v1`，沿用官方训练、验证和测试划分。分词器仅在训练文本上拟合。
- 微调数据：https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes ，沿用官方划分，标签 0 为负面、1 为正面。研究与教学使用，数据权益以来源说明为准。
- 分词器：https://huggingface.co/docs/tokenizers/python/latest/quicktour.html 。
- 注意力论文：https://arxiv.org/abs/1706.03762 。
- GPT-2 报告：https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf 。

## 实测结果

模型具有 4,240,896 个参数。预训练完成 2500 步，累计采样 20,480,000 个词元；按验证损失选择第 2500 步。相同分词器与上下文评价方式下，测试词元困惑度由随机初始化的 4306.96 降至 72.28。

| 监督训练方式 | 选定轮次 | 测试准确率 | 宏 F1 |
| --- | ---: | ---: | ---: |
| 冻结预训练主干 | 6 | 59.57% | 0.5948 |
| 随机主干全参数训练 | 2 | 73.73% | 0.7371 |
| 预训练后全参数微调 | 1 | 70.54% | 0.7052 |

各组均训练 6 轮，采用验证交叉熵最低的检查点。完整结果、逐条测试预测与超参数见 `pretraining.json`、`finetuning.json`。本次固定种子为 2026，表格展示该次运行，未作多种子统计。影评划分之间的完全相同文本数量均为零，见 `data_audit.json`。

模型学得局部语言结构和文本分类能力；本次随机主干组分类表现高于预训练微调组。WikiText 通用语言建模目标与影评情感目标存在差异，全参数微调后期还出现验证损失上升。生成内容适合观察学习过程，事实信息需要另外核实。

## 运行与推理

使用现有 Python 环境安装 `requirements.txt` 中的软件包，无需下载外部预训练模型。实际训练环境的软件版本见 `environment.json`。

```bash
python -m pip install -r requirements.txt
python test_model.py
python prepare.py
python experiment.py pretrain --steps 2500
python experiment.py finetune --epochs 6
python inference.py generate "The film was" --new-tokens 60
python inference.py classify "A moving and beautifully acted film."
```

`prepare.py` 从官方数据集仓库下载六份 Parquet，在 WikiText 训练文本上拟合分词器，生成本地词元数组。`experiment.py` 自动选择可用加速设备，也支持 CPU。`inference.py` 默认使用 CPU；以 4 个线程连续测试 30 次单条分类，包含分词与张量构造的平均时间为 2.29 ms（示例文本和测量方法见结果及代码）。

预训练模型接收英语文本并逐词元续写。分类模型接收一条影评并输出正负情感概率。分类任务使用新建的二分类头，生成任务继续使用预训练语言模型头。

## 交付内容

核心源代码为 `model.py`、`prepare.py`、`experiment.py`、`inference.py`、`test_model.py`，分词器为 `tokenizer.json`。`model_tests.json` 保存因果性、共享权重、梯度、保存加载与右侧填充等检查结果。`plot_results.py` 从实际结果生成图件，绘图另需 Matplotlib、Seaborn、Poppler 及可显示中文的字体；脚本使用 `assets/Times_New_Roman+SimSun.ttf`，该本地字体文件不在公开仓库分发。

模型文件位于 `checkpoints/`：`pretrained.pt` 为语言模型，`full.pt` 为全参数微调模型；`scratch.pt` 与 `frozen.pt` 为对照模型。各文件约 16 MiB，作为 Release 附件分发，下载后放入 `checkpoints/` 即可。公开数据从上述来源下载，数据与模型权重不纳入 Git 历史。

```bash
mkdir -p checkpoints
for name in pretrained full scratch frozen; do
  curl -L "https://github.com/LHX369963/embodied-intelligence-homework3/releases/download/submission/$name.pt" \
    -o "checkpoints/$name.pt"
done
```

权重及结果固定对应本项目分词器与配置。`pretrained.pt` 用于续写，`full.pt` 用于主实验微调推理；`scratch.pt` 在本次分类测试中取得三组最高准确率，作为对照一并提供。

本地报告使用 `main.typ` 编写、`typst compile main.typ main.pdf` 编译。含个人信息的这两个报告文件保留本地；公开仓库提供源代码、结果和模型下载。
