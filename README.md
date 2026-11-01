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
- 参考实现与教学思路：https://github.com/rasbt/LLMs-from-scratch 。本项目自行实现小型 GPT，使用 PyTorch 张量与优化器接口，参数统一随机初始化。
- 预训练数据：https://huggingface.co/datasets/Salesforce/wikitext ，配置 `wikitext-2-raw-v1`，沿用官方训练、验证和测试划分。分词器仅在训练文本上拟合。
- 微调数据：https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes ，沿用官方划分，标签 0 为负面、1 为正面。研究与教学使用，数据权益以来源说明为准。
- 分词器：https://huggingface.co/docs/tokenizers/python/latest/quicktour.html 。
- 注意力论文：https://arxiv.org/abs/1706.03762 。
- GPT-2 报告：https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf 。

## 实测结果

模型具有 4,240,896 个参数。预训练在第 19500 步触发验证早停，累计采样 159,744,000 个词元，恢复第 19000 步的最低验证损失检查点。相同分词器与上下文评价方式下，测试词元困惑度由随机初始化的 4306.96 降至 56.06。

| 监督训练方式 | 停止轮次 | 选定轮次 | 测试准确率 | 宏 F1 |
| --- | ---: | ---: | ---: | ---: |
| 冻结预训练主干 | 4 | 2 | 60.60% | 0.6034 |
| 随机主干全参数训练 | 4 | 2 | 73.64% | 0.7362 |
| 预训练后全参数微调 | 6 | 4 | 71.67% | 0.7127 |

预训练每 250 步验证一次，分类每轮验证一次。两阶段均以历史最低验证交叉熵为基准，连续两次评估处于该值及以上时结束训练，并载入最低损失检查点评估测试集。本次预训练和三组分类训练均由该早停规则结束。代码设置预训练 20000 步和分类 50 轮的上限，实际停止位置见表格与 JSON。

预训练学习率在前 100 步预热至 0.0006，第 2500 步余弦衰减至 0.00006，之后维持该学习率。完整结果、停止原因、计数、逐条测试预测和超参数见 `pretraining.json`、`finetuning.json`。本次随机种子为 2026，影评划分之间的完全相同文本数量均为零，见 `data_audit.json`。

模型学得局部语言结构和文本分类能力。本次随机主干组分类准确率高于预训练微调组，三组对照展示了预训练表征与情感分类目标之间的实际迁移表现。生成样例展示英文局部结构与专名拼接等现象。

## 运行与推理

使用现有 Python 环境安装 `requirements.txt` 中的软件包。实际训练环境的软件版本见 `environment.json`。

```bash
python -m pip install -r requirements.txt
python test_model.py
python prepare.py
python experiment.py pretrain --steps 20000
python experiment.py finetune --epochs 50
python inference.py generate "The film was" --new-tokens 60
python inference.py classify "A moving and beautifully acted film."
```

`prepare.py` 从官方数据集仓库下载六份 Parquet，在 WikiText 训练文本上拟合分词器，生成本地词元数组。`experiment.py` 执行训练与验证选模，`inference.py` 加载选定模型完成文本生成或影评分类。

预训练模型接收英语文本并逐词元续写。分类模型接收一条影评并输出正负情感概率。分类任务使用新建的二分类头，生成任务继续使用预训练语言模型头。

## 交付内容

核心源代码为 `model.py`、`prepare.py`、`experiment.py`、`inference.py`、`test_model.py`，分词器为 `tokenizer.json`。`model_tests.json` 保存因果性、共享权重、梯度、保存加载、右侧填充与早停计数及重置检查结果。`plot_results.py` 从实际结果生成图件，绘图另需 Matplotlib、Seaborn、Poppler 及可显示中文的字体；脚本使用 `assets/Times_New_Roman+SimSun.ttf`，该本地字体文件不在公开仓库分发。

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
