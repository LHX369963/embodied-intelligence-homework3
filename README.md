# 作业三：从零构建大模型

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

实验完成后，本文件补充运行命令与交付清单。报告章节将依照作业的五项要求排列。
