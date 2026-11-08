\# CLIP 基础验证实验



\## 实验配置



\- 模型：sentence-transformers/clip-ViT-B-32

\- 运行设备：CPU

\- 测试图片：4 张

\- 正常闭集查询：5 条

\- 库外查询：1 条

\- 图片向量形状：(4, 512)

\- 文本向量形状：(6, 512)

\- 闭集 Top-1 Accuracy：5/5 = 1.0

\## 实验结果



\- Top-1 Accuracy：4/4 = 1.0



| 查询 | Top-1 | Top-1 相似度 | Top-2 | Top-2 相似度 |

|---|---|---:|---|---:|

| coffee beans beside a drink | coffee | 0.2687 | cat | 0.1834 |

| a furry feline | cat | 0.2654 | coffee | 0.1718 |

| a space rocket before launch | rocket | 0.2737 | astronaut | 0.2063 |



其余结果见本地文件 `outputs/clip\_smoke\_test.json`。



\## 初步结论



\- CLIP 图文编码、相似度计算和 Top-K 排序流程已跑通。

\- 当前相似度是向量余弦相似度，不是分类概率。

\- 测试集只有 4 张图片，100% 仅用于验证流程，不能代表正式系统性能。

