\# EduVLM：多模态学习资料检索与问答系统



一个面向基础物理学习资料的多模态检索与问答实验项目，实现并比较：



\- 文本查询到学习资料的语义检索；

\- 图片查询到文档图片的 CLIP 检索；

\- 图片与问题直接输入视觉语言模型；

\- 检索增强问答（RAG），返回答案、Top-K 证据和引用校验结果；

\- 直接问答与 RAG 的人工评分、Token 成本及失败案例分析；

\- Gradio 本地交互演示。



本项目重点不是训练大模型，而是构建一个可运行、可评价、可分析失败原因的多模态 RAG 流程。



\## 系统架构



```mermaid

flowchart LR

&#x20;   Image\[输入图片] --\&gt; Direct\[直接问答]

&#x20;   Question\[输入问题] --\&gt; Direct

&#x20;   Direct --\&gt; VLM1\[Qwen-VL]

&#x20;   VLM1 --\&gt; DirectAnswer\[直接答案]



&#x20;   Question --\&gt; BGE\[BGE 文本向量检索]

&#x20;   Documents\[物理文档 title + text] --\&gt; BGE

&#x20;   BGE --\&gt; TopK\[Top-K 候选资料]

&#x20;   Image --\&gt; RAG\[检索增强问答]

&#x20;   Question --\&gt; RAG

&#x20;   TopK --\&gt; Evidence\[证据上下文]

&#x20;   Evidence --\&gt; RAG

&#x20;   RAG --\&gt; VLM2\[Qwen-VL]

&#x20;   VLM2 --\&gt; RAGAnswer\[RAG 答案]

&#x20;   RAGAnswer --\&gt; Citation\[引用编号校验]

&#x20;   TopK --\&gt; Citation



&#x20;   QueryImage\[图片查询] --\&gt; CLIP\[CLIP 图像向量检索]

&#x20;   DocImages\[文档图片] --\&gt; CLIP

&#x20;   CLIP --\&gt; ImageResults\[图片检索评价]



&#x20;   DirectAnswer --\&gt; UI\[Gradio 展示与本地记录]

&#x20;   RAGAnswer --\&gt; UI

&#x20;   Citation --\&gt; UI

&#x20;   TopK --\&gt; UI

```



Gradio 主问答流程使用 BGE 检索问题对应的文字资料。CLIP 图片检索作为独立检索实验，用于验证图片到文档图片的检索能力。



\## 数据集



当前包含 12 篇基础物理资料，每篇包含：



\- `doc\_id`；

\- 标题与正文；

\- 对应图片；

\- 标签；

\- 来源与许可信息。



检索数据：



\- 20 条文字查询：18 条闭集开发查询、2 条库外查询；

\- 11 条图片查询：9 条图像变换近重复查询、2 条库外图片查询。



问答数据：



\- `data/qa\_benchmark.jsonl` 共 14 条；

\- `qa001–qa002` 为开发集；

\- `qa003–qa014` 为测试集；

\- 测试集包含 10 条闭集问题和 2 条库外图片问题；

\- 每条样本包含图片、问题、参考答案、相关文档、题型和数据划分。



详细来源与许可说明见 `data/README.md`。三个 OpenStax WebP 图片目前不直接提交，需根据具体图片署名信息核对 reuse 条件；自制图片和生成脚本另行提供。



\## 检索实验



\### 中文文字检索对比



相同 12 篇文档、18 条闭集开发查询和评价流程：



| 模型 | Recall@1 | Recall@3 | MRR |

|---|---:|---:|---:|

| CLIP 文本基线 | 0.2778 | 0.6111 | 0.4810 |

| BGE 中文语义检索 | 0.9444 | 1.0000 | 0.9722 |



该结果说明模型与任务匹配的重要性，但数据规模较小，不能将指标推广为通用检索准确率。



\### 图片检索



在 9 条由自制文档图生成的旋转、灰度、轻度裁剪、压缩、亮度和对比度变换查询上：



\- Recall@1：1.0000；

\- Recall@3：1.0000；

\- MRR：1.0000。



这些结果仅表示图像变换近重复检索，不代表对任意新图片的语义泛化能力。两条库外图片仍被返回了物理文档，说明检索系统不会自动获得拒答能力。



\## 直接问答与 RAG 评价



`test\_v1` 使用 12 条测试样本，每条分别调用直接问答和 RAG，共 24 个回答。



| 指标 | 直接回答 | RAG 回答 |

|---|---:|---:|

| 平均正确性（0–2） | 2.000 | 2.000 |

| 平均完整性（0–1） | 0.833 | 1.000 |

| 平均图片依据（0–1） | 1.000 | 1.000 |

| 平均无依据陈述得分（0–1） | 1.000 | 1.000 |

| 平均通用总分（0–5） | 4.833 | 5.000 |

| 总 Token | 16311 | 20287 |

| length 数量 | 2 | 0 |



RAG 证据指标：



\- 引用编号有效率：11/12＝91.67%；

\- 平均证据相关性：2.000/2；

\- 平均引用支持性：1.000/1；

\- RAG 相比直接回答增加 3976 Token，约增加 24.4%。



在本测试集上，没有观察到 RAG 纠正直接回答物理错误的案例，因此不能声称 RAG 提高了正确率。当前主要收益是证据可追溯和回答完整，代价是额外 Token。



\## 典型失败与处理



\### 检索排名错误



开发集 q018 中，正确的机械能资料排名第二，摩擦资料排名第一。RAG 在该样本中仍选择了正确资料，说明 Top-1 错误不一定导致最终答案错误，但不能据此认为生成模型总能纠正检索错误。



\### 生成截断



qa007 和 qa008 的直接回答因达到输出 Token 上限而以 `length` 结束。系统保留并展示 `finish\_reason`，不将截断回答视为完整回答。



\### 引用格式失败



qa008 的 RAG 回答生成了字面量 `【doc\_id】`。项目随后：



\- 增加引用编号提取与 Top-K 合法性校验；

\- 在提示词中加入具体正确示例并禁止输出占位符；

\- 在 Gradio 中展示引用警告；

\- 保留 test\_v1 原始结果，不覆盖或美化失败案例。



编号合法只表示它来自本次候选集，不代表资料内容一定支持回答，证据支持性仍需评价。



\## 本地安装



建议使用 Python 3.11 和独立 Conda 环境：



```bat

conda create -n eduvlm python=3.11 -y

conda activate eduvlm

python -m pip install -r requirements.txt

```



项目已在 Windows、CPU 环境下运行。没有 NVIDIA GPU 也可以完成检索实验；视觉问答通过在线 Qwen-VL API 调用，不需要本地加载大型视觉语言模型。



\## API 配置



复制示例配置：



```bat

copy .env.example .env

```



在本机 `.env` 中填写：



```text

DASHSCOPE\_API\_KEY=你的APIKey

DASHSCOPE\_WORKSPACE\_ID=你的业务空间ID

DASHSCOPE\_REGION=cn-beijing

QWEN\_VL\_MODEL=qwen3-vl-plus

```



`.env` 已被 Git 忽略。不要提交真实 API Key，不要将 Key 写入源码或公开截图。



调用视觉问答 API 可能产生费用。批量测试前应先用 `--limit 1` 验证流程。



\## 运行 Gradio



```bat

python src\\app.py

```



终端显示本地地址后，在浏览器打开：



```text

http://127.0.0.1:7860

```



支持：



\- 直接问答；

\- 检索增强问答；

\- 两种模式对比；

\- Top-K 与最大输出 Token 设置；

\- 证据表、引用校验、结束原因和 Token 展示。



每次运行保存到 `outputs/gradio\_runs/`，不会提交到 GitHub。按 `Ctrl+C` 停止本地服务。



如果 Hugging Face 模型已缓存但反复发生网络检查超时，可在当前 CMD 中设置：



```bat

set HF\_HUB\_OFFLINE=1

set TRANSFORMERS\_OFFLINE=1

python src\\app.py

```



这些变量只影响 Hugging Face 模型加载，不会阻止 Qwen-VL API 调用。



\## 主要实验命令



生成自制图和图像查询：



```bat

python scripts\\generate\_diagrams.py

python scripts\\generate\_image\_queries.py

```



运行文字检索对比：



```bat

python src\\text\_retrieval.py --model BAAI/bge-small-zh-v1.5 --output outputs\\text\_retrieval\_bge\_small\_zh.json --query-instruction "为这个句子生成表示以用于检索相关文章："

```



运行图片检索：



```bat

python src\\image\_retrieval.py

```



先运行一条问答测试：



```bat

python src\\qa\_batch.py --split test --run-name test\_v1 --limit 1

```



确认后断点续跑：



```bat

python src\\qa\_batch.py --split test --run-name test\_v1

```



批量实验会调用付费 API。已完成的样本会跳过，不重复请求。更改实验配置时请使用新的 `--run-name`，不要覆盖旧实验。



生成评分表：



```bat

python src\\make\_qa\_scoring\_sheet.py

```



运行引用校验测试：



```bat

python src\\test\_citation\_validation.py

```



\## 项目结构



```text

eduvlm/

├── data/

│   ├── documents.jsonl

│   ├── queries.jsonl

│   ├── image\_queries.jsonl

│   ├── qa\_benchmark.jsonl

│   └── README.md

├── docs/

│   ├── retrieval\_model\_comparison.md

│   ├── image\_retrieval.md

│   ├── qa\_evaluation\_protocol.md

│   ├── qa\_evaluation\_test\_v1.md

│   └── failure\_handling.md

├── scripts/

│   ├── generate\_diagrams.py

│   └── generate\_image\_queries.py

├── src/

│   ├── text\_retrieval.py

│   ├── image\_retrieval.py

│   ├── qwen\_vl.py

│   ├── qa\_comparison.py

│   ├── qa\_batch.py

│   ├── citation\_validation.py

│   ├── test\_citation\_validation.py

│   └── app.py

├── outputs/                 # 本地原始结果，不提交

├── .env.example

├── .gitignore

└── requirements.txt

```



\## 结果复现与局限



\- 文档、查询和问答测试集规模较小；

\- 部分示意图包含明显文字和公式，直接模型可能借此答对；

\- 图片查询主要是近重复变换，不是独立语义图片测试；

\- 库外样本数量有限，尚未解决通用拒答问题；

\- API 模型版本、提示词和生成参数会影响结果；

\- 原始输出保存在本地 `outputs/`，GitHub 文档保存汇总指标与失败分析；

\- 不根据模型回答删除困难样本，也不将开发集结果混入测试集。



后续可增加独立真实拍照图片、更大规模测试集、重排序、结构化引用输出和独立拒答评价。

