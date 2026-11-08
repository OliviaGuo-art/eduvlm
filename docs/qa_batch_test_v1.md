\# QA 批量实验 test\_v1



\## 运行配置



\- 测试集：data/qa\_benchmark.jsonl 中 split=test 的样本

\- 测试样本：12 条

\- 闭集样本：10 条

\- 库外样本：2 条

\- 问答模型：qwen3-vl-plus

\- 检索模型：BAAI/bge-small-zh-v1.5

\- Top-K：3

\- 最大输出：700 Token

\- 温度：0.2

\- 直接回答：12 条

\- RAG 回答：12 条



\## 完成情况



\- Selected items：12

\- Complete items：12

\- 总 Token：36598

\- 原始结果目录：outputs/qa\_batch/test\_v1/



每条结果单独保存为 qa\_id.json。批量脚本支持断点续跑，已完成样本不会重复请求。



原始答案不提交到 GitHub；后续提交评分、汇总指标和失败案例分析。

