\# 数据来源说明



本项目的文字说明由项目作者根据基础物理概念重新整理，不直接复制教材段落。



\## 外部图片



\### projectile\_motion\_001



\- 来源：OpenStax, University Physics Volume 1, 4.3 Projectile Motion

\- 页面：https://openstax.org/books/university-physics-volume-1/pages/4-3-projectile-motion

\- 教材许可：CC BY-NC-SA 4.0

\- 用途：非商业学习与研究项目

\- 备注：上传公开仓库前核对原图图注与署名要求

## 问答基准数据

`data/qa_benchmark.jsonl` 包含 14 条带参考答案的视觉问答样本。

- qa001–qa002：开发集，已用于自由落体和机械能链路调试。
- qa003–qa012：闭集测试集，包含计算、概念和混淆问题。
- qa013–qa014：库外图片测试集，不计入闭集检索指标。
- 每条记录包含图片、问题、参考答案、相关文档、题型和数据划分。
- 测试集在批量运行前固定，不根据模型输出删除或改写困难样本。


\### simple\_harmonic\_motion\_001



\- 来源：OpenStax, University Physics Volume 1, 15.1 Simple Harmonic Motion

\- 页面：https://openstax.org/books/university-physics-volume-1/pages/15-1-simple-harmonic-motion

\- 教材许可：CC BY-NC-SA 4.0



\### series\_parallel\_circuit\_001



\- 来源：OpenStax, College Physics 2e, 21.1 Resistors in Series and Parallel

\- 页面：https://openstax.org/books/college-physics-2e/pages/21-1-resistors-in-series-and-parallel

\- 教材许可：CC BY-NC-SA 4.0

\## 图像查询数据



`data/image\_queries.jsonl` 记录图片查询及其相关文档标注。



\- iq001–iq009：由九张 CC-BY-4.0 自制物理示意图生成，使用旋转、灰度、轻度边缘裁剪、JPEG 压缩、亮度和对比度变化等确定性变换。

\- iq010：自制几何示意图，作为库外查询。

\- iq011：自制细胞示意图，作为库外查询。

\- 生成脚本：`scripts/generate\_image\_queries.py`。

\- 查询图片目录：`data/raw/image\_queries/`。

\- 这些查询评价的是图像变换条件下的近重复检索，不是完全独立、未见过的语义图片测试集。

\- 三个 OpenStax 图片没有用于生成公开提交的查询图片。

