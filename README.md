# AdaML

AdaML 核心模型代码。

## 目录

- `main.py`：训练、验证和测试入口。
- `plus/`：需求提取、需求图、六输入噪声估计和损失函数。
- `layer/`：匹配层。
- `data/`：数据加载。
- `util/`：日志、随机种子和评估指标。

## 使用

在本目录安装依赖并运行：

```bash
python -m pip install -r requirements.txt

python main.py --dataset diginetica --ad True
python main.py --dataset tafeng --ad True
python main.py --dataset retailrocket --ad True
```
