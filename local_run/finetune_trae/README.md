# BERT向量和排序模型微调项目

本项目提供了基于BERT模型的向量和排序模型微调功能，支持带权重的均方差损失函数（向量化模型）和Pairwise损失函数（排序模型）。特别注意：排序模型采用了query和文档拼接后输入的方式进行评分。

## 项目功能

1. **自动生成样本数据**：生成100条符合格式要求（query、doc、rel、weight）的样本数据，并按60:20:20的比例分割为训练集、测试集和验证集。
2. **向量化模型微调**：使用带权重的均方差损失函数对BERT模型进行微调，使其能够生成高质量的文本向量表示。
3. **排序模型微调**：使用Pairwise损失函数对BERT模型进行微调，采用query和文档拼接后输入的方式进行评分，使其能够有效地对文本进行排序。
4. **模型评估**：提供模型评估功能，计算损失值和NDCG等指标。
5. **模型部署**：将训练好的模型部署为FastAPI服务，提供RESTful API接口。

## 文件结构

```
finetune_trae/
├── generate_sample_data.py  # 样本数据生成脚本
├── config.yaml              # 配置文件
├── run_finetune.py          # 主执行脚本
├── README.md                # 项目说明文档
├── data/                    # 数据目录（自动生成）
│   ├── train_data.json      # 训练集数据
│   ├── test_data.json       # 测试集数据
│   └── val_data.json        # 验证集数据
└── output/                  # 模型输出目录（自动生成）
    ├── vector_model/        # 向量化模型输出
    └── ranking_model/       # 排序模型输出
```

## 环境要求

本项目基于LLaMA-Factory环境，需要以下依赖：

- Python 3.10+
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn
- YAML

可以通过以下命令安装依赖：

```bash
pip install -r /Users/python_projects/LLaMA-Factory/requirements.txt
```

## 使用指南

### 1. 生成样本数据

运行以下命令生成100条样本数据：

```bash
python run_finetune.py --generate
```

生成的数据将保存在`data`目录下，包含：
- `train_data.json`: 60条训练数据
- `test_data.json`: 20条测试数据
- `val_data.json`: 20条验证数据

### 2. 微调向量化模型

运行以下命令微调向量化模型：

```bash
python run_finetune.py --vector
```

向量化模型使用带权重的均方差损失函数，训练好的模型将保存在`output/vector_model`目录下。

### 3. 微调排序模型

运行以下命令微调排序模型：

```bash
python run_finetune.py --ranking
```

排序模型使用Pairwise损失函数，训练好的模型将保存在`output/ranking_model`目录下。

### 4. 评估模型

评估向量化模型：

```bash
python run_finetune.py --eval-vector
```

评估排序模型：

```bash
python run_finetune.py --eval-ranking
```

### 5. 部署模型

部署向量化模型为API服务：

```bash
python run_finetune.py --deploy-vector
```

部署排序模型为API服务：

```bash
python run_finetune.py --deploy-ranking
```

### 6. 运行所有流程

运行以下命令执行所有流程（生成数据、微调向量化模型、微调排序模型、评估模型）：

```bash
python run_finetune.py --all
```

## 配置说明

可以通过修改`config.yaml`文件来自定义模型参数：

- **通用配置**：模型名称、最大序列长度、设备选择、保存路径等
- **向量化模型配置**：学习率、批次大小、训练轮数等
- **排序模型配置**：学习率、批次大小、训练轮数、损失函数类型等
- **部署配置**：服务端口、主机地址等

## 实现细节

### 向量化模型

向量化模型使用BERT模型提取文本的向量表示，采用带权重的均方差损失函数进行训练。训练过程中，模型学习将输入文本映射到一个高维向量空间，使得相似文本的向量表示在空间中距离更近。

### 排序模型

排序模型基于BERT模型，采用Pairwise损失函数进行训练。**重要改进**：在最新实现中，排序模型不再分别处理query和文档，而是使用`[SEP]`分隔符将它们拼接后一起输入模型，直接学习query和文档的联合表示，这样的实现更符合现代排序模型的设计理念。训练过程中，模型学习区分相关文档和不相关文档，对于给定的查询，给相关文档分配更高的分数，给不相关文档分配更低的分数。

## 数据格式

样本数据格式为JSON，每条数据包含以下字段：

- `key`: 样本的唯一标识
- `query`: 查询文本
- `doc`: 文档文本
- `rel`: 相关度得分（0.0-1.0）
- `weight`: 样本权重（0.5-1.5）

示例：

```json
{
  "key": "sample_1",
  "query": "什么是人工智能？",
  "doc": "人工智能是一种重要的技术，它具有以下特点：...",
  "rel": 0.95,
  "weight": 1.2
}
```

## API接口说明

### 向量化模型API

- **端口**：8001
- **接口**：
  - `/embed`: 获取单个文本的向量表示
  - `/embed_batch`: 批量获取文本的向量表示
  - `/score`: 计算查询和文档的相似度分数

### 排序模型API

- **端口**：8002
- **接口**：
  - `/score`: 计算查询和文档的排序分数（自动拼接query和文档）
  - `/score_batch`: 批量计算查询和文档的排序分数（自动拼接query和每个文档）

## 注意事项

1. 如果没有GPU，可以在`config.yaml`中将`device`设置为`cpu`。
2. 首次运行时，系统会自动下载BERT模型，可能需要一些时间。
3. 训练过程中的日志会同时输出到控制台和`finetune.log`文件中。
4. 可以根据实际需求调整`config.yaml`中的参数，以获得更好的模型性能。