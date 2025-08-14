# Patent Classification with BERT

基于BERT的专利文本二分类系统，用于判断专利文本的有效性。

## 项目简介

本项目实现了一个使用BERT模型进行专利文本二分类的完整工作流程，主要功能包括：

- 专利文本数据预处理和数据集划分
- 基于BERT的模型微调训练
- 模型推理和预测
- 支持GPU加速训练

## 项目结构

```
Patent-Classification-BERT/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包
├── train.py                 # 模型训练脚本
├── inference.py             # 原始推理脚本
├── inference_clf.py         # 分类推理脚本
├── split_dataset.py         # 数据集划分脚本
├── download_model.py        # 模型下载脚本
├── run_train.ps1           # Windows训练启动脚本
├── config/
│   └── config.json         # 配置文件
├── data/                   # 原始数据文件
├── dataset/                # 处理后的数据集
│   ├── data.json          # 原始数据
│   ├── train.jsonl        # 训练集
│   ├── val.jsonl          # 验证集
│   ├── test.jsonl         # 测试集
│   └── split_stats.json   # 数据集统计信息
├── model/                  # 预训练模型文件
│   ├── config.json
│   ├── model.safetensors
│   └── vocab.txt
└── outputs/               # 训练输出目录
    └── bert-classification/
```

## 环境要求

- Python 3.8+
- PyTorch 2.8.0+
- CUDA 12.6+ (可选，用于GPU加速)
- 8GB+ RAM (推荐)
- 4GB+ GPU显存 (可选)

## 安装指南

1. **克隆项目**
```bash
git clone https://github.com/your-username/Patent-Classification-BERT.git
cd Patent-Classification-BERT
```

2. **创建虚拟环境**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 数据格式

### 输入数据格式 (data.json)
```json
[
    {
        "id": "CN1065029C",
        "text": "专利文本内容...",
        "valid": true
    },
    {
        "id": "CN1063527C", 
        "text": "专利文本内容...",
        "valid": false
    }
]
```

### 字段说明
- `id`: 专利唯一标识符
- `text`: 专利文本内容（模型输入）
- `valid`: 布尔值，表示专利是否有效（分类标签）

## 使用指南

### 1. 下载预训练模型

```bash
python download_model.py --model anferico/bert-for-patents --out model
```

### 2. 数据集划分

将原始数据按8:1:1的比例划分为训练集、验证集和测试集：

```bash
python split_dataset.py --input dataset/data.json --outdir dataset --ratios 0.8,0.1,0.1
```

参数说明：
- `--input`: 输入数据文件路径
- `--outdir`: 输出目录
- `--ratios`: 训练集:验证集:测试集比例
- `--seed`: 随机种子（默认42）

### 3. 模型训练

#### 使用脚本启动 (推荐)
```bash
# Windows PowerShell
.\run_train.ps1

# 或手动运行
python train.py --model ./model --train_file ./dataset/train.jsonl --validation_file ./dataset/val.jsonl --output_dir ./outputs/bert-classification
```

#### 训练参数说明
- `--model`: 预训练模型路径
- `--train_file`: 训练数据文件
- `--validation_file`: 验证数据文件
- `--output_dir`: 模型输出目录
- `--max_seq_length`: 最大序列长度（默认512）

#### 数据加载与打包加速
支持两种数据形态:
1. 原始 tokenized jsonl (动态 padding)
2. 预打包定长张量 *_packed.pt (跳过解析与 padding, 启动更快)

在 `config/config.json` 中通过 `packConfig` 控制:
```jsonc
"packConfig": {
   "enable": true,
   "max_seq_length": 512,      // 省略或 null 时自动选取当前 jsonl 最长序列
   "pad_token_id": 0,
   "suffix": "_packed.pt",
   "overwrite": false          // 已存在是否重新生成
}
```
启用后训练启动若发现缺少对应打包文件或需要覆盖，会自动生成 `train_packed.pt` / `val_packed.pt`。

手动批量打包示例:
```bash
python pack_dataset.py --inputs dataset/train.jsonl dataset/val.jsonl --out_dir dataset --max_seq_length 512
```
随后再次训练将直接加载打包文件（省略动态 padding）。如需关闭自动打包，将 `enable` 设为 false。

#### 训练配置
模型会自动检测GPU可用性并调整参数：
- **GPU可用时**: batch_size=16, fp16=True
- **仅CPU时**: batch_size=4, fp16=False

### 4. 模型推理

#### 单条文本预测
```bash
python inference_clf.py --model ./outputs/bert-classification --text "要预测的专利文本内容"
```

#### 批量预测
```bash
# 从文件批量预测
python inference_clf.py --model ./outputs/bert-classification --input_file test_texts.txt --output_file predictions.jsonl
```

#### 推理参数说明
- `--model`: 训练好的模型路径
- `--text`: 单条文本输入
- `--input_file`: 输入文件（.txt或.jsonl格式）
- `--output_file`: 输出文件（可选）
- `--batch_size`: 批处理大小（默认8）

### 5. 输出格式

预测结果示例：
```json
{
    "text": "专利文本内容...",
    "predicted_label": "true",
    "predicted_id": 1,
    "probabilities": {
        "false": 0.2345,
        "true": 0.7655
    }
}
```

## 模型性能

### 训练参数
- 学习率: 2e-5
- 训练轮数: 3
- 批处理大小: 16 (GPU) / 4 (CPU)
- 序列最大长度: 512
- 优化器: AdamW
- 学习率调度: 10% warmup

### 硬件要求
- **最低配置**: CPU训练，8GB RAM
- **推荐配置**: GPU (4GB+ 显存)，16GB RAM
- **训练时间**: 约1-3小时（取决于数据集大小和硬件）

## 故障排除

### 常见问题

1. **CUDA不可用**
   - 确保安装了正确版本的PyTorch CUDA
   - 检查NVIDIA驱动和CUDA版本兼容性

2. **显存不足**
   - 减小批处理大小
   - 启用梯度检查点: `gradient_checkpointing=True`

3. **训练速度慢**
   - 确保使用GPU训练
   - 检查数据加载器的`num_workers`设置

4. **模型效果不佳**
   - 增加训练轮数
   - 调整学习率
   - 检查数据质量和标签平衡性

## 开发指南

### 添加新功能

1. **自定义数据处理**
   - 修改 `split_dataset.py` 中的预处理逻辑
   - 调整 `train.py` 中的 `preprocess_function`

2. **调整模型架构**
   - 修改 `train.py` 中的模型配置
   - 自定义损失函数和评估指标

3. **扩展推理功能**
   - 在 `inference_clf.py` 中添加后处理逻辑
   - 支持更多输入格式

### 代码规范

- 使用类型提示
- 添加详细的文档字符串
- 遵循PEP 8编码规范
- 编写单元测试

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交Issue和Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/your-username/Patent-Classification-BERT/issues)
- 邮箱: your.email@example.com

## 致谢

- [Hugging Face Transformers](https://huggingface.co/transformers/) - 提供BERT模型和训练框架
- [anferico/bert-for-patents](https://huggingface.co/anferico/bert-for-patents) - 专利领域预训练模型
- PyTorch团队 - 深度学习框架

---

**注意**: 本项目仅用于学术研究和技术学习，请确保数据使用符合相关法律法规。
