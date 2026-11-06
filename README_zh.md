# AS4000 上的 RGB-热红外小目标检测

为支持极地环境下的 RGB-热红外小目标检测研究，我们构建了 AS4000。该基准来自 NOAA Arctic Seals 2019 航空调查影像（AFSC2026），包含 3983 对空间对应的 RGB 与热红外图像，并带有像素级目标标注，关注异构冰面和水面背景中的极小北极海豹。与以通用物体为主的常规 RGB-T 基准不同，AS4000 面向稀疏野生动物目标、复杂背景干扰以及两种模态的互补响应。RGB 图像记录表面纹理、冰面结构和目标外观，背景包括破碎海冰、不规则冰缘和复杂表面纹理；热红外图像提供与辐射相关的信息，背景强度和空间模式变化很大。两种模态互补但并不一致，小海豹容易与周围冰面结构或局部热异常混淆，因此前景和背景存在明显歧义。3983 对图像划分为 1500 对训练、500 对验证和 1983 对测试。训练集用于优化，验证集用于选择模型，测试集只用于性能评估。约 45% 的图像只包含一个目标，随着目标数量增加，图像数量下降，目标分布以稀疏为主。大多数目标面积集中在约 10–20 像素，空间范围极小。信杂比分布较宽，多数目标相对周围杂波只有偏低到中等的对比度。成对的 RGB-T 观测、像素级掩膜以及这些目标与背景统计，使 AS4000 成为北极海豹多模态小目标检测的专用基准。

## 方法

MLRF-Net 面向航空 RGB-T 图像中的极小北极海豹检测。目标尺寸很小，冰面背景复杂，两路模态的响应也不一致。网络由跨模态低秩适配（CM-LoRA）和双向空间细化融合（BSRF）组成。

CM-LoRA（`cm_lora.py`）冻结预训练的 [SAM3](https://github.com/Sompote/SAM3_LoRA) 编码器，在 RGB 和热红外分支上分别使用模态适配器，再用一组共享低秩参数协同更新两路特征，从而在保留预训练知识的同时完成轻量适配。

BSRF（`bisf.py` 中的 `M1`）先用双向交叉注意力交换两路特征，再分别做空间细化。融合后的表示吸收互补信息，便于在复杂背景中定位小目标。

在 AS4000 上，MLRF-Net 的 IoU 为 51.93%，\(F_1\) 为 68.36%，检测概率为 89.15%，虚警率为 \(17.17\times10^{-6}\)，四项精度指标均高于所比较的方法。

## 环境

在带 NVIDIA GPU 的 Linux 机器上运行，Python 3.12，PyTorch 2.7 或更高版本。

```bash
pip install -r requirements.txt
```

`third_party/sam3` 是本仓库实际导入的 SAM 3 代码，来自 [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA)。其中的 SAM 3 代码属于 Meta 的 SAM 材料，许可见 `third_party/sam3/LICENSE`。

## 下载链接

把下面四个链接填进 `paths.py`。脚本只读取这个文件，不会自动下载。

```python
TRAIN_URL = ""
VAL_URL = ""
TEST_URL = ""
WEIGHT_URL = ""
```

下载后按下面的结构放置。如果压缩包里的文件名不同，修改 `paths.py` 里的路径，或在命令行传入对应参数。

训练集、验证集和测试集：

```text
data/train/RGB/*.jpg
data/train/T/*.tif
data/train/box/*.txt
data/train/GT/*.png
data/val/...
data/test/...
```

同一张样本在 `RGB`、`T`、`box` 和 `GT` 中的文件主名必须一致。框文件是像素 `xyxy`，一行一个框。掩膜中大于 128 的像素视为前景。热红外 TIFF 的归一化写在 `data.py` 中，请保持原有运算顺序，这样冻结的热红外 LoRA 才能看到原来的输入分布。三个划分分别有 1500、500 和 1983 对图像。

权重：

```text
weights/sam3.pt
weights/rgb_lora.pt
weights/thermal_lora.pt
weights/best.pth
```

`sam3.pt` 是 SAM3 图像模型权重。`rgb_lora.pt` 和 `thermal_lora.pt` 是冻结的模态 LoRA。`best.pth` 保存训练好的 `m1` 和 `lora3`。

## 训练

```bash
python train.py
torchrun --nproc_per_node=4 train.py
```

每张 GPU 每次处理 1 张图像。多卡时，M1 和 LoRA3 的梯度在卡间取平均。默认训练 30 个 epoch，学习率每 50 个 epoch 乘以 0.1。验证集上最好的权重保存到 `train_logs/best.pth`。

## 测试

```bash
python test.py
```

每张测试图像输出一张 8 位概率图，保存在 `test_maps/`，分辨率为原始图像尺寸。

## 评估

```bash
python evaluate.py --pred_root test_maps
```

输出 S、加权 F、E、MAE、IoU、F1、Pd 和 Fa。IoU 与 F1 在概率大于 0.5 处二值化。Pd 和 Fa 使用 8 连通域，质心距离小于 3 像素视为匹配。Fa 是未匹配预测连通域的面积除以图像总像素数。

## 计算量

```bash
python efficiency.py
```

统计 RGB LoRA、热红外 LoRA、LoRA3 和 M1 的参数量、GFLOPs 和 FPS。LoRA3 在两路模态上各执行一次，计算量按两次计入。SAM3 主干、解码器和文本编码不计入。一次乘加计为 1，GFLOPs 等于乘加次数除以 1e9。

## 引用

AS4000 来自 NOAA Arctic Seals 2019 航空调查影像（AFSC2026）。本仓库在 SAM License 下使用 SAM 3。

SAM3 及其 LoRA 实现遵循 [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA)：

```bibtex
@software{sam3_lora,
  title = {SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning},
  author = {AI Research Group, KMUTT},
  year = {2026},
  version = {0.3.0},
  organization = {King Mongkut's University of Technology Thonburi},
  url = {https://github.com/Sompote/SAM3_LoRA}
}
```
