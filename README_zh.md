# AS4000 上的 RGB-热红外小目标检测

![AS4000 的 RGB、热红外与真值示例](assets/seal_show.png)

为支持极地环境下的 RGB-热红外小目标检测研究，我们构建了 AS4000。该基准来自 NOAA Arctic Seals 2019 航空调查影像（AFSC2026），包含 3983 对空间对应的 RGB 与热红外图像，并带有像素级目标标注，关注异构冰面和水面背景中的极小北极海豹。与以通用物体为主的常规 RGB-T 基准不同，AS4000 面向稀疏野生动物目标、复杂背景干扰以及两种模态的互补响应。RGB 图像记录表面纹理、冰面结构和目标外观，背景包括破碎海冰、不规则冰缘和复杂表面纹理；热红外图像提供与辐射相关的信息，背景强度和空间模式变化很大。两种模态互补但并不一致，小海豹容易与周围冰面结构或局部热异常混淆，因此前景和背景存在明显歧义。3983 张图像共有 12554 个目标，平均每张 3.15 个。训练集为 1500 张图、4300 个目标，平均每张 2.87 个；验证集为 500 张图、871 个目标，平均每张 1.74 个；测试集为 1983 张图、7383 个目标，平均每张 3.72 个。训练集用于优化，验证集用于选择模型，测试集只用于性能评估。大多数目标面积集中在约 10–20 像素，空间范围极小。信杂比分布较宽，多数目标相对周围杂波只有偏低到中等的对比度。成对的 RGB-T 观测、像素级掩膜以及这些目标与背景统计，使 AS4000 成为北极海豹多模态小目标检测的专用基准。

## 方法

MLRF-Net 面向航空 RGB-T 图像中的极小北极海豹检测。目标尺寸很小，冰面背景复杂，两路模态的响应也不一致。网络由跨模态低秩适配（CM-LoRA）和双向空间细化融合（BSRF）组成。

CM-LoRA 冻结预训练的 [SAM3](https://github.com/Sompote/SAM3_LoRA) 编码器，在 RGB 和热红外分支上分别使用模态适配器，再用一组共享低秩参数协同更新两路特征，从而在保留预训练知识的同时完成轻量适配。

BSRF 先用双向交叉注意力交换两路特征，再分别做空间细化。融合后的表示吸收互补信息，便于在复杂背景中定位小目标。

在 AS4000 上，MLRF-Net 的 IoU 为 51.93%，F<sub>1</sub> 为 68.36%，检测概率为 89.15%，虚警率为 17.17×10<sup>-6</sup>，四项精度指标均高于所比较的方法。

<table>
  <caption>AS4000 上与现有方法的效率和精度对比。<em>F</em><sub>a</sub> 的单位为 10<sup>-6</sup>。↑ 表示越大越好，↓ 表示越小越好。每列最优结果为<strong style="color:#d1242f">红色加粗</strong>，次优结果为<span style="color:#0969da">蓝色</span>。</caption>
  <thead>
    <tr>
      <th rowspan="2" align="left">方法</th>
      <th rowspan="2" align="left">发表</th>
      <th colspan="3" align="center">效率指标</th>
      <th colspan="4" align="center">精度指标</th>
    </tr>
    <tr>
      <th align="center">Params ↓</th>
      <th align="center">GFLOPs ↓</th>
      <th align="center">FPS ↑</th>
      <th align="center">IoU ↑</th>
      <th align="center"><em>F</em><sub>1</sub> ↑</th>
      <th align="center"><em>P</em><sub>d</sub> ↑</th>
      <th align="center"><em>F</em><sub>a</sub> ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DNA-Net</td><td>TIP, 2023</td>
      <td align="center">4.70</td><td align="center">14.28</td><td align="center">64.20</td>
      <td align="center">30.00</td><td align="center">46.15</td><td align="center">79.05</td><td align="center">87.30</td>
    </tr>
    <tr>
      <td>UIU-Net</td><td>TIP, 2023</td>
      <td align="center">50.54</td><td align="center">54.50</td><td align="center">118.40</td>
      <td align="center">21.49</td><td align="center">35.37</td><td align="center">68.74</td><td align="center">29.65</td>
    </tr>
    <tr>
      <td>MSHNet</td><td>CVPR, 2024</td>
      <td align="center">4.07</td><td align="center"><span style="color:#0969da">6.11</span></td><td align="center">152.30</td>
      <td align="center">29.23</td><td align="center">45.23</td><td align="center">80.94</td><td align="center">67.28</td>
    </tr>
    <tr>
      <td>RPCANet</td><td>WACV, 2024</td>
      <td align="center"><strong style="color:#d1242f">0.68</strong></td><td align="center">44.57</td><td align="center">211.30</td>
      <td align="center">30.64</td><td align="center">46.91</td><td align="center">80.62</td><td align="center">77.57</td>
    </tr>
    <tr>
      <td>SCTransNet</td><td>TGRS, 2024</td>
      <td align="center">11.33</td><td align="center">10.12</td><td align="center">109.20</td>
      <td align="center">20.29</td><td align="center">33.74</td><td align="center">62.21</td><td align="center">25.28</td>
    </tr>
    <tr>
      <td>HDNet</td><td>TGRS, 2025</td>
      <td align="center">3.68</td><td align="center"><strong style="color:#d1242f">5.70</strong></td><td align="center">85.50</td>
      <td align="center">27.26</td><td align="center">42.84</td><td align="center"><span style="color:#0969da">82.14</span></td><td align="center">54.50</td>
    </tr>
    <tr>
      <td>BGM</td><td>TGRS, 2025</td>
      <td align="center">4.08</td><td align="center">6.77</td><td align="center"><span style="color:#0969da">212.10</span></td>
      <td align="center"><span style="color:#0969da">30.74</span></td><td align="center"><span style="color:#0969da">47.02</span></td><td align="center">78.68</td><td align="center">96.32</td>
    </tr>
    <tr>
      <td>ILNet</td><td>TAES, 2025</td>
      <td align="center">4.04</td><td align="center">21.42</td><td align="center">108.40</td>
      <td align="center">19.17</td><td align="center">32.18</td><td align="center">58.95</td><td align="center"><span style="color:#0969da">19.30</span></td>
    </tr>
    <tr>
      <td>DRPCA-Net</td><td>TGRS, 2025</td>
      <td align="center">1.17</td><td align="center">73.84</td><td align="center">11.20</td>
      <td align="center">29.99</td><td align="center">46.14</td><td align="center">76.85</td><td align="center">104.04</td>
    </tr>
    <tr>
      <td>L2SKNet</td><td>TGRS, 2025</td>
      <td align="center"><span style="color:#0969da">0.90</span></td><td align="center">6.90</td><td align="center"><strong style="color:#d1242f">332.20</strong></td>
      <td align="center">30.09</td><td align="center">46.26</td><td align="center">77.20</td><td align="center">86.24</td>
    </tr>
    <tr>
      <td>RPCANet++</td><td>TPAMI, 2026</td>
      <td align="center">2.96</td><td align="center">190.67</td><td align="center">80.40</td>
      <td align="center">30.10</td><td align="center">46.27</td><td align="center">76.92</td><td align="center">88.16</td>
    </tr>
    <tr>
      <td>DURD</td><td>TGRS, 2026</td>
      <td align="center">2.47</td><td align="center">52.85</td><td align="center">125.30</td>
      <td align="center">30.06</td><td align="center">46.23</td><td align="center">77.58</td><td align="center">102.84</td>
    </tr>
    <tr>
      <td>SAM3</td><td>ICLR, 2026</td>
      <td align="center">473.59</td><td align="center">7149.94</td><td align="center">3.30</td>
      <td align="center">22.47</td><td align="center">36.70</td><td align="center">68.79</td><td align="center">350.80</td>
    </tr>
    <tr>
      <td><strong>本文</strong></td><td>--</td>
      <td align="center">9.09</td><td align="center">110.11</td><td align="center">13.61</td>
      <td align="center"><strong style="color:#d1242f">51.93</strong></td><td align="center"><strong style="color:#d1242f">68.36</strong></td><td align="center"><strong style="color:#d1242f">89.15</strong></td><td align="center"><strong style="color:#d1242f">17.17</strong></td>
    </tr>
  </tbody>
</table>

## 环境

在带 NVIDIA GPU 的 Linux 机器上运行，Python 3.12，PyTorch 2.7 或更高版本。

```bash
pip install -r requirements.txt
```

`third_party/sam3` 是本仓库实际导入的 SAM 3 代码，来自 [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA)。其中的 SAM 3 代码属于 Meta 的 SAM 材料，许可见 `third_party/sam3/LICENSE`。

## 下载链接

训练集和 4 个权重文件放在百度网盘。验证集或测试集请联系 [1546144577@qq.com](mailto:1546144577@qq.com)。

| 内容 | 链接 |
| --- | --- |
| 训练集 | [【百度网盘】](https://pan.baidu.com/s/1jybcryZ1uE8vph3uX5hVKw?pwd=Seal) |
| 验证集 | [【邮箱】](mailto:1546144577@qq.com) |
| 测试集 | [【邮箱】](mailto:1546144577@qq.com) |
| `sam3.pt` | [【百度网盘】](https://pan.baidu.com/s/1JwaF6WmSvZWhKpDP5FQzIQ?pwd=SAM3) |
| `rgb_lora.pt` | [【百度网盘】](https://pan.baidu.com/s/1bXAps5I6cxG5c9McIgVq6A?pwd=LoRA) |
| `thermal_lora.pt` | [【百度网盘】](https://pan.baidu.com/s/18khwgIBfMxJAAub6xG7Fmw?pwd=LoRA) |
| `best.pth` | [【百度网盘】](https://pan.baidu.com/s/1Jft2i6gBvHjPJtzwamEaYw?pwd=best) |

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

```bibtex
% 在这里粘贴本论文的 BibTeX。
```

## 参考文献

```bibtex
@inproceedings{carion2026sam,
  title={Sam 3: Segment anything with concepts},
  author={Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and Debnath, Shoubhik and Hu, Ronghang and Suris Coll-Vinent, Didac and Ryali, Chaitanya and Alwala, Kalyan Vasudev and Khedr, Haitham and Huang, Andrew and others},
  booktitle={International conference on learning representations},
  volume={2026},
  pages={138846--138923},
  year={2026}
}

@software{sam3_lora,
  title = {SAM3-LoRA: Low-Rank Adaptation for Fine-Tuning},
  author = {AI Research Group, KMUTT},
  year = {2026},
  version = {0.3.0},
  organization = {King Mongkut's University of Technology Thonburi},
  url = {https://github.com/Sompote/SAM3_LoRA}
}
```
