# RGB-Thermal Small Target Detection on AS4000

[中文 README](README_zh.md)

![RGB, thermal, and ground-truth examples from AS4000](assets/seal_show.png)

To support RGB-Thermal small target detection research in polar environments, we establish AS4000, a benchmark derived from the NOAA Arctic Seals 2019 aerial survey imagery (AFSC2026). AS4000 contains 3,983 spatially corresponding RGB and thermal images with pixel-level target annotations, and it focuses on extremely small Arctic seals against heterogeneous ice and water backgrounds. Unlike conventional RGB-T benchmarks dominated by generic objects, AS4000 is a task-specific dataset for sparse wildlife targets, complex background interference, and complementary modality responses. RGB images record surface texture, ice structure, and target appearance, including fragmented sea ice, irregular ice boundaries, and complex surface patterns, while thermal images provide radiation-related information and show substantial variation in background intensity and spatial structure. The two modalities are complementary but heterogeneous, and small seals can resemble surrounding ice or local thermal anomalies, which creates considerable foreground-background ambiguity. The 3,983 images contain 12,554 targets in total, or 3.15 targets per image. The training split has 1,500 images and 4,300 targets (2.87 per image), the validation split has 500 images and 871 targets (1.74 per image), and the test split has 1,983 images and 7,383 targets (3.72 per image). Training and validation are used for optimization and model selection, and the test split is reserved for evaluation. Most targets cover about 10–20 pixels, and the signal-to-clutter ratio has a broad distribution, with most targets at relatively low to moderate contrast against the surrounding clutter. Paired RGB-T observations, pixel-level masks, and these target-background statistics make AS4000 a dedicated benchmark for multimodal Arctic-seal small-target detection.

## Method

MLRF-Net detects extremely small Arctic seals in airborne RGB-T imagery, where the targets are tiny, the ice background is cluttered, and the two modalities respond differently. The network combines Cross-Modal Low-Rank Adaptation (CM-LoRA) and Bidirectional Spatial Refinement Fusion (BSRF).

CM-LoRA freezes the pretrained [SAM3](https://github.com/Sompote/SAM3_LoRA) encoders and adapts them with a modality-specific adapter on each branch plus one shared low-rank update, so RGB and thermal features are tuned together while the pretrained weights stay intact.

BSRF exchanges the two feature maps with bidirectional cross-attention and then refines each modality spatially. The fused representation uses complementary cues and makes the small targets easier to localize.

On AS4000, MLRF-Net reaches 51.93% IoU, 68.36% F1, 89.15% detection probability, and a false-alarm rate of 17.17×10<sup>-6</sup>, ahead of the compared methods on all four metrics.

<table>
  <caption>Comparison of efficiency and accuracy on AS4000. <em>F</em><sub>a</sub> is reported in units of 10<sup>-6</sup>. ↑ means higher is better, and ↓ means lower is better. The best result in each column is shown in <strong style="color:#d1242f">bold red</strong>, and the second-best result in <span style="color:#0969da">blue</span>.</caption>
  <thead>
    <tr>
      <th rowspan="2" align="left">Methods</th>
      <th rowspan="2" align="left">Pub.</th>
      <th colspan="3" align="center">Efficiency</th>
      <th colspan="4" align="center">Accuracy</th>
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
      <td><strong>Ours</strong></td><td>--</td>
      <td align="center">9.09</td><td align="center">110.11</td><td align="center">13.61</td>
      <td align="center"><strong style="color:#d1242f">51.93</strong></td><td align="center"><strong style="color:#d1242f">68.36</strong></td><td align="center"><strong style="color:#d1242f">89.15</strong></td><td align="center"><strong style="color:#d1242f">17.17</strong></td>
    </tr>
  </tbody>
</table>

## Setup

Use a Linux machine with an NVIDIA GPU, Python 3.12, and PyTorch 2.7 or newer.

```bash
pip install -r requirements.txt
```

`third_party/sam3` is the SAM 3 package imported by this repository. It comes from [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA). The SAM 3 code is SAM material from Meta and remains under the license in `third_party/sam3/LICENSE`.

## Download links

The training set and the four weight files are on Baidu Netdisk. For the validation set or the test set, email [1546144577@qq.com](mailto:1546144577@qq.com).

| Content | Link |
| --- | --- |
| Training set | [Baidu Netdisk](https://pan.baidu.com/s/1jybcryZ1uE8vph3uX5hVKw?pwd=Seal) |
| Validation set | [Email](mailto:1546144577@qq.com) |
| Test set | [Email](mailto:1546144577@qq.com) |
| `sam3.pt` | [Baidu Netdisk](https://pan.baidu.com/s/1JwaF6WmSvZWhKpDP5FQzIQ?pwd=SAM3) |
| `rgb_lora.pt` | [Baidu Netdisk](https://pan.baidu.com/s/1bXAps5I6cxG5c9McIgVq6A?pwd=LoRA) |
| `thermal_lora.pt` | [Baidu Netdisk](https://pan.baidu.com/s/18khwgIBfMxJAAub6xG7Fmw?pwd=LoRA) |
| `best.pth` | [Baidu Netdisk](https://pan.baidu.com/s/1Jft2i6gBvHjPJtzwamEaYw?pwd=best) |
| Prediction results | [Baidu Netdisk](https://pan.baidu.com/s/12qjWecExrdNCf_wFrBwfZA?pwd=pred) |

After downloading, arrange the files as follows. If an archive uses different names, edit the path constants in `paths.py` or pass the command-line arguments.

Training, validation, and test splits:

```text
data/train/RGB/*.jpg
data/train/T/*.tif
data/train/box/*.txt
data/train/GT/*.png
data/val/...
data/test/...
```

A stem must match across `RGB`, `T`, `box`, and `GT`. Boxes are pixel `xyxy`, one box per line. Mask foreground is any PNG value above 128. Thermal TIFF values use the normalization in `data.py`; keep that arithmetic order so the frozen thermal LoRA sees its original input distribution. The three splits contain 1,500, 500, and 1,983 pairs.

Weights:

```text
weights/sam3.pt
weights/rgb_lora.pt
weights/thermal_lora.pt
weights/best.pth
```

`sam3.pt` is the SAM3 image checkpoint. `rgb_lora.pt` and `thermal_lora.pt` are the frozen modality LoRA weights. `best.pth` contains the trained `m1` and `lora3` state dictionaries.

## Train

```bash
python train.py
torchrun --nproc_per_node=4 train.py
```

The batch size is 1 image per GPU. With more than one GPU, gradients of M1 and LoRA3 are averaged across GPUs. Training runs for 30 epochs by default, and the learning rate is multiplied by 0.1 every 50 epochs. The best validation checkpoint is saved to `train_logs/best.pth`.

## Test

```bash
python test.py
```

This writes one 8-bit probability map per test image to `test_maps/`, at the original image resolution.

## Evaluate

```bash
python evaluate.py --pred_root test_maps
```

The report contains S, weighted F, E, MAE, IoU, F1, Pd, and Fa. IoU and F1 threshold probabilities at 0.5. Pd and Fa match 8-connected components whose centroids are less than 3 pixels apart. Fa is the area of unmatched prediction components divided by the number of image pixels.

## Efficiency

```bash
python efficiency.py
```

The script reports parameters, GFLOPs, and FPS for RGB LoRA, thermal LoRA, LoRA3, and M1. LoRA3 is executed on both modalities, so its compute is counted twice. The SAM3 trunk, decoder, and text encoder are left out of the total. One multiply-add counts as 1 GFLOP / 1e9.

## Citation

```bibtex
% Paste the BibTeX entry of this paper here.
```

## References

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VisionVerse/AS4000&type=Date)](https://www.star-history.com/#VisionVerse/AS4000&Date)
