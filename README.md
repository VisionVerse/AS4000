# AS4000: A Benchmark for RGB-Thermal Arctic Seal Small Target Detection via Multimodal Low-Rank Adaptation Fusion

[![Stars](https://img.shields.io/github/stars/VisionVerse/AS4000?style=social)](https://github.com/VisionVerse/AS4000/stargazers)

**🖼️ 3,983** paired aerial images · **🦭 12,554** seals · **📐 mostly 10–20** pixels · **🏷️ pixel-level masks**

This repository is the **1st** work focused on small-target detection of Arctic seals, and AS4000 is the **1st** multimodal dataset built for that task. Remote-sensing imagery of Arctic seals is important for monitoring polar wildlife and for ecological assessment of seal populations in a changing sea-ice environment.

AS4000 is a benchmark for detecting extremely small Arctic seals in airborne RGB–thermal imagery. The dataset is the main release. MLRF-Net is the baseline we provide with it.

<p align="center">
  <img src="assets/seal_show.png" alt="RGB, thermal, and ground-truth examples from AS4000" width="100%">
</p>

<p align="center">
  RGB shows ice texture. Thermal shows a different radiation pattern. The seal is only a few pixels wide.
</p>

## ✨ Highlights

<table>
  <tr><td>🖼️ Pairs</td><td>3,983 spatially aligned RGB–thermal images</td></tr>
  <tr><td>🦭 Targets</td><td>12,554 seals, 3.15 per image</td></tr>
  <tr><td>🏷️ Labels</td><td>Pixel masks and bounding boxes</td></tr>
  <tr><td>📐 Target size</td><td>Most seals cover about 10–20 pixels</td></tr>
  <tr><td>✂️ Split</td><td>1,500 train / 500 val / 1,983 test</td></tr>
</table>

## 🗂️ Dataset

AS4000 is built for RGB–thermal small-target detection in polar environments. Conventional RGB–T benchmarks are dominated by generic objects. AS4000 is built around sparse wildlife, cluttered ice and water, and two modalities that do not describe the scene in the same way.

RGB records surface texture, ice structure, and the visible appearance of a seal: broken sea ice, irregular ice edges, and fine surface patterns. Thermal records radiation. Its background intensity and spatial pattern change substantially from the RGB view. The two signals are complementary and inconsistent. A seal can look like a piece of ice, or like a small thermal glitch.

| Split | Images | Targets | Per image | Download |
| --- | ---: | ---: | ---: | --- |
| Train | 1,500 | 4,300 | 2.87 | [Baidu Netdisk](https://pan.baidu.com/s/1jybcryZ1uE8vph3uX5hVKw?pwd=Seal) |
| Validation | 500 | 871 | 1.74 | [Email](mailto:1546144577@qq.com) |
| Test | 1,983 | 7,383 | 3.72 | [Email](mailto:1546144577@qq.com) |
| **Total** | **3,983** | **12,554** | **3.15** | |

<p align="center"><img src="assets/seal_target_number_distribution_v2.png" alt="Distribution of target count per image" width="32%"> <img src="assets/seal_target_size_distribution_v2.png" alt="Distribution of target size" width="32%"> <img src="assets/seal_target_scr_distribution_v2.png" alt="Distribution of target signal-to-clutter ratio" width="32%"></p>

The signal-to-clutter ratio is broad. Most targets have only low to moderate contrast against the surrounding clutter.

Each sample uses the same file stem across four folders:

```text
RGB/*.jpg    T/*.tif    box/*.txt    GT/*.png
```

Boxes are pixel `xyxy`, one box per line. Mask pixels above 128 are foreground. Thermal TIFF values must keep the normalization in `data.py`, so the frozen thermal LoRA sees the distribution it was trained on.

## 🧠 Baseline: MLRF-Net

The repository also releases a baseline for this benchmark. MLRF-Net targets the same setting: tiny seals, cluttered ice, and RGB–thermal responses that do not match. It combines Cross-Modal Low-Rank Adaptation (CM-LoRA) and Bidirectional Spatial Refinement Fusion (BSRF).

CM-LoRA freezes the pretrained [SAM3](https://github.com/Sompote/SAM3_LoRA) encoders and adapts them with a modality-specific adapter on each branch plus one shared low-rank update, so RGB and thermal features are tuned together while the pretrained weights stay intact.

BSRF exchanges the two feature maps with bidirectional cross-attention and then refines each modality spatially. The fused representation uses complementary cues and makes the small targets easier to localize.

![MLRF-Net framework](assets/mlrf-net-v4.png)

On AS4000, MLRF-Net reaches 51.93% IoU, 68.36% F1, 89.15% detection probability, and a false-alarm rate of 17.17×10<sup>-6</sup>, ahead of the compared methods on all four accuracy metrics.

Comparison of efficiency and accuracy on AS4000. F<sub>a</sub> is reported in units of 10<sup>-6</sup>. ↑ means higher is better, and ↓ means lower is better. The best result in each column is shown in $\color{red}{\pmb{\mathrm{bold\ red}}}$, and the second-best result in $\color{blue}{\mathrm{blue}}$.

| Methods | Pub. | Params ↓ | GFLOPs ↓ | FPS ↑ | IoU ↑ | F<sub>1</sub> ↑ | P<sub>d</sub> ↑ | F<sub>a</sub> ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DNA-Net | TIP, 2023 | 4.70 | 14.28 | 64.20 | 30.00 | 46.15 | 79.05 | 87.30 |
| UIU-Net | TIP, 2023 | 50.54 | 54.50 | 118.40 | 21.49 | 35.37 | 68.74 | 29.65 |
| MSHNet | CVPR, 2024 | 4.07 | $\color{blue}{6.11}$ | 152.30 | 29.23 | 45.23 | 80.94 | 67.28 |
| RPCANet | WACV, 2024 | $\color{red}{\pmb{0.68}}$ | 44.57 | 211.30 | 30.64 | 46.91 | 80.62 | 77.57 |
| SCTransNet | TGRS, 2024 | 11.33 | 10.12 | 109.20 | 20.29 | 33.74 | 62.21 | 25.28 |
| HDNet | TGRS, 2025 | 3.68 | $\color{red}{\pmb{5.70}}$ | 85.50 | 27.26 | 42.84 | $\color{blue}{82.14}$ | 54.50 |
| BGM | TGRS, 2025 | 4.08 | 6.77 | $\color{blue}{212.10}$ | $\color{blue}{30.74}$ | $\color{blue}{47.02}$ | 78.68 | 96.32 |
| ILNet | TAES, 2025 | 4.04 | 21.42 | 108.40 | 19.17 | 32.18 | 58.95 | $\color{blue}{19.30}$ |
| DRPCA-Net | TGRS, 2025 | 1.17 | 73.84 | 11.20 | 29.99 | 46.14 | 76.85 | 104.04 |
| L2SKNet | TGRS, 2025 | $\color{blue}{0.90}$ | 6.90 | $\color{red}{\pmb{332.20}}$ | 30.09 | 46.26 | 77.20 | 86.24 |
| RPCANet++ | TPAMI, 2026 | 2.96 | 190.67 | 80.40 | 30.10 | 46.27 | 76.92 | 88.16 |
| DURD | TGRS, 2026 | 2.47 | 52.85 | 125.30 | 30.06 | 46.23 | 77.58 | 102.84 |
| SAM3 | ICLR, 2026 | 473.59 | 7149.94 | 3.30 | 22.47 | 36.70 | 68.79 | 350.80 |
| **Ours** | -- | 9.09 | 110.11 | 13.61 | $\color{red}{\pmb{51.93}}$ | $\color{red}{\pmb{68.36}}$ | $\color{red}{\pmb{89.15}}$ | $\color{red}{\pmb{17.17}}$ |

## 🛠️ Setup

Linux, an NVIDIA GPU, Python 3.12, and PyTorch 2.7 or newer.

```bash
pip install -r requirements.txt
```

`third_party/sam3` is the SAM 3 package imported by this repository. It comes from [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA). The SAM 3 code is SAM material from Meta and remains under the license in `third_party/sam3/LICENSE`.

## 📦 Download

The four weight files and the prediction results are on Baidu Netdisk.

| Content | Link |
| --- | --- |
| `sam3.pt` | [Baidu Netdisk](https://pan.baidu.com/s/1JwaF6WmSvZWhKpDP5FQzIQ?pwd=SAM3) |
| `rgb_lora.pt` | [Baidu Netdisk](https://pan.baidu.com/s/1bXAps5I6cxG5c9McIgVq6A?pwd=LoRA) |
| `thermal_lora.pt` | [Baidu Netdisk](https://pan.baidu.com/s/18khwgIBfMxJAAub6xG7Fmw?pwd=LoRA) |
| `best.pth` | [Baidu Netdisk](https://pan.baidu.com/s/1Jft2i6gBvHjPJtzwamEaYw?pwd=best) |
| Prediction results | [Baidu Netdisk](https://pan.baidu.com/s/12qjWecExrdNCf_wFrBwfZA?pwd=pred) |

Place the downloaded files as follows. If an archive uses different names, edit the paths in `paths.py` or pass them on the command line.

```text
data/train/RGB/*.jpg
data/train/T/*.tif
data/train/box/*.txt
data/train/GT/*.png
data/val/...
data/test/...

weights/sam3.pt
weights/rgb_lora.pt
weights/thermal_lora.pt
weights/best.pth
```

`sam3.pt` is the SAM3 image checkpoint. `rgb_lora.pt` and `thermal_lora.pt` are the frozen modality LoRA weights. `best.pth` contains the trained BSRF and CM-LoRA state dictionaries.

## 🚀 Train

```bash
python train.py \
  --checkpoint weights/sam3.pt \
  --rgb_lora weights/rgb_lora.pt \
  --thermal_lora weights/thermal_lora.pt \
  --train_data_dir data/train \
  --val_data_dir data/val

torchrun --nproc_per_node=4 train.py \
  --checkpoint weights/sam3.pt \
  --rgb_lora weights/rgb_lora.pt \
  --thermal_lora weights/thermal_lora.pt \
  --train_data_dir data/train \
  --val_data_dir data/val
```

The batch size is 1 image per GPU. With more than one GPU, gradients of BSRF and CM-LoRA are averaged across GPUs. Training runs for 30 epochs by default. The best validation checkpoint is saved to `train_logs/best.pth`.

## 🔎 Test

```bash
python test.py \
  --checkpoint weights/best.pth \
  --sam_checkpoint weights/sam3.pt \
  --rgb_lora weights/rgb_lora.pt \
  --thermal_lora weights/thermal_lora.pt \
  --test_data_dir data/test
```

This writes one 8-bit probability map per test image to `test_maps/`, at the original image resolution.

## 📊 Evaluate

```bash
python evaluate.py --pred_root test_maps
```

The report contains S, weighted F, E, MAE, IoU, F1, Pd, and Fa. IoU and F1 threshold probabilities at 0.5. Pd and Fa match 8-connected components whose centroids are less than 3 pixels apart. Fa is the area of unmatched prediction components divided by the number of image pixels.

<!--
## 📝 Citation

```bibtex
% Paste the BibTeX entry of this paper here.
```

## 📚 References

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

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VisionVerse/AS4000&type=Date)](https://www.star-history.com/#VisionVerse/AS4000&Date)
-->
