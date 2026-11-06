# RGB-Thermal Small Target Detection on AS4000

[中文 README](README_zh.md)

To support RGB-Thermal small target detection research in polar environments, we establish AS4000, a benchmark derived from the NOAA Arctic Seals 2019 aerial survey imagery (AFSC2026). AS4000 contains 3,983 spatially corresponding RGB and thermal images with pixel-level target annotations, and it focuses on extremely small Arctic seals against heterogeneous ice and water backgrounds. Unlike conventional RGB-T benchmarks dominated by generic objects, AS4000 is a task-specific dataset for sparse wildlife targets, complex background interference, and complementary modality responses. RGB images record surface texture, ice structure, and target appearance, including fragmented sea ice, irregular ice boundaries, and complex surface patterns, while thermal images provide radiation-related information and show substantial variation in background intensity and spatial structure. The two modalities are complementary but heterogeneous, and small seals can resemble surrounding ice or local thermal anomalies, which creates considerable foreground-background ambiguity. The 3,983 pairs are divided into 1,500 training pairs, 500 validation pairs, and 1,983 testing pairs. Training and validation are used for optimization and model selection, and the test split is reserved for evaluation. About 45% of the images contain a single target, and the number of images decreases as the target count increases, so the targets are predominantly sparse. Most targets cover about 10–20 pixels, and the signal-to-clutter ratio has a broad distribution, with most targets at relatively low to moderate contrast against the surrounding clutter. Paired RGB-T observations, pixel-level masks, and these target-background statistics make AS4000 a dedicated benchmark for multimodal Arctic-seal small-target detection.

## Method

Accurate detection of Arctic seals in airborne remote sensing imagery is essential for polar wildlife monitoring and ecological assessment. Extremely small target sizes, complex ice backgrounds, and heterogeneous RGB-Thermal (RGB-T) responses make reliable detection difficult. AS4000 is introduced for this task. It contains 3,983 paired RGB and thermal images with pixel-level annotations.

We propose the Multimodal Low-Rank Refinement Fusion Network (MLRF-Net). It integrates Cross-Modal Low-Rank Adaptation (CM-LoRA) and Bidirectional Spatial Refinement Fusion (BSRF). CM-LoRA combines modality-specific adapters with shared low-rank parameters, so the RGB and thermal encoders adapt together while the pretrained weights stay in place and only a small set of parameters is trained. BSRF refines the multimodal features with bidirectional cross-attention and modality-specific spatial refinement. It uses complementary information from the two modalities and improves target localization in complex backgrounds.

On AS4000, MLRF-Net reaches an IoU of 51.93%, an F<sub>1</sub> score of 68.36%, a detection probability of 89.15%, and a false alarm rate of 17.17×10<sup>-6</sup>. All four accuracy metrics are higher than those of the compared methods. The benchmark and source code are available at [https://github.com/VisionVerse/AS4000](https://github.com/VisionVerse/AS4000).

In this repository, CM-LoRA is `cm_lora.py`. Modality-specific LoRA modules are injected into `qkv` and `proj` in the vision trunk and then frozen. Their rank is 16 and alpha is 32. The shared low-rank parameters are LoRA3, with rank 8, alpha 16, and dropout 0.05. LoRA3 is trained, shared by the two modalities, and not registered inside [SAM3](https://github.com/Sompote/SAM3_LoRA). SAM3 itself stays frozen. BSRF is the `M1` module in `bisf.py`. It fuses the last 256-channel FPN level with bidirectional cross-attention, a 3×3 depthwise convolution, and spatial self-attention, without residuals. The fused map is `(R0 + T0) / 2 + Conv1x1(cat(R, T))`, and the 1×1 layer is initialized to zero.

Each image is resized to 1008. The fixed text prompt is `visual`. Ground-truth boxes are positive prompts in normalized `cxcywh` and are passed to the decoder once. Mask logits from the top-k queries, where k is the number of boxes, are merged by a pixel-wise maximum. The training loss is structure loss, sigmoid focal loss, and Dice loss with equal weights. Structure loss is weighted BCE plus weighted IoU. Focal loss uses alpha 0.25 and gamma 2. AdamW uses learning rate `5e-5` and weight decay `0.01`. The checkpoint with the highest validation IoU is kept.

## Setup

Use a Linux machine with an NVIDIA GPU, Python 3.12, and PyTorch 2.7 or newer.

```bash
pip install -r requirements.txt
```

`third_party/sam3` is the SAM 3 package imported by this repository. It comes from [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA). The SAM 3 code is SAM material from Meta and remains under the license in `third_party/sam3/LICENSE`.

## Download links

Paste the four URLs into `paths.py`. The scripts read that file and do not download anything by themselves.

```python
TRAIN_URL = ""
VAL_URL = ""
TEST_URL = ""
WEIGHT_URL = ""
```

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

AS4000 is derived from the NOAA Arctic Seals 2019 aerial survey imagery (AFSC2026). This repository uses SAM 3 under the SAM License.

SAM3 and its LoRA implementation follow [SAM3-LoRA](https://github.com/Sompote/SAM3_LoRA):

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
