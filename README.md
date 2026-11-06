[中文说明](README_zh.md)

# RGB-Thermal Small Target Detection on AS4000

To support RGB-Thermal small target detection research in polar environments, we establish AS4000, a benchmark derived from the NOAA Arctic Seals 2019 aerial survey imagery (AFSC2026). AS4000 contains 3,983 spatially corresponding RGB and thermal images with pixel-level target annotations, and it focuses on extremely small Arctic seals against heterogeneous ice and water backgrounds. Unlike conventional RGB-T benchmarks dominated by generic objects, AS4000 is a task-specific dataset for sparse wildlife targets, complex background interference, and complementary modality responses. RGB images record surface texture, ice structure, and target appearance, including fragmented sea ice, irregular ice boundaries, and complex surface patterns, while thermal images provide radiation-related information and show substantial variation in background intensity and spatial structure. The two modalities are complementary but heterogeneous, and small seals can resemble surrounding ice or local thermal anomalies, which creates considerable foreground-background ambiguity. The 3,983 pairs are divided into 1,500 training pairs, 500 validation pairs, and 1,983 testing pairs. Training and validation are used for optimization and model selection, and the test split is reserved for evaluation. About 45% of the images contain a single target, and the number of images decreases as the target count increases, so the targets are predominantly sparse. Most targets cover about 10–20 pixels, and the signal-to-clutter ratio has a broad distribution, with most targets at relatively low to moderate contrast against the surrounding clutter. Paired RGB-T observations, pixel-level masks, and these target-background statistics make AS4000 a dedicated benchmark for multimodal Arctic-seal small-target detection.

## Method

The code trains and evaluates the full model.

- SAM3 is frozen.
- An RGB LoRA and a thermal LoRA are injected into `qkv` and `proj` in the vision trunk, then frozen. Rank is 16 and alpha is 32.
- One shared LoRA3 is attached to the same linear layers and is trained. Rank is 8, alpha is 16, and dropout is 0.05. Its parameters are shared by the two modalities and are not registered inside SAM3.
- BISF (`M1` in `bisf.py`) fuses the last 256-channel FPN level and is trained. It runs bidirectional spatial cross-attention, a 3x3 depthwise convolution, and spatial self-attention, all without residuals. The fused map is `(R0 + T0) / 2 + Conv1x1(cat(R, T))`, and the 1x1 layer is zero-initialized.

Each image is resized to 1008. The fixed text prompt is `visual`. Ground-truth boxes are positive prompts in normalized `cxcywh` and are passed to the decoder once. Mask logits from the top-k queries, where k is the number of boxes, are merged by a pixel-wise maximum. The training loss is structure loss, sigmoid focal loss, and Dice loss with equal weights. Structure loss is weighted BCE plus weighted IoU. Focal loss uses alpha 0.25 and gamma 2. AdamW uses learning rate `5e-5` and weight decay `0.01`. The checkpoint with the highest validation IoU is kept.

## Setup

Use a Linux machine with an NVIDIA GPU, Python 3.12, and PyTorch 2.7 or newer.

```bash
pip install -r requirements.txt
```

`third_party/sam3` is the SAM 3 package imported by this repository. It is SAM material from Meta and remains under the license in `third_party/sam3/LICENSE`.

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
