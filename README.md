# SpaXFus: Spatial-X Fusion for Remote Sensing Images

This is the official implementation of the paper **'Spatial-X Fusion for Remote Sensing Images'**.

SpaXFus is a unified framework designed for four distinct remote sensing image fusion tasks.

---

## 📂 Project Structure

```text
├── data/                   # Place your datasets here
├── dataset.py              # Dataset loader for pan-sharpening
├── dataset_mh.py           # Dataset loader for Hyper- & Multi-spectral Fusion
├── dataset_sar.py          # Dataset loader for PolSAR Fusion
├── dataset_stf.py          # Dataset loader for Spatiotemproal Fusion
├── train_adalr_HMFus.py    # Training script for Hyper- & Multi-spectral Fusion
├── train_adalr_PanSharP.py # Training script for pan-sharpening
├── train_adalr_PolSARFus.py# Training script for PolSAR Fusion
├── train_adalr_STFus.py    # Training script for Spatiotemproal Fusion
└── test.py                 # Unified testing script
```

## 🛠 Data Preparation

1. Download the provided datasets and place them into the `data/` folder.  
   - If you use the datasets provided in this work, you only need to download them and adjust the corresponding paths in the training/testing scripts.

2. If you want to use your own datasets:  
   - Make sure to structure your data according to the corresponding `dataset*.py` file (e.g., `dataset.py`, `dataset_mh.py`, etc.).  
   - After organizing your dataset, modify the paths in the training/testing scripts to point to your custom data.

---

## 🚀 Training

To start training, select the script corresponding to your task. 

### Configuration
Before running the scripts, you must manually set the following parameters inside the respective `train_adalr_*.py` file:
* `ratio`: The spatial resolution enhancement ratio.
* `hrms_channels`: Number of channels for HR image.
* `lrms_channels`: Number of channels for LR image.

### Run Commands

#### 1. Pansharpening
```bash
python train_adalr_PanSharP.py --batchSize 4 --lr 0.0001 --gpus "0"
```
#### 2. Hyperspectral and Multispectral Fusion
```bash
python train_adalr_HMFus.py --batchSize 4 --lr 0.0001 --gpus "0"
```
#### 3. Spatiotemproal Fusion
```bash
python train_adalr_STFus.py --batchSize 4 --lr 0.0001 --gpus "0"
```
#### 4. PolSAR Fusion
```bash
python train_adalr_PolSARFus.py --batchSize 4 --lr 0.0001 --gpus "0"
```
## 🧪 Testing
All tasks use a single unified testing script.

1. Open `test.py`.

2. Modify the checkpoint path to point to your trained .pth file.

3. Modify the test dataset path to point to your testing data.

4. Run the script:
```bash
python test.py
```
The results can be found in `output/` folder.  
## 📝 Citation
If this work is helpful in your research, please cite our paper:

**BibTeX:**

```bibtex
@article{spaxfus2024,
  title={Spatial-X Fusion for Remote Sensing Images},
  author={Jiang He and et al.},
  journal={RSE},
  year={2025}
}
