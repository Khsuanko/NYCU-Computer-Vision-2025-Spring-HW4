# NYCU-Computer-Vision-2025-Spring-HW4
StudentID: 110550122  
Name: 柯凱軒

## Introduction
The objective of this task is to restore clean images from degraded images (two types: rain and snow). Image restoration in the presence of degradations like rain and snow is crucial for improving the visual quality of images captured in adverse weather conditions. We aim to build a single deep learning model that can effectively remove both types of degradations, thus improving the Peak Signal-to-Noise Ratio (PSNR) of the restored images.
  
Our method leverages PromptIR, a transformer-based image restoration model. PromptIR employs prompt-based feature learning to adaptively handle various degradations. We train this model from scratch without any external data or pre-trained weights.




## How to install
1. Install Dependencies  
```python
pip install torch torchvision torchaudio numpy opencv-python scikit-image tqdm matplotlib pycocotools
```
2. Ensure you have the dataset structured as follows:
```python
./data/
    ├── train/
    ├── test/
```
3. Run the code
```python
python train.py
python test.py
```
## Performance snapshot
![performance](https://github.com/Khsuanko/NYCU-Computer-Vision-2025-Spring-HW4/blob/main/performance.png)
