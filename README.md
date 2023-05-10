# EL-GAN 

## Preparation

### 1. Create virtual environment (optional)
All code was developed and tested on Ubuntu 18.04 with Python 3.8.5 (Anaconda) and PyTorch 1.7.1.

```bash
$ conda create -n EL-GAN python=3.8.5
$ conda activate EL-GAN
```

### 2. Clone the repository
```bash
$ git clone git@github.com:Azure616/EL-GAN.git
$ cd EL-GAN
```

### 3. Install dependencies
```bash
$ pip install -r requirements.txt
```

### 4. Download datasets

To download [COCO-Stuff](http://cocodataset.org) dataset to `datasets/coco`:
```bash
$ bash scripts/download_coco.sh
```

To download [Visual Genome](https://visualgenome.org) dataset to `datasets/vg` and preprocess it:
```bash
$ bash scripts/download_vg.sh
$ python scripts/preprocess_vg.py
```

### 5. Download pretrained models
Download the [trained models](https://drive.google.com/file/d/1--VejuQEBgUExImv7KWMGFtlqYgoSPzG/view?usp=sharing) to `pretrained/`.
## Run codes

### Test models

Test on the COCO-Stuff dataset:
```bash
$ python test.py --dataset coco --model_path pretrained/coco128.pth --sample_path samples/coco128
```

Test on the Visual Genome dataset:
```bash
$ python test.py --dataset coco --model_path pretrained/vg128.pth --sample_path samples/vg128
```

## Results

### 1. Qualitative Results of Different Models (64×64 and 128×128)

Only Layout2Im [ZMYS19], LAMA64 [LWK∗21], and Ours64 images have a resolution of 64×64, and the rest are 128×128.  

<p align='center'><img src='images/results1.png' width='1000px'></p>

### 2. Qualitative Results of Different Models (256×256)

All images are of 256×256 resolution.  

<p align='center'><img src='images/results2.png' width='1000px'></p>



## Contact

If you encounter any problems, please contact us.
## Reference
Our project borrows some source files from LostGANs(https://github.com/WillSuen/LostGANs.git). 