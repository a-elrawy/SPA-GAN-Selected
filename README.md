# Spatial Attention GAN for Image-to-Image Translation

PyTorch Implementation of [SPA-GAN](https://arxiv.org/pdf/1908.06616.pdf).
## Overview
![alt text](img/cyclegan.png)

## Architecture
![img.png](img/img.png)
#### CycleGAN (a) and SPA-GAN (b) architecture 

## Prerequites
* [Python](https://www.continuum.io/downloads)
* [PyTorch](http://pytorch.org/)


<br>

## Usage

#### Clone the repository

```bash
git clone https://github.com/a-elrawy/SPA-GAN.git
cd SPA-GAN/
```

#### Train the model

```bash
python main.py
```

<br>

## Results

#### 1) Facades

From Facades to Map            |  From Map to Facades
:-------------------------:|:-------------------------:
![alt text](img/spa-f-m.png)  |  ![alt text](img/spa-m-f.png)

