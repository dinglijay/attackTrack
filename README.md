# attackTrack
Adversarial  Attack on Visual Object Tracking

## Setup SiamMask
1. Clone the repository **on the ../ path**
```
    git clone https://github.com/foolwood/SiamMask.git
```
2. Download the SiamMask pretrained model
```
    cd ../SiamMask/experiments/siammask_sharp
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
```
## Setup PythonPath in attackTrack
```
    . export.sh
```
## Run Demo
```
    python demo.py \
           --resume ../SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth \
           --config ../SiamMask/experiments/siammask_sharp/config_davis.json \
           --gt_file groundtruth_rect.txt \
           --base_path ../DylanTrack/dataset/OTB/Car24/img/
```
