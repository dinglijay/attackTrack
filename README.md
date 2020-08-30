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

## Setup Pysot
1. Clone the repository **on the ../ path**
```
    git clone https://github.com/STVIR/pysot.git
```
2. Build extensions
```
    cd pysot/
    python setup.py build_ext --inplace
```
3. Download pretrained model
```
    # siamrpn_alex_dwxcorr
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e51IL1UZ-5seum2yUYpf98l2lJGUTnhs' -O experiments/siamrpn_alex_dwxcorr/model.pth

    # siamrpn_mobilev2_l234_dwxcorr
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lPiRjFvajwrhOHVuXrygAj2cRb2BFFfz' -O experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth

    # siammask_r50_l3
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dQoI2o5Bzfn_IhNJNgcX4OE79BIHwr8s' -O experiments/siammask_r50_l3/model.pth

    # siamrpn_r50_l234_dwxcorr
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH" -O experiments/siamrpn_r50_l234_dwxcorr/model.pth && rm -rf /tmp/cookies.txt
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

    python ../pysot/tools/demo.py \
           --config ../pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
           --snapshot ../pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth \
           --video ../pysot/demo/bag.avi 
```
## Kornia Bug Fix
```
    # kornia.color.hsv.py L140:
    # avoid gradient to NAN when Backward 
    s: torch.Tensor = deltac / (v + 1e-20)
```

### Test and Evaluate
```
    # Link data 
    cd pysot/testing_dataset
    ln -s /DataServer/tracking_dbs/vot2019 VOT2019
    ln -s /DataServer/tracking_dbs/OTB100 OTB100
    cd attackTrack/data
    ln -s /DataServer/tracking_dbs/vot2019 VOT2019
    ln -s /DataServer/tracking_dbs/OTB100 OTB100

    # test in pysot
    python -u ../../tools/test.py --snapshot model.pth --dataset OTB100 --config config.yaml
    python ../../tools/eval.py --tracker_path ./results --dataset OTB100 --num 4 --tracker_prefix 'model'

    # test in attackTracking
    python experiment/test.py --snapshot ../pysot/experiments/siammask_r50_l3/model.pth --dataset OTB100 --config ../pysot/experiments/siammask_r50_l3/config.yaml
    python experiment/eval.py --tracker_path ./results --dataset OTB100 --num 4 --tracker_prefix 'model'
```

### Test and Evaluation on LaSOT
```
python experiment/test.py --snapshot ../pysot/experiments/siammask_r50_l3/model.pth --dataset LaSOT-cup --config ../pysot/experiments/siammask_r50_l3/config.yaml --video cup-10 --vis

python experiment/test.py --snapshot ../pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth --dataset LaSOT-cup --config ../pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml --patch data/lasot/cup/cup-7/large_siamrpn_feat-delta-tv.png

python experiment/eval.py --tracker_path ./results --dataset LaSOT-car --num 4 --tracker_prefix 'model'

python experiment/test.py --snapshot ../pysot/experiments/siamrpn_alex_dwxcorr/model.pth  --dataset LaSOT-cup --config ../pysot/experiments/siamrpn_alex_dwxcorr/config.yaml --video cup-10 --vis
```

### scripts
``` 
python script/video2imgs.py data/own/Human/Human4/Human4.mp4
python script/genGt.py data/own/Human/Human4
python dataset/create_json.py data/own

```

### WebCam Tracking
```
python myutils/pysot_track.py --snapshot ../pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth --config ../pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml

python myutils/mask_track.py --video_name data/physical/Bottlenew/VID_20200825_181144.mp4
```