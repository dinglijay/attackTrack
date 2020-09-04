#!/bin/sh
cd ../pysot
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../SiamMask
export SiamMask=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../../../attackTrack
export DISPLAY=attackSer1:10.0
# export CUDA_VISIBLE_DEVICES=1
