#!/bin/sh
cd ../SiamMask
export SiamMask=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
cd ../../../attackTrack
# export DISPLAY=saimServer2:10.0
# export CUDA_VISIBLE_DEVICES=1
