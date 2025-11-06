#!/bin/bash

module load python
conda activate cml

unset DISPLAY
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export QVIT_PROFILE=0
export TRANSFORMERS_CACHE=$SCRATCH/hf/transformers
export HF_HOME=$SCRATCH/hf
export HF_DATASETS_CACHE=$SCRATCH/hf/datasets
