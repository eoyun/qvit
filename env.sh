#!/bin/bash

module load python
conda activate cml

export QVIT_PROFILE=0
export TRANSFORMERS_CACHE=$SCRATCH/hf/transformers
export HF_HOME=$SCRATCH/hf
export HF_DATASETS_CACHE=$SCRATCH/hf/datasets
