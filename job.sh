#!/bin/bash
#SBATCH --job-name=test_conda
#SBATCH --account=m4138_g          # GPU 리포 계정 필수
#SBATCH --time=48:00:00
#SBATCH -C gpu
#SBATCH --qos=shared             # 공유 GPU 정책
#SBATCH --gpus=1                   # GPU 1장
#SBATCH --ntasks=1                 # 태스크 1개
#SBATCH --cpus-per-task=32         # 적당한 CPU 스레드 (필요 시 조정)
#SBATCH -o ./outputs/%x_%j.out
#SBATCH -e ./outputs/%x_%j.err

source ./env.sh

FFN=4
MHA=4
embed=4
ffn_dim=4
NLayer=2
epoch=50
label="251028_v5"

srun python3 train_qt.py --n_q_ffn $FFN --n_q_mha $MHA --ffn_dim $ffn_dim --embed $embed --n_layer $NLayer --epoch $epoch --label $label

