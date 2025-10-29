#!/bin/bash
#SBATCH --job-name=classic
#SBATCH --account=m4138_g          # GPU 리포 계정 필수
#SBATCH --time=48:00:00
#SBATCH -C gpu
#SBATCH --qos=shared              # 공유 GPU 정책
#SBATCH --gpus=1                   # GPU 1장
#SBATCH --ntasks=1                 # 태스크 1개
#SBATCH --cpus-per-task=32         # 적당한 CPU 스레드 (필요 시 조정)
#SBATCH -o ./outputs/%x_%j.out
#SBATCH -e ./outputs/%x_%j.err

source ~/4l/env.sh             

srun python3 train.py

