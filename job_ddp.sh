#!/bin/bash
#SBATCH -J train_ddp_interactive        # Job 이름
#SBATCH -A m4138_g                      # Account
#SBATCH -q regular                 # QoS
#SBATCH -C gpu                          # GPU 노드
#SBATCH -G 4                            # 총 GPU 수
#SBATCH -N 1                            # 노드 수
#SBATCH -t 12:00:00                     # 시간 제한
#SBATCH -o outputs/%x-%j.out               # 로그 저장 경로
#SBATCH -e outputs/%x-%j.err           # 로그 저장 경로

# 환경 설정 (필요시 수정)
source ./env.sh
#
# # torchrun 실행
torchrun \
  --nproc_per_node=4 \
  train_ddp_fix.py \
  --n_qubits_ffn 4 \
  --n_qubits_transformer 0 \
  --ffn_dim 4 \
  --embed_dim 4 \
  --num_quantum_block 2 \
  --n_qlayers 1 \
  --epochs 50 \
  --label 251104_ddp
                  
