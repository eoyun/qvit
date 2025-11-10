#!/bin/bash
#SBATCH -J qvit_sweep
#SBATCH -A m4138_g
#SBATCH -q regular              # shared 금지. full node 사용
#SBATCH -C gpu
#SBATCH -N 1                    # 각 task당 1노드
#SBATCH --ntasks-per-node=1     # 노드당 1개 랭처
#SBATCH -G 4                    # Perlmutter GPU 4개 = full GPU node
#SBATCH --exclusive             # 노드 독점
#SBATCH -t 12:00:00
#SBATCH -o outputs/%x-%j_%a.out
#SBATCH -e outputs/%x-%j_%a.err
#SBATCH --array=0-12             # 조합 개수-1 로 조정

# 성능/안정
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

mkdir -p outputs
# conda 등 환경이 필요하면 여기에 로드
# source ~/.bashrc
# conda activate cml

# (n_qubits_ffn  n_qlayers  num_quantum_block)
COMBOS=(
"0 0 0 0"
"4 0 1 1"
"4 0 2 1"
"0 4 1 1"
"0 4 2 1"
"4 4 1 1"
"4 4 2 1"
"4 0 1 2"
"4 0 2 2"
"0 4 1 2"
"0 4 2 2"
"4 4 1 2"
"4 4 2 2"
)

read -r NQUBITS_FFN NQUBITS_TRANS NQLAYERS NQBLOCKS <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
LABEL="251110_qsweep_q${NQUBITS_FFN}_${NQUBITS_TRANS}_l${NQLAYERS}_b${NQBLOCKS}"

# 각 task는 자기 노드에서 4프로세스만 띄움
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_ddp_fix.py \
  --n_qubits_ffn "${NQUBITS_FFN}" \
  --n_qubits_transformer "${NQUBITS_TRANS}" \
  --ffn_dim 4 \
  --embed_dim 4 \
  --num_quantum_block "${NQBLOCKS}" \
  --n_qlayers "${NQLAYERS}" \
  --epochs 50 \
  --label "${LABEL}"

