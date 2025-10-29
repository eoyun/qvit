# train_ddp.py  (원본의 데이터셋/모델 사용 유지 + DDP 안전 개선)
import os
import argparse
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc

# -------------------- args --------------------
p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--weight_decay", type=float, default=1e-4)
p.add_argument("--num_workers", type=int, default=2)
p.add_argument("--image_size", type=int, default=98)
p.add_argument("--patch_size", type=int, default=14)
p.add_argument("--embed_dim", type=int, default=4)  # ViT dim
p.add_argument("--num_heads", type=int, default=2)
p.add_argument("--num_blocks", type=int, default=2)
p.add_argument("--ffn_dim", type=int, default=4)
p.add_argument("--n_qlayers", type=int, default=1)
p.add_argument("--n_qubits_transformer", type=int, default=4)
p.add_argument("--n_qubits_ffn", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--q_device", type=str, default="lightning.gpu")
p.add_argument("--patience", type=int, default=10)
p.add_argument("--csv_path", type=str, default="rm_invalid.csv")
p.add_argument("--label",type=str,default="",help="label the output name")
args = p.parse_args()

epochs = args.epochs
batch_size = args.batch_size
image_size = args.image_size
patch_size = args.patch_size
embed_dim = args.embed_dim
num_heads = args.num_heads
num_blocks = args.num_blocks
ffn_dim = args.ffn_dim
n_qlayers = args.n_qlayers
n_qubits_transformer = args.n_qubits_transformer
n_qubits_ffn = args.n_qubits_ffn
dropout = args.dropout
q_device = args.q_device
label = args.label
results_dir = f"/pscratch/sd/e/eoyun/4l/results/pytorch/quantum/{epochs}_epochs_{n_qubits_ffn}_qubitOnFFN_{n_qubits_transformer}_qubitOnMHA_{n_qlayers}_qubitLayer_{embed_dim}_embedDimension_{label}"
ckpt_dir = f"/pscratch/sd/e/eoyun/4l/ckpts/pytorch/quantum/{epochs}_epochs_{n_qubits_ffn}_qubitOnFFN_{n_qubits_transformer}_qubitOnMHA_{n_qlayers}_qubitLayer_{embed_dim}_embedDimension_{label}"
patience = args.patience
csv_path = args.csv_path
label_col = "Label"
path_col = "ImagePath"

# -------------------- Dist init --------------------
use_cuda = torch.cuda.is_available()
distributed = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
backend = "nccl" if use_cuda else "gloo"  # CHANGED: GPU 없으면 gloo
if distributed:
    dist.init_process_group(backend=backend, init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if use_cuda:
    torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")

def is_dist():
    return dist.is_available() and dist.is_initialized()
def rank():
    return dist.get_rank() if is_dist() else 0
def world_size():
    return dist.get_world_size() if is_dist() else 1
def barrier():
    if is_dist(): dist.barrier()
def reduce_mean_scalar(x: float) -> float:  # CHANGED: 전역 평균
    if not is_dist(): return float(x)
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= world_size()
    return float(t.item())
def bcast_float(x: float, src: int = 0) -> float:
    if not is_dist(): return float(x)
    t = torch.tensor([x], dtype=torch.float32, device=device)
    dist.broadcast(t, src=src)
    return float(t.item())
def bcast_bool(flag: bool, src: int = 0) -> bool:
    if not is_dist(): return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.broadcast(t, src=src)
    return bool(int(t.item()))
def all_gather_numpy_1d(arr_np: np.ndarray) -> np.ndarray:  # CHANGED: 가변 길이 수집
    if not is_dist(): return arr_np
    if arr_np is None or arr_np.size == 0:
        t = torch.empty(0, dtype=torch.int64, device=device)
    else:
        t = torch.as_tensor(arr_np, device=device).view(-1)
    l = torch.tensor([t.numel()], device=device, dtype=torch.int64)
    ls = [torch.zeros_like(l) for _ in range(world_size())]
    dist.all_gather(ls, l)
    maxlen = int(torch.stack(ls).max().item()) if len(ls) else 0
    pad = torch.zeros(maxlen, dtype=t.dtype, device=device)
    if t.numel() > 0: pad[:t.numel()] = t
    outs = [torch.zeros_like(pad) for _ in range(world_size())]
    dist.all_gather(outs, pad)
    outs_trim = [out[: int(li.item())].clone() for out, li in zip(outs, ls)]
    if len(outs_trim) == 0: return np.array([], dtype=np.int64)
    return torch.cat(outs_trim, dim=0).detach().cpu().numpy()

# -------------------- Data --------------------
df = pd.read_csv(csv_path)
# 필요시 클래스 제한 샘플링은 여기서 조정
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

labels_sorted = sorted(df[label_col].unique().tolist())
label_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}
idx_to_label = {i: lab for lab, i in label_to_idx.items()}
df["LabelIdx"] = df[label_col].map(label_to_idx)
num_classes = len(labels_sorted)

X = df[path_col].values
y = df["LabelIdx"].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_tf = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

train_ds = ImageDataset(X_train, y_train, transform=train_tf)
val_ds   = ImageDataset(X_val,   y_val,   transform=test_tf)
test_ds  = ImageDataset(X_test,  y_test,  transform=test_tf)

train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
val_sampler   = DistributedSampler(val_ds,   shuffle=False) if distributed else None
test_sampler  = DistributedSampler(test_ds,  shuffle=False) if distributed else None

train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=False,
                          num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers>0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, sampler=val_sampler, shuffle=False,
                          num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers>0)
test_loader  = DataLoader(test_ds,  batch_size=batch_size*2, sampler=test_sampler, shuffle=False,
                          num_workers=args.num_workers, pin_memory=use_cuda, persistent_workers=args.num_workers>0)

# -------------------- Model --------------------
from qvit import VisionTransformer  # 원본과 동일

model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    in_channels=3,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    num_classes=num_classes,
    ffn_dim=ffn_dim,
    n_qubits_transformer=n_qubits_transformer,
    n_qubits_ffn=n_qubits_ffn,
    n_qlayers=n_qlayers,
    dropout=dropout,
    q_device=q_device
).to(device)

if distributed:
    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)  # CHANGED
else:
    ddp_model = model

# -------------------- Loss/Opt/Sched --------------------
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt)**self.gamma * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

loss_fn = FocalLoss(gamma=2.0, reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

# -------------------- Paths --------------------
os.makedirs(results_dir, exist_ok=True) if rank()==0 else None
os.makedirs(ckpt_dir, exist_ok=True) if rank()==0 else None
best_model_path = os.path.join(ckpt_dir, f"best_vit_e{epochs}_dim{embed_dim}_ql{n_qlayers}.pth")
metrics_history = {"train_loss":[], "val_loss":[], "train_f1":[], "val_f1":[]}

# -------------------- Train/Eval helpers --------------------
def run_epoch(model, loader, train: bool):
    model.train(mode=train)
    losses = []
    preds_local = []
    labels_local = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(images)
                loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(images)
                loss = loss_fn(logits, labels)
            losses.append(float(loss.item()))

        preds = logits.argmax(dim=1)
        preds_local.append(preds.detach().cpu().numpy())
        labels_local.append(labels.detach().cpu().numpy())

    loss_local = float(np.mean(losses)) if len(losses) else 0.0
    loss_mean = reduce_mean_scalar(loss_local)

    preds_np_local = np.concatenate(preds_local) if preds_local else np.array([], dtype=np.int64)
    labels_np_local = np.concatenate(labels_local) if labels_local else np.array([], dtype=np.int64)
    preds_all = all_gather_numpy_1d(preds_np_local)
    labels_all = all_gather_numpy_1d(labels_np_local)
    if rank() == 0 and labels_all.size > 0:
        f1 = f1_score(labels_all, preds_all, average="macro")
    else:
        f1 = 0.0
    f1 = bcast_float(f1, src=0)
    return loss_mean, f1

# -------------------- Train loop --------------------
best_val_f1 = -1.0
epochs_no_improve = 0

for epoch in range(1, epochs+1):
    if distributed and train_sampler is not None: train_sampler.set_epoch(epoch)
    if distributed and val_sampler   is not None: val_sampler.set_epoch(epoch)

    train_loss, train_f1 = run_epoch(ddp_model, train_loader, train=True)
    val_loss,   val_f1   = run_epoch(ddp_model, val_loader,   train=False)

    if rank() == 0:
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["train_f1"].append(train_f1)
        metrics_history["val_f1"].append(val_f1)
        print(f"Epoch {epoch:03d} | Train {train_loss:.4f}/{train_f1:.4f} | Val {val_loss:.4f}/{val_f1:.4f}")
        scheduler.step(val_f1)

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "label_map": idx_to_label,
                "args": vars(args),
            }, best_model_path)
        else:
            epochs_no_improve += 1

        # plot learning curves
        epochs_range = range(1, len(metrics_history["train_loss"]) + 1)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epochs_range, metrics_history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, metrics_history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(epochs_range, metrics_history["train_f1"], label="Train F1")
        plt.plot(epochs_range, metrics_history["val_f1"], label="Val F1")
        plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("F1"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"training_history_{epoch}.png"))
        plt.close()

    early_stop = bcast_bool(epochs_no_improve >= patience, src=0)
    if early_stop:
        if rank() == 0:
            print(f"Early stopping at epoch {epoch}.")
        break

barrier()

# -------------------- Test --------------------
test_loss, test_f1 = run_epoch(ddp_model, test_loader, train=False)
if rank() == 0:
    print(f"[TEST] Loss {test_loss:.4f} F1 {test_f1:.4f}")

    # ROC 곡선도 rank0에서만
    # 다중 클래스 One-vs-Rest ROC
    y_true = []
    y_score = []
    ddp_model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = ddp_model(images)
                prob = torch.softmax(logits, dim=1)
            y_true.append(labels.detach().cpu().numpy())
            y_score.append(prob.detach().cpu().numpy())
    y_true = np.concatenate(y_true); y_score = np.concatenate(y_score)
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.int32)
        fpr[c], tpr[c], _ = roc_curve(y_bin, y_score[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    plt.figure(figsize=(7,6))
    any_plotted = False
    for c in range(num_classes):
        name = idx_to_label[c]
        if np.isfinite(roc_auc[c]):
            plt.plot(fpr[c], tpr[c], label=f"{name} (AUC={roc_auc[c]:.2f})"); any_plotted = True
    plt.plot([0,1],[0,1],"k--")
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Multi-Class ROC")
    if any_plotted: plt.legend(loc="lower right")
    else: plt.legend([], frameon=False)
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

# -------------------- Teardown --------------------
if is_dist():
    dist.destroy_process_group()

