import torch
import torch.nn as nn

import pennylane as qml

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=98, patch_size=7, in_channels=3, embed_dim=16):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        # Calculate number of patches
        self.n_patches = (image_size // patch_size) ** 2
        # Conv2d projects each patch to an embed_dim vector
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x shape: (batch, in_channels, image_size, image_size)
        x = self.proj(x)                   # (batch, embed_dim, image_size/patch, image_size/patch)
        x = x.flatten(2)                   # flatten spatial dimensions -> (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)              # -> (batch, n_patches, embed_dim)
        return x

import math
#class PositionalEncoder(nn.Module):
#    def __init__(self, embed_dim, max_seq_len=512):
#        super().__init__()
#        self.embed_dim = embed_dim
#        # Initialize positional encoding matrix (max_seq_len x embed_dim)
#        pe = torch.zeros(max_seq_len, embed_dim)
#        for pos in range(max_seq_len):
#            for i in range(0, embed_dim, 2):
#                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
#                if i + 1 < embed_dim:  # cosine for the next dimension
#                    pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_dim)))
#        pe = pe.unsqueeze(0)  # shape (1, max_seq_len, embed_dim) for broadcasting
#        self.register_buffer('pe', pe)    # store as buffer, not a parameter
#    def forward(self, x):
#        # x shape: (batch, seq_len, embed_dim)
#        x = x * math.sqrt(self.embed_dim)
#        seq_len = x.size(1)
#        # Add positional encoding up to the sequence length
#        x = x + self.pe[:, :seq_len]
#        return x

class AddClsPos(nn.Module):
    """
    입력 토큰에 [CLS] 토큰을 붙이고, 위치 임베딩을 더한다.
    """
    def __init__(self, n_patches=N_PATCH, embed_dim=D_MODEL):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        cls = self.cls.expand(B, -1, -1)          # [B,1,D]
        x = torch.cat([cls, x], dim=1)            # [B,N+1,D]
        x = x + self.pos[:, :N+1, :]              # [B,N+1,D]
        return x

import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, mask=None, use_bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"Embedding dim ({embed_dim}) must be divisible by number of heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # dimension per head
        # These will be defined in subclasses (as linear or quantum layers)
        self.q_linear = None
        self.k_linear = None
        self.v_linear = None
        self.combine_heads = None
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # to optionally store attention weights

    def separate_heads(self, x):
        # Reshape x from (batch, seq_len, embed_dim) to (batch, num_heads, seq_len, d_k)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # -> (batch, num_heads, seq_len, d_k)

    def attention(self, Q, K, V, mask=None):
        # Scaled dot-product attention:contentReference[oaicite:28]{index=28}
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask shape: (batch, seq_len), we expand it for broadcasting
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)              # dropout on attention weights
        context = torch.matmul(attn, V)        # result shape: (batch, num_heads, seq_len, d_k)
        return context, attn

    def downstream(self, Q, K, V, batch_size, mask=None):
        # Compute attention and reshape output back to [batch, seq_len, embed_dim]
        context, self.attn_weights = self.attention(Q, K, V, mask)
        # Concatenate heads: transpose and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return context

    def forward(self, x, mask=None):
        raise NotImplementedError("Base class - use a concrete implementation")

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim, num_heads, dropout=0.1, mask=None, use_bias=False):
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        # Linear projection layers for Q, K, V and output:contentReference[oaicite:29]{index=29}
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        # Project input to Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        # Compute attention using base class method
        context = self.downstream(
            self.separate_heads(Q),
            self.separate_heads(K),
            self.separate_heads(V),
            batch_size, mask
        )
        # Final linear to combine heads:contentReference[oaicite:30]{index=30}
        return self.combine_heads(context)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, embed_dim, num_heads, dropout=0.1, mask=None, use_bias=False,
                 n_qubits=4, n_qlayers=1, q_device="default.qubit"):
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        # Quantum setup: embed_dim must match n_qubits:contentReference[oaicite:31]{index=31}
        assert n_qubits == embed_dim, \
            f"n_qubits ({n_qubits}) must equal embed_dim ({embed_dim}) for quantum attention."
        # Initialize quantum device
        if 'qulacs' in q_device:
            self.dev = qml.device(q_device, wires=n_qubits, gpu=True)
        elif 'braket' in q_device:
            self.dev = qml.device(q_device, wires=n_qubits, parallel=True)
        else:
            self.dev = qml.device(q_device, wires=n_qubits)
        # Define quantum circuit for linear layer (angle embedding + entanglers):contentReference[oaicite:32]{index=32}
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.qlayer = qml.QNode(circuit, self.dev, interface="torch")
        # Define trainable weight shapes for the quantum layer (for n_qlayers of entanglers)
        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        # Construct quantum layers acting like linear projections:contentReference[oaicite:33]{index=33}
        self.q_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        self.k_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        self.v_linear = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
        self.combine_heads = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        # Apply the quantum "linear" layer to each time step for Q, K, V
        Q_seq = [self.q_linear(x[:, t, :]) for t in range(seq_len)]
        K_seq = [self.k_linear(x[:, t, :]) for t in range(seq_len)]
        V_seq = [self.v_linear(x[:, t, :]) for t in range(seq_len)]
        # Stack lists into tensors of shape (batch, seq_len, embed_dim)
        Q = torch.stack(Q_seq, dim=1)
        K = torch.stack(K_seq, dim=1)
        V = torch.stack(V_seq, dim=1)
        # Compute attention on these quantum-projected Q, K, V
        context = self.downstream(
            self.separate_heads(Q),
            self.separate_heads(K),
            self.separate_heads(V),
            batch_size, mask
        )
        # Apply the output quantum linear to each time step of context
        out_seq = [self.combine_heads(context[:, t, :]) for t in range(context.size(1))]
        output = torch.stack(out_seq, dim=1)  # (batch, seq_len, embed_dim)
        return output

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        # Two linear layers: input -> hidden (ffn_dim), and hidden -> output (embed_dim)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        raise NotImplementedError("Base FFN class")

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        x = F.relu(self.linear1(x))   # linear + ReLU
        x = self.dropout(x)           # dropout on hidden layer
        x = self.linear2(x)           # linear projection back to embed_dim
        return x

class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim, n_qubits, n_qlayers=1, dropout=0.1, q_device="default.qubit"):
        # Note: here ffn_dim is effectively n_qubits (size of quantum hidden layer)
        super().__init__(embed_dim, ffn_dim=n_qubits, dropout=dropout)
        # Set up quantum device for FFN (similar to attention)
        if 'qulacs' in q_device:
            self.dev = qml.device(q_device, wires=n_qubits, gpu=True)
        elif 'braket' in q_device:
            self.dev = qml.device(q_device, wires=n_qubits, parallel=True)
        else:
            self.dev = qml.device(q_device, wires=n_qubits)
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.qlayer = qml.QNode(circuit, self.dev, interface="torch")
        self.weight_shapes = {"weights": (n_qlayers, n_qubits)}
        # Quantum layer for the feed-forward "activation":contentReference[oaicite:37]{index=37}
        self.vqc = qml.qnn.TorchLayer(self.qlayer, self.weight_shapes)
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()
        x = self.linear1(x)  # project down to n_qubits
        # Apply quantum layer to each token position:contentReference[oaicite:38]{index=38}
        out_seq = [self.vqc(x[:, t, :]) for t in range(seq_len)]
        # Stack back to (batch, seq_len, n_qubits)
        x_q = torch.stack(out_seq, dim=1)
        # Project back to embed_dim
        x = self.linear2(x_q)
        return x


class ViTBlockBase(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # LayerNorm and dropout for post-attention and post-FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # To be set in subclasses:
        self.attn = None
        self.ffn = None
    def forward(self, x):
        # Self-attention sub-layer with residual connection:contentReference[oaicite:40]{index=40}
        attn_out = self.attn(x)
        x = x + attn_out                # add skip connection
        x = self.norm1(x)
        x = self.dropout1(x)
        # Feed-forward sub-layer with residual connection
        ff_out = self.ffn(x)
        x = x + ff_out                  # add skip connection
        x = self.norm2(x)
        x = self.dropout2(x)
        return x

class ViTBlockClassical(ViTBlockBase):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__(embed_dim, num_heads, ffn_dim, dropout)
        # Classical attention and FFN
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout=dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout=dropout)

class ViTBlockQuantum(ViTBlockBase):
    def __init__(self, embed_dim, num_heads, ffn_dim,
                 n_qubits_transformer, n_qubits_ffn,
                 n_qlayers=1, dropout=0.1, q_device="default.qubit"):
        super().__init__(embed_dim, num_heads, ffn_dim, dropout)
        # Quantum multi-head attention (embed_dim must == n_qubits_transformer)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout=dropout,
                                              n_qubits=n_qubits_transformer,
                                              n_qlayers=n_qlayers, q_device=q_device)
        # Quantum or classical FFN depending on n_qubits_ffn
        if n_qubits_ffn and n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, n_qubits=n_qubits_ffn,
                                          n_qlayers=n_qlayers, dropout=dropout, q_device=q_device)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout=dropout)

class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=98, patch_size=7, in_channels=3,
                 embed_dim=16, num_heads=2, num_blocks=2, num_classes=10,
                 ffn_dim=32,
                 n_qubits_transformer=0, n_qubits_ffn=0, n_qlayers=1,
                 dropout=0.1, q_device="default.qubit"):
        super().__init__()
        # Embedding layers
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        n_patches = (image_size / patch_size) ** 2
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.embed = AddClsPos(n_patches, embed_dim)
        # Configure transformer blocks (quantum or classical)
        print(f"++ There will be {num_blocks} transformer blocks.")
        if n_qubits_transformer and n_qubits_transformer > 0:
            print(f"++ Using quantum attention with {n_qubits_transformer} qubits and {n_qlayers} q-layers per block.")
            if n_qubits_ffn and n_qubits_ffn > 0:
                print(f"++ Using quantum feed-forward network with {n_qubits_ffn} qubits.")
            else:
                print("++ Using classical feed-forward network.")
            print(f"++ Quantum device: {q_device}")
            # For quantum mode, ensure dimensions match
            assert embed_dim == n_qubits_transformer, "embed_dim must equal n_qubits_transformer in quantum mode"
            blocks = [
                ViTBlockQuantum(embed_dim, num_heads, ffn_dim,
                                n_qubits_transformer=n_qubits_transformer,
                                n_qubits_ffn=n_qubits_ffn,
                                n_qlayers=n_qlayers, dropout=dropout, q_device=q_device)
                for _ in range(num_blocks)
            ]#:contentReference[oaicite:48]{index=48}
        else:
            blocks = [
                ViTBlockClassical(embed_dim, num_heads, ffn_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]#:contentReference[oaicite:49]{index=49}
        self.transformers = nn.Sequential(*blocks)
        # Classification head (multi-class or binary)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: (batch, in_channels, image_size, image_size)
        x = self.patch_embed(x)        # -> (batch, n_patches, embed_dim)
        x = self.embed(x)        # add positional encoding:contentReference[oaicite:50]{index=50}
        x = self.transformers(x)       # transformer encoder blocks
        # Global average pooling over patch dimension
        x = x.mean(dim=1)              # (batch, embed_dim)
        x = self.dropout(x)
        logits = self.classifier(x)    # (batch, num_classes) or (batch, 1)
        return logits

