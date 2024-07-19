import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

class ModelArgs:
    def __init__(self, d_model=128, n_layer=4, seq_len=96, d_state=16, expand=2, dt_rank='auto',
                 d_conv=4, pad_multiple=8, conv_bias=True, bias=False,
                 num_channels=24, patch_len=16, stride=8, forecast_len=96, sigma=0.5, reduction_ratio=8, verbose=False):
        self.d_model = d_model
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.d_state = d_state
        self.v = verbose
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.pad_multiple = pad_multiple
        self.conv_bias = conv_bias
        self.bias = bias
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_len = forecast_len
        self.sigma = sigma
        self.reduction_ratio = reduction_ratio
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.forecast_len % self.pad_multiple != 0:
            self.forecast_len += (self.pad_multiple - self.forecast_len % self.pad_multiple)

class ChannelMixup(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            B, V, L = x.shape
            perm = torch.randperm(V)
            lambda_ = torch.normal(mean=0, std=self.sigma, size=(V,)).to(x.device)
            x_mixed = x + lambda_.unsqueeze(1) * x[:, perm]
            return x_mixed
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).squeeze(-1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).squeeze(-1))))
        out = self.sigmoid(avg_out + max_out)
        return out.unsqueeze(-1)

class PatchMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([MambaBlock(args) for _ in range(args.n_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

class CMambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_mamba = PatchMamba(args)
        self.channel_attention = ChannelAttention(args.d_model, args.reduction_ratio)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x = self.patch_mamba(x)
        attn = self.channel_attention(x.permute(0, 2, 1))
        x = x * attn.permute(0, 2, 1)
        return self.norm(x)

class CMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.channel_mixup = ChannelMixup(args.sigma)
        self.patch_embedding = nn.Linear(args.patch_len * args.num_channels, args.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, args.num_patches, args.d_model))
        
        self.c_mamba_blocks = nn.ModuleList([CMambaBlock(args) for _ in range(args.n_layer)])
        
        self.norm_f = RMSNorm(args.d_model)
        self.output_layer = nn.Linear(args.d_model * args.num_patches, args.num_channels * args.forecast_len)

    def forward(self, input_ids):
        print("input_ids", input_ids.shape) if self.args.v else None
        x = self.channel_mixup(input_ids)
        print("after channel mixup", x.shape) if self.args.v else None
        # Patching
        B, V, L = x.shape
        P = self.args.patch_len
        S = self.args.stride

        # Manual patching
        patches = []
        for i in range(0, L - P + 1, S):
            patch = x[:, :, i:i+P].reshape(B, -1)
            patches.append(patch)
        num_patches = (L - P) // S + 1
        print(f"Calculated number of patches: {num_patches}") if self.args.v else None

        x = torch.stack(patches, dim=1)  # (B, num_patches, V*P)
        print("after patching", x.shape) if self.args.v else None
        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, d_model)
        print("after patch embedding", x.shape) if self.args.v else None
        # Adjust positional encoding
        pos_encoding = self.pos_encoding[:, :x.size(1), :]
        print(f"Positional encoding shape: {pos_encoding.shape}") if self.args.v else None
        # Add positional encoding
        x = x + pos_encoding
        print("after positional encoding", x.shape) if self.args.v else None
        # Apply C-Mamba blocks
        for block in self.c_mamba_blocks:
            x = block(x)
        print("after C-Mamba blocks", x.shape) if self.args.v else None
        x = self.norm_f(x)
        print("after norm_f", x.shape) if self.args.v else None
        # Output layer
        x = x.reshape(x.shape[0], -1)
        print("before output layer", x.shape) if self.args.v else None
        logits = self.output_layer(x)
        print("after output layer", logits.shape) if self.args.v else None
        logits = logits.reshape(-1, self.args.num_channels, self.args.forecast_len)
        print("final logits", logits.shape) if self.args.v else None
        return logits

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


if __name__ == "__main__":
    args = ModelArgs(
        d_model=128,          # Dimension of the model
        n_layer=4,            # Number of C-Mamba blocks
        seq_len=96,           # Length of input sequence (look-back window)
        num_channels=17,      # Number of numerical channels in your data
        patch_len=16,         # Length of each patch
        stride=8,             # Stride for patching
        forecast_len=96,      # Number of future time steps to predict
        d_state=16,           # Dimension of SSM state
        expand=2,             # Expansion factor for inner dimension
        dt_rank='auto',       # Rank for delta projection, 'auto' sets it to d_model/16
        d_conv=4,             # Kernel size for temporal convolution
        pad_multiple=8,       # Padding to ensure sequence length is divisible by this
        conv_bias=True,       # Whether to use bias in convolution
        bias=False,           # Whether to use bias in linear layers
        sigma=0.5,            # Standard deviation for channel mixup
        reduction_ratio=4,     # Reduction ratio for channel attention
        verbose=False
    )
    model = CMamba(args)
    print(model)
    # Example input
    x = torch.randn(32, args.num_channels, args.seq_len)
    output = model(x)
    print(output.shape)  # Should be (32, forecast_len)