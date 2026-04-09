import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# ---------------------------------
# EMA (Exponential Moving Average) Class
# ---------------------------------
class EMA:
    def __init__(self, model: nn.Module, beta: float = 0.99):
        self.beta = beta
        self.step = 0
        
        # Create a copy of the model for EMA
        self.ema_model = copy.deepcopy(model)

        # Freeze the EMA model parameters
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
            
    def update(self, model: nn.Module):
        """
        Update the EMA model parameters using the current model parameters.
        This should be called after each training step.
        """
        
        self.step += 1
        
        for current_param , ema_param in zip(model.parameters(), self.ema_model.parameters()):
            # Update EMA parameter 
            ema_param.data.mul_(self.beta)
            ema_param.data.add_(current_param.data * (1.0 - self.beta))
            
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.ema_model.state_dict())
    
    def save_pretrained(self, path: str):
        torch.save(self.ema_model.state_dict(), path)
        
# ---------------------------------
# SwiGLU Activation Class
# ---------------------------------
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w_out = nn.Linear(hidden_features, out_features)
        
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)
    def forward(self, x):
        # 1. create gate
        gate = F.silu(self.w1(x))  # shape: (batch_size, hidden_features)
        # 2. apply gate to second linear transformation
        x = self.w2(x) * gate
        # 3. project to output features
        x = self.w_out(x)
        return x


# ---------------------------------
# Time Embedding Class for Diffusion Step Information
# ---------------------------------
class TimestepEmbedder(nn.Module):
    """
    Standard sinusoidal timestep embedding followed by an MLP.
    """
    def __init__(self, freq_dim: int, embed_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        
    @staticmethod
    def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Generates sinusoidal embeddings for the given timesteps.
        This is a common technique in diffusion models to encode the timestep information.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]   # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(t, self.freq_dim)
        return self.mlp(x) 

# ---------------------------------
# Efficient Attention using SDPA (Flash Attention)
# ---------------------------------
class EfficientAttention(nn.Module):
    """
    A memory-efficient multi-head attention implementation that 
    leverages torch.nn.functional.scaled_dot_product_attention.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        # 1. Linear projection -> (B, N, 3, H, D) -> (3, B, H, N, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Scaled Dot Product Attention (triggers Flash Attention if available)
        attn_out = F.scaled_dot_product_attention(q, k, v) # (B, H, N, D)

        # 3. Reshape and project back
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


# ---------------------------------
# Transformer Block Class
# ---------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # --- Original Implementation (Commented out) ---
        # self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # --- New Efficient Implementation ---
        self.attn = EfficientAttention(embed_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = SwiGLU(embed_dim, hidden_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. self-attention with residual
        x = x + self.attn(self.norm1(x))
        # 2. MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------------
# Patch Embedding for Diffusion Transformer
# ---------------------------------
class PatchEmbed(nn.Module):
    """
    Converts 2D latent images into a 1D sequence of flattened patches.
    patch_size=16 -> 96x96 image gives (96/16)^2 = 36 patches per image.
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape : (Batch size, channels, H, W)
        x = self.proj(x)                  # Output: (N, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # Output: (N, seq_len, embed_dim) where seq_len = (H/patch_size)*(W/patch_size)
        return x

# ---------------------------------
# Diffusion Policy Class
# ---------------------------------
class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int = 2,
        state_dim: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_blocks: int = 4,
        # --- Image input parameters ---
        use_image: bool = False,      # Enable image conditioning
        in_channels: int = 3,         # RGB channels
        image_size: int = 128,        # Resize target (H = W)
        patch_size: int = 8,          # 縮小 Patch 從 16 -> 8 以提升精度
        use_checkpoint: bool = False  # Enable Gradient Checkpointing
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.use_image = use_image
        self.use_checkpoint = use_checkpoint
        
        # 1. Observation (state) embedding
        self.obs_embedder = nn.Linear(state_dim, embed_dim)
        # 2. Time embedding for diffusion step information
        self.time_embedder = TimestepEmbedder(freq_dim=embed_dim, embed_dim=embed_dim)
        # 3. Noised action embedding
        self.action_embedder = nn.Linear(action_dim, embed_dim)
        
        # 4. Image patch embedding (only when use_image=True)
        if use_image:
            self.patch_embedder = PatchEmbed(
                in_channels=in_channels,
                embed_dim=embed_dim,
                patch_size=patch_size
            )
            # Number of patches per single image frame
            patches_per_side = image_size // patch_size          
            self.num_patches_per_image = patches_per_side ** 2   
            print(f"[DiffusionPolicy] Image mode: {image_size}x{image_size}, "
                  f"patch_size={patch_size}, {self.num_patches_per_image} patches/image")
        
        # 5. Transformer blocks for processing the combined token sequence
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)
        ])
        
        # 6. Final projection: maps last action tokens back to action dimension
        self.final_proj = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        observations: torch.Tensor,      # (B, obs_horizon, state_dim)
        diffusion_steps: torch.Tensor,   # (B,)
        noised_actions: torch.Tensor,    # (B, pred_horizon, action_dim)
        images: torch.Tensor = None,     # (B, obs_horizon, C, H, W) optional
    ) -> torch.Tensor:
        
        # 1. Embed state observations: (B, obs_horizon, embed_dim)
        obs_emb = self.obs_embedder(observations)
        
        # 2. Embed diffusion timestep: (B, 1, embed_dim)
        time_emb = self.time_embedder(diffusion_steps).reshape(-1, 1, self.embed_dim)
        
        # 3. Embed noised actions: (B, pred_horizon, embed_dim)
        action_emb = self.action_embedder(noised_actions)
        
        # 4. Build token sequence: start with [obs | time | action]
        #    Image tokens are prepended when available
        tokens = [obs_emb, time_emb, action_emb]
        
        # 5. Patchify image and prepend as conditioning tokens
        if self.use_image and images is not None:
            # images: (B, T_obs, C, H, W)
            B, T_obs, C, H, W = images.shape
            
            # Flatten B, T_obs to process all frames/views through patch embedder
            imgs_flat = images.reshape(B * T_obs, C, H, W)
            img_patches = self.patch_embedder(imgs_flat)  # (B*T_obs, num_patches, embed_dim)
            
            # Restore and concatenate all camera patches
            # (B, T_obs * num_patches, embed_dim)
            img_patches = img_patches.reshape(B, T_obs * self.num_patches_per_image, self.embed_dim)
            
            # Prepend image tokens
            tokens = [img_patches] + tokens
        
        # 6. Concatenate all tokens along sequence dimension
        x = torch.cat(tokens, dim=1)  # (B, total_seq_len, embed_dim)

        # 7. Add sinusoidal positional encoding
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = TimestepEmbedder.sinusoidal(positions, self.embed_dim)
        x = x + pos_emb
        
        # 8. Pass through Transformer blocks (with optional Gradient Checkpointing)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Need to wrap block.forward to call it with checkpoint
                import torch.utils.checkpoint as cp
                x = cp.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # 9. Extract only the action tokens (always at the tail of the sequence)
        action_seq_len = noised_actions.shape[1]
        action_out = x[:, -action_seq_len:, :]  # (B, pred_horizon, embed_dim)
        output = self.final_proj(action_out)     # (B, pred_horizon, action_dim)
        return output
