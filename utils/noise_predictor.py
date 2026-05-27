import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import torchvision.models as models

# ---------------------------------
# EMA (Exponential Moving Average) Class
# ---------------------------------
class EMA:
    def __init__(self, model: nn.Module, beta: float = 0.99):
        self.beta = beta
        self.step = 0
        
        self.ema_model = copy.deepcopy(model)

        for param in self.ema_model.parameters():
            param.requires_grad_(False)
            
    def update(self, model: nn.Module):
        self.step += 1
        for current_param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
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
        gate = F.silu(self.w1(x))
        x = self.w2(x) * gate
        x = self.w_out(x)
        return x


# ---------------------------------
# Time Embedding Class
# ---------------------------------
class TimestepEmbedder(nn.Module):
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(t, self.freq_dim)
        return self.mlp(x) 


# ---------------------------------
# Efficient Attention (With Mask Support)
# ---------------------------------
class EfficientAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch's native scaled dot product attention, passing down the boolean mask
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_out)


# ---------------------------------
# Transformer Block Class
# ---------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientAttention(embed_dim, num_heads)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = SwiGLU(embed_dim, hidden_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------
# Spatial Softmax Block
# ---------------------------------
class SpatialSoftmax(nn.Module):
    """
    Extracts spatial expected coordinates (Keypoints) from feature maps.
    """
    def __init__(self, height: int, width: int, num_channels: int, temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels

        # Create normalized coordinate grids [-1, 1]
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1., 1., height),
            torch.linspace(-1., 1., width),
            indexing='ij'
        )
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # feature: (B, C, H, W)
        B, C, H, W = feature.shape
        assert H == self.height and W == self.width, f"Expected {self.height}x{self.width}, got {H}x{W}"

        # Flatten spatial dimensions: (B, C, H*W)
        feature_flat = feature.reshape(B, C, H * W)
        
        # Apply temperature and softmax to get probabilities
        weights = F.softmax(feature_flat / self.temperature, dim=-1)

        # Calculate expected coordinates
        expected_x = torch.sum(self.pos_x * weights, dim=-1) # (B, C)
        expected_y = torch.sum(self.pos_y * weights, dim=-1) # (B, C)

        # Stack coordinates: (B, C, 2) and flatten to (B, C * 2)
        expected_xy = torch.stack([expected_x, expected_y], dim=-1)
        return expected_xy.reshape(B, C * 2)


# ---------------------------------
# Residual Block Class
# ---------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        # Shortcut branch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------
# Vision Encoder Class
# ---------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, image_size: int = 128, embed_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),  
            ResBlock(32, 64, stride=2),   
            ResBlock(64, 128, stride=2),  
            ResBlock(128, 256, stride=2)  
        )
        
        feat_size = image_size // 16
        num_channels = 256
        
        self.spatial_softmax = SpatialSoftmax(
            height=feat_size, 
            width=feat_size, 
            num_channels=num_channels
        )
        
        self.proj = nn.Sequential(
            nn.Linear(num_channels * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)                     
        keypoints = self.spatial_softmax(features) 
        img_token = self.proj(keypoints)           
        return img_token


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
        use_image: bool = False,
        in_channels: int = 3,         
        image_size: int = 128,        
        use_checkpoint: bool = False  
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.use_image = use_image
        self.use_checkpoint = use_checkpoint
        
        # 1. Input projectors
        self.obs_embedder = nn.Linear(state_dim, embed_dim)
        self.time_embedder = TimestepEmbedder(freq_dim=embed_dim, embed_dim=embed_dim)
        self.action_embedder = nn.Linear(action_dim, embed_dim)
        
        # 2. Learnable modality embeddings to help the Transformer decouple input sources
        self.obs_modality_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.action_modality_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.time_modality_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        if use_image:
            self.img_modality_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 3. Vision Encoder
        if use_image:
            self.vision_encoder = VisionEncoder(
                in_channels=in_channels,
                image_size=image_size,
                embed_dim=embed_dim
            )
            print(f"[DiffusionPolicy] Image mode: {image_size}x{image_size} using SpatialSoftmax")
        
        # 4. Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)
        ])
        
        # 5. Final action projector
        self.final_proj = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        observations: torch.Tensor,      # (B, obs_horizon, state_dim)
        diffusion_steps: torch.Tensor,   # (B,)
        noised_actions: torch.Tensor,    # (B, pred_horizon, action_dim)
        images: torch.Tensor = None,     # (B, obs_horizon, C, H, W) optional
    ) -> torch.Tensor:
        B, T_obs, _ = observations.shape
        T_pred = noised_actions.shape[1]
        device = observations.device
        
        # 1. Project modalities and inject modality embeddings
        obs_emb = self.obs_embedder(observations) + self.obs_modality_emb
        time_emb = self.time_embedder(diffusion_steps).reshape(-1, 1, self.embed_dim) + self.time_modality_emb
        action_emb = self.action_embedder(noised_actions) + self.action_modality_emb
        
        tokens = [obs_emb, time_emb, action_emb]
        
        # 2. Process visual features if active
        if self.use_image and images is not None:
            B_img, T_img, C, H, W = images.shape
            imgs_flat = images.reshape(B_img * T_img, C, H, W)
            
            img_tokens = self.vision_encoder(imgs_flat)  # (B*T_obs, embed_dim)
            img_tokens = img_tokens.reshape(B_img, T_img, self.embed_dim) + self.img_modality_emb
            
            tokens = [img_tokens] + tokens
        
        # 3. Concatenate tokens along sequence dimension
        x = torch.cat(tokens, dim=1)

        # 4. Generate temporally-aligned sinusoidal positional encodings
        # Assign same positional steps [0 ... T_obs-1] to both Image and Obs tokens.
        # Global time token receives index 0, and future actions start after the history window.
        img_positions = torch.arange(T_obs, device=device)
        obs_positions = torch.arange(T_obs, device=device)
        time_position = torch.zeros(1, dtype=torch.long, device=device)
        action_positions = torch.arange(T_obs, T_obs + T_pred, device=device)
        
        if self.use_image and images is not None:
            positions = torch.cat([img_positions, obs_positions, time_position, action_positions], dim=0)
        else:
            positions = torch.cat([obs_positions, time_position, action_positions], dim=0)
            
        pos_emb = TimestepEmbedder.sinusoidal(positions, self.embed_dim)
        x = x + pos_emb
        
        # 5. Create block-causal attention mask to protect history tokens from leakage
        # Conditioning/History tokens occupy the first (L - T_pred) indices.
        L_total = x.shape[1]
        N_hist = L_total - T_pred
        attn_mask = torch.ones((L_total, L_total), dtype=torch.bool, device=device)
        
        # Block attention flowing from condition tokens into future target actions
        attn_mask[:N_hist, N_hist:] = False
        
        # 6. Pass through Transformer blocks with attention masking
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                import torch.utils.checkpoint as cp
                x = cp.checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask)
        
        # 7. Extract future action predictions at the tail of the sequence
        action_out = x[:, -T_pred:, :]
        output = self.final_proj(action_out)
        return output