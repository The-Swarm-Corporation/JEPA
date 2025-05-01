import math
import random
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor
from torchvision import transforms
import numpy as np


class MaskGenerator:
    """
    Mask generator for the I-JEPA model.
    
    This class generates context and target masks according to the I-JEPA paper's
    masking strategy.
    
    Args:
        grid_size: Size of the grid (e.g., 14 for a 224x16 image).
        num_target_blocks: Number of target blocks to generate.
        target_scale_range: Range of scales for target blocks.
        target_aspect_ratio_range: Range of aspect ratios for target blocks.
        context_scale_range: Range of scales for context block.
    """
    def __init__(
        self,
        grid_size: int,
        num_target_blocks: int = 4,
        target_scale_range: Tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio_range: Tuple[float, float] = (0.75, 1.5),
        context_scale_range: Tuple[float, float] = (0.85, 1.0)
    ):
        self.grid_size = grid_size
        self.num_target_blocks = num_target_blocks
        self.target_scale_range = target_scale_range
        self.target_aspect_ratio_range = target_aspect_ratio_range
        self.context_scale_range = context_scale_range
        
        logger.info(f"Initialized mask generator with grid size {grid_size}x{grid_size}")
        logger.info(f"Target blocks: {num_target_blocks}")
        logger.info(f"Target scale range: {target_scale_range}")
        logger.info(f"Target aspect ratio range: {target_aspect_ratio_range}")
        logger.info(f"Context scale range: {context_scale_range}")
    
    def _generate_block_mask(
        self, 
        scale_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
        is_square: bool = False
    ) -> Tuple[List[int], List[int]]:
        """
        Generate indices for a single block mask.
        
        Args:
            scale_range: Range of scales for the block.
            aspect_ratio_range: Range of aspect ratios for the block.
            is_square: If True, generate a square block.
            
        Returns:
            Tuple containing:
                - List of h indices.
                - List of w indices.
        """
        # Number of patches
        N = self.grid_size ** 2
        
        # Sample random scale and aspect ratio
        scale = random.uniform(*scale_range)
        
        if is_square:
            aspect_ratio = 1.0
        else:
            aspect_ratio = random.uniform(*aspect_ratio_range)
        
        # Calculate block height and width
        area = scale * N
        h = int(math.sqrt(area / aspect_ratio))
        w = int(math.sqrt(area * aspect_ratio))
        
        # Ensure h and w are at least 1 and at most grid_size
        h = max(1, min(h, self.grid_size))
        w = max(1, min(w, self.grid_size))
        
        # Sample random top-left corner
        top = random.randint(0, self.grid_size - h)
        left = random.randint(0, self.grid_size - w)
        
        # Generate indices
        h_indices = list(range(top, top + h))
        w_indices = list(range(left, left + w))
        
        return h_indices, w_indices
    
    def generate_masks(
        self, 
        batch_size: int,
        device: torch.device
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Generate random context and target masks for a batch of images.
        
        Args:
            batch_size: Number of images in batch.
            device: Device to create masks on.
            
        Returns:
            Tuple containing:
                - context_mask: Boolean mask for context blocks. Shape: (B, N)
                - target_masks: List of boolean masks for target blocks. Each mask shape: (B, N)
        """
        # Number of patches
        N = self.grid_size ** 2
        
        # Initialize masks
        context_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        target_masks = [torch.zeros(batch_size, N, dtype=torch.bool, device=device) 
                        for _ in range(self.num_target_blocks)]
        
        # Generate masks for each image in batch
        for b in range(batch_size):
            # Sample target blocks
            for i in range(self.num_target_blocks):
                h_indices, w_indices = self._generate_block_mask(
                    self.target_scale_range, 
                    self.target_aspect_ratio_range
                )
                
                # Set target mask
                for h in h_indices:
                    for w in w_indices:
                        idx = h * self.grid_size + w
                        target_masks[i][b, idx] = True
            
            # Sample context block (square)
            h_indices, w_indices = self._generate_block_mask(
                self.context_scale_range, 
                (1.0, 1.0),
                is_square=True
            )
            
            # Set context mask
            for h in h_indices:
                for w in w_indices:
                    idx = h * self.grid_size + w
                    # Only set to True if not in any target block
                    if not any(target_mask[b, idx] for target_mask in target_masks):
                        context_mask[b, idx] = True
        
        return context_mask, target_masks
    
    def generate_mask_indices(
        self,
        batch_size: int
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[List[Tuple[int, int]]]]]:
        """
        Generate indices for context and target blocks.
        
        This is useful for visualization purposes.
        
        Args:
            batch_size: Number of images in batch.
            
        Returns:
            Tuple containing:
                - context_indices: List of context indices for each image in batch.
                  Each element is a list of (h, w) indices.
                - target_indices: List of target indices for each image in batch and each target block.
                  Shape: (B, num_target_blocks, variable_length)
        """
        # Initialize indices
        context_indices = [[] for _ in range(batch_size)]
        target_indices = [[[] for _ in range(self.num_target_blocks)] for _ in range(batch_size)]
        
        # Generate indices for each image in batch
        for b in range(batch_size):
            # Sample target blocks
            for i in range(self.num_target_blocks):
                h_indices, w_indices = self._generate_block_mask(
                    self.target_scale_range, 
                    self.target_aspect_ratio_range
                )
                
                # Store target indices
                for h in h_indices:
                    for w in w_indices:
                        target_indices[b][i].append((h, w))
            
            # Sample context block (square)
            h_indices, w_indices = self._generate_block_mask(
                self.context_scale_range, 
                (1.0, 1.0),
                is_square=True
            )
            
            # Store context indices
            for h in h_indices:
                for w in w_indices:
                    # Check if this index is in any target block
                    if not any((h, w) in target_block for target_block in target_indices[b]):
                        context_indices[b].append((h, w))
        
        return context_indices, target_indices


def visualize_masks(
    image: Tensor,
    context_indices: List[Tuple[int, int]],
    target_indices: List[List[Tuple[int, int]]],
    patch_size: int,
    save_path: str = None
):
    """
    Visualize context and target masks on an image.
    
    Args:
        image: Input image. Shape: (C, H, W)
        context_indices: List of (h, w) indices for context block.
        target_indices: List of lists of (h, w) indices for target blocks.
        patch_size: Size of each patch.
        save_path: If provided, save the visualization to this path.
    """
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Transpose to (H, W, C) for plotting
    if image.shape[0] == 3:
        image = rearrange(image, 'c h w -> h w c')
    
    # Ensure image is in range [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Plot context block with green border
    for h, w in context_indices:
        rect = plt.Rectangle(
            (w * patch_size, h * patch_size),
            patch_size, patch_size,
            linewidth=1, edgecolor='green', facecolor='none'
        )
        plt.gca().add_patch(rect)
    
    # Plot target blocks with different colors
    colors = ['red', 'blue', 'yellow', 'purple']
    for i, target_block in enumerate(target_indices):
        color = colors[i % len(colors)]
        for h, w in target_block:
            rect = plt.Rectangle(
                (w * patch_size, h * patch_size),
                patch_size, patch_size,
                linewidth=1, edgecolor=color, facecolor='none'
            )
            plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()


def create_masked_image(
    image: Tensor,
    context_indices: List[Tuple[int, int]],
    target_indices: List[List[Tuple[int, int]]],
    patch_size: int
) -> Tuple[Tensor, Tensor]:
    """
    Create masked versions of an image showing only context and target regions.
    
    Args:
        image: Input image. Shape: (C, H, W)
        context_indices: List of (h, w) indices for context block.
        target_indices: List of lists of (h, w) indices for target blocks.
        patch_size: Size of each patch.
        
    Returns:
        Tuple containing:
            - context_image: Image with only context region visible.
            - target_image: Image with only target regions visible.
    """
    # Get image dimensions
    C, H, W = image.shape
    grid_size = H // patch_size
    
    # Create masks
    context_mask = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    for h, w in context_indices:
        context_mask[h, w] = True
    
    target_mask = torch.zeros((grid_size, grid_size), dtype=torch.bool)
    for target_block in target_indices:
        for h, w in target_block:
            target_mask[h, w] = True
    
    # Upscale masks to image size
    context_mask = context_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    target_mask = target_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
    
    # Create masked images
    context_image = image.clone()
    context_image[:, ~context_mask] = 0.0
    
    target_image = image.clone()
    target_image[:, ~target_mask] = 0.0
    
    return context_image, target_image


def preprocess_image(
    image_path: str,
    size: int = 224
) -> Tensor:
    """
    Preprocess an image for I-JEPA model.
    
    Args:
        image_path: Path to image file.
        size: Size to resize image to.
        
    Returns:
        Tensor: Preprocessed image. Shape: (1, C, H, W)
    """
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = transform(image_path)
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    return image


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Create 2D sine-cosine positional embedding.
    
    Args:
        embed_dim: Embedding dimension. Must be divisible by 2.
        grid_size: The height and width of the grid (assuming square grid).
        
    Returns:
        np.ndarray: Positional embeddings of shape (grid_size*grid_size, embed_dim)
    """
    grid_h = grid_size
    grid_w = grid_size
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, -1)  # Reshape to (2, grid_size*grid_size)
    
    # Scale the positional encoding
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embeddings from a grid.
    
    Args:
        embed_dim: Embedding dimension. Must be divisible by 2.
        grid: Grid tensor of shape (2, 1, H, W) containing coordinates.
        
    Returns:
        np.ndarray: Positional embeddings
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be divisible by 2")
    
    # Use half of dimensions for each of sin/cos
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Embedding dimension. Must be divisible by 2.
        pos: Position tensor.
        
    Returns:
        np.ndarray: Positional embeddings
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embedding dimension must be divisible by 2")
        
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim // 2 - 1)
    omega = 1.0 / (10000 ** omega)  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_channels: Number of input image channels.
        embed_dim: Embedding dimension.
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Tensor: Output tensor of shape (B, N, D) where N is the number of patches.
        """
        B, C, H, W = x.shape
        # Shape check
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # Shape: (B, D, grid_size, grid_size)
        x = self.proj(x)
        # Shape: (B, N, D) where N = grid_size * grid_size
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention module.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to query, key, value.
        attn_drop: Dropout rate for attention.
        proj_drop: Dropout rate for projection.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False, 
        attn_drop: float = 0., 
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, D).
            
        Returns:
            Tensor: Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        
        # Shape: (B, N, 3*D)
        qkv = self.qkv(x)
        # Shape: (3, B, num_heads, N, head_dim)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        # Shape: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        # Shape: (B, num_heads, N, head_dim)
        x = (attn @ v)
        # Shape: (B, N, D)
        x = rearrange(x, 'b h n d -> b n (h d)')
        
        # Project back to input dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Multi-layer Perceptron module.
    
    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden feature dimension.
        out_features: Output feature dimension.
        drop: Dropout rate.
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        drop: float = 0.
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4., 
        qkv_bias: bool = False, 
        drop: float = 0., 
        attn_drop: float = 0.
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            drop=drop
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tensor: Output tensor.
        """
        # Attention block with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer.
    
    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        use_abs_pos: Whether to use absolute positional embeddings.
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768, 
        depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4., 
        qkv_bias: bool = True, 
        drop_rate: float = 0., 
        attn_drop_rate: float = 0.,
        use_abs_pos: bool = True
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.use_abs_pos = use_abs_pos
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim),
                requires_grad=False
            )
            # Initialize positional embeddings
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, int(self.num_patches**0.5)
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize weights.
        
        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            mask: Boolean mask tensor indicating which patches to keep.
                  Shape: (B, L) where L is the number of patches.
            
        Returns:
            Tensor: Output tensor of shape (B, N, D) where N is the number of patches.
        """
        # Shape: (B, N, D)
        x = self.patch_embed(x)
        
        if mask is not None:
            # Keep only unmasked patches
            B, L, D = x.shape
            # Create a list to store masked patches for each batch
            masked_patches = []
            for b in range(B):
                # Get indices of unmasked patches for this batch
                unmasked_indices = torch.nonzero(mask[b]).squeeze()
                # Select patches for this batch
                batch_patches = x[b, unmasked_indices]
                masked_patches.append(batch_patches)
            
            # Find the maximum number of unmasked patches across all batches
            max_patches = max(p.shape[0] for p in masked_patches)
            
            # Pad and stack the masked patches
            padded_patches = []
            for patches in masked_patches:
                if patches.shape[0] < max_patches:
                    # Pad with zeros
                    padding = torch.zeros(max_patches - patches.shape[0], D, 
                                        dtype=patches.dtype, device=patches.device)
                    patches = torch.cat([patches, padding], dim=0)
                padded_patches.append(patches)
            
            # Stack all batches
            x = torch.stack(padded_patches, dim=0)
        
        # Add positional embedding if enabled
        if self.use_abs_pos:
            # Only add positional embedding to unmasked patches
            if mask is not None:
                # Get the positional embeddings for unmasked patches
                pos_embed = self.pos_embed.expand(B, -1, -1)
                # Create a list to store masked positional embeddings
                masked_pos_embed = []
                for b in range(B):
                    unmasked_indices = torch.nonzero(mask[b]).squeeze()
                    batch_pos_embed = pos_embed[b, unmasked_indices]
                    # Pad to match the number of patches
                    if batch_pos_embed.shape[0] < max_patches:
                        padding = torch.zeros(max_patches - batch_pos_embed.shape[0], D,
                                            dtype=batch_pos_embed.dtype, device=batch_pos_embed.device)
                        batch_pos_embed = torch.cat([batch_pos_embed, padding], dim=0)
                    masked_pos_embed.append(batch_pos_embed)
                pos_embed = torch.stack(masked_pos_embed, dim=0)
                x = x + pos_embed
            else:
                x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        return x


class Predictor(nn.Module):
    """
    Predictor module for I-JEPA. This is a lightweight ViT that predicts
    target block representations from context block representations.
    
    Args:
        in_dim: Input dimension from context encoder.
        pred_dim: Predictor embedding dimension.
        out_dim: Output dimension (should match target encoder's output dimension).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(
        self, 
        in_dim: int, 
        pred_dim: int = 384, 
        out_dim: int = 768, 
        depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.
    ):
        super().__init__()
        
        # Input projection
        self.in_proj = nn.Linear(in_dim, pred_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        
        # Positional embedding for mask tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, pred_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=pred_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=True, 
                drop=0., 
                attn_drop=0.
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(pred_dim, out_dim)
        
        # Normalization layers
        self.norm_in = nn.LayerNorm(pred_dim)
        self.norm_out = nn.LayerNorm(out_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        nn.init.normal_(self.mask_token, std=0.02)
    
    def _init_weights(self, m: nn.Module) -> None:
        """
        Initialize weights.
        
        Args:
            m: Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, context: Tensor, pos_ids: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            context: Context block representation from context encoder.
                    Shape: (B, Nc, D) where Nc is the number of patches in context.
            pos_ids: Position IDs for the target block.
                    Shape: (B, Nt, 2) where Nt is the number of patches in target.
                    Each position ID is (h, w) in the original grid.
            
        Returns:
            Tensor: Predicted target block representation.
                   Shape: (B, Nt, D) where Nt is the number of patches in target.
        """
        B, Nc, D = context.shape
        Nt = pos_ids.shape[1]
        
        # Project context to predictor dimension
        # Shape: (B, Nc, pred_dim)
        x_ctx = self.in_proj(context)
        
        # Create mask tokens for target
        # Shape: (B, Nt, pred_dim)
        x_mask = self.mask_token.expand(B, Nt, -1)
        
        # Create positional embeddings for target positions
        # We'll convert the 2D positions to 1D grid positions and use them
        # to index into a learned positional embedding
        grid_size = int(math.sqrt(Nc + Nt))  # Assuming square grid
        
        # Convert 2D positions to 1D grid indices
        # pos_ids shape: (B, Nt, 2) -> (B, Nt)
        grid_indices = pos_ids[:, :, 0] * grid_size + pos_ids[:, :, 1]
        
        # Create positional embeddings
        # Shape: (grid_size*grid_size, pred_dim)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        pos_embed = torch.from_numpy(pos_embed).float().to(pos_ids.device)
        
        # Ensure grid indices are within bounds
        grid_indices = torch.clamp(grid_indices, 0, pos_embed.shape[0] - 1)
        
        # Get positional embeddings for target positions
        # Shape: (B, Nt, pred_dim)
        target_pos_embed = pos_embed[grid_indices.view(-1)].view(B, Nt, -1)
        
        # Add positional embeddings to mask tokens
        x_mask = x_mask + target_pos_embed
        
        # Concatenate context and target representations
        # Shape: (B, Nc+Nt, pred_dim)
        x = torch.cat([x_ctx, x_mask], dim=1)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract target representations (the last Nt elements)
        # Shape: (B, Nt, pred_dim)
        x = x[:, -Nt:]
        
        # Project to output dimension
        # Shape: (B, Nt, out_dim)
        x = self.out_proj(x)
        x = self.norm_out(x)
        
        return x


class IJEPA(nn.Module):
    """
    Image-based Joint-Embedding Predictive Architecture (I-JEPA).
    
    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension for the encoder.
        context_encoder_depth: Depth of the context encoder.
        context_encoder_num_heads: Number of attention heads in context encoder.
        predictor_depth: Depth of the predictor.
        predictor_dim: Embedding dimension for the predictor.
        momentum: Momentum for updating target encoder's weights.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        context_encoder_depth: int = 12,
        context_encoder_num_heads: int = 12,
        predictor_depth: int = 12,
        predictor_dim: int = 384,
        momentum: float = 0.996
    ):
        super().__init__()
        
        # Initialize context encoder
        self.context_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=context_encoder_depth,
            num_heads=context_encoder_num_heads,
            use_abs_pos=True
        )
        
        # Initialize target encoder with same architecture as context encoder
        # The target encoder's weights will be updated via EMA of context encoder
        self.target_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=context_encoder_depth,
            num_heads=context_encoder_num_heads,
            use_abs_pos=True
        )
        
        # Initialize predictor
        self.predictor = Predictor(
            in_dim=embed_dim,
            pred_dim=predictor_dim,
            out_dim=embed_dim,
            depth=predictor_depth,
            num_heads=context_encoder_num_heads
        )
        
        # Configure target encoder as a momentum updated version of context encoder
        for param_q, param_k in zip(self.context_encoder.parameters(), 
                                   self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.momentum = momentum
        self.register_buffer("momentum_schedule", torch.linspace(momentum, 1.0, 1000))
        self.current_step = 0
        
        # Grid size based on image and patch size
        self.grid_size = img_size // patch_size
        self.patch_size = patch_size
        
        logger.info(f"Initialized I-JEPA model with grid size {self.grid_size}x{self.grid_size}")
    
    def _update_target_encoder(self) -> None:
        """
        Update target encoder's weights with exponential moving average of context encoder.
        """
        # Get current momentum value
        m = self.momentum_schedule[min(self.current_step, len(self.momentum_schedule) - 1)]
        
        # Update target encoder's weights
        for param_q, param_k in zip(self.context_encoder.parameters(), 
                                   self.target_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        
        self.current_step += 1
    
    def _prepare_images(
        self, 
        images: Tensor, 
        context_mask: Tensor, 
        target_masks: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Prepare images for forward pass by applying masks.
        
        Args:
            images: Input images. Shape: (B, C, H, W)
            context_mask: Boolean mask for context blocks. Shape: (B, N) where N is number of patches.
            target_masks: List of B boolean masks for target blocks. Each mask shape: (B, N)
            
        Returns:
            Tuple containing:
                - Masked context images: Shape (B, C, H, W) with masked regions zeroed out
                - Context mask: Shape (B, N)
                - List of target position IDs: Each shape (B, Nt, 2) where Nt is number of patches in target
                - List of target masks: Each shape (B, N)
        """
        B, C, H, W = images.shape
        
        # Create position IDs for target patches
        target_pos_ids = []
        for target_mask in target_masks:
            # Convert mask to position IDs
            # Shape: (B, N) -> (B, Nt, 2) where Nt is number of True values in mask
            pos_ids = []
            for b in range(B):
                # Get indices of True values in the mask
                indices = torch.nonzero(target_mask[b]).squeeze()
                
                # Convert 1D indices to 2D grid positions
                h_indices = indices // self.grid_size
                w_indices = indices % self.grid_size
                
                # Stack h and w indices
                # Shape: (Nt, 2)
                b_pos_ids = torch.stack([h_indices, w_indices], dim=1)
                pos_ids.append(b_pos_ids)
            
            # Pad position IDs to have same shape across batch
            max_targets = max(p.shape[0] for p in pos_ids)
            padded_pos_ids = []
            for p in pos_ids:
                if p.shape[0] < max_targets:
                    # Pad with zeros
                    padding = torch.zeros(max_targets - p.shape[0], 2, 
                                         dtype=p.dtype, device=p.device)
                    p = torch.cat([p, padding], dim=0)
                padded_pos_ids.append(p)
            
            # Stack position IDs across batch
            # Shape: (B, Nt, 2)
            target_pos_ids.append(torch.stack(padded_pos_ids))
        
        return images, context_mask, target_pos_ids, target_masks
    
    def forward(
        self, 
        images: Tensor, 
        context_mask: Tensor, 
        target_masks: List[Tensor]
    ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Forward pass.
        
        Args:
            images: Input images. Shape: (B, C, H, W)
            context_mask: Boolean mask for context blocks. Shape: (B, N) where N is number of patches.
            target_masks: List of boolean masks for target blocks. Each mask shape: (B, N)
            
        Returns:
            Dict containing:
                - loss: Total loss
                - target_losses: List of losses for each target block
                - context_representations: Representations from context encoder
                - target_representations: List of target representations from target encoder
                - predicted_representations: List of predicted target representations
        """
        # Prepare images and masks for forward pass
        images, context_mask, target_pos_ids, target_masks = self._prepare_images(
            images, context_mask, target_masks
        )
        
        # Get context representations using context encoder
        # Shape: (B, Nc, D) where Nc is number of patches in context
        context_representations = self.context_encoder(images, context_mask)
        
        # Get full target representations using target encoder (without masking)
        # Shape: (B, N, D) where N is total number of patches
        with torch.no_grad():
            target_representations_full = self.target_encoder(images)
        
        # Extract target representations for each target block
        # Each shape: (B, Nt, D) where Nt is number of patches in target
        target_representations = []
        for target_mask in target_masks:
            # Extract target representations based on mask
            # Shape: (B, Nt, D)
            target_repr = []
            for b in range(images.shape[0]):
                # Get indices of True values in the mask
                indices = torch.nonzero(target_mask[b]).squeeze()
                # Extract representations at those indices
                target_repr.append(target_representations_full[b, indices])
            
            # Stack representations across batch
            # Shape: (B, Nt, D)
            max_targets = max(t.shape[0] for t in target_repr)
            padded_target_repr = []
            for t in target_repr:
                if t.shape[0] < max_targets:
                    # Pad with zeros
                    padding = torch.zeros(max_targets - t.shape[0], t.shape[1], 
                                         dtype=t.dtype, device=t.device)
                    t = torch.cat([t, padding], dim=0)
                padded_target_repr.append(t)
            
            target_representations.append(torch.stack(padded_target_repr))
        
        # Predict target representations for each target block
        # Each shape: (B, Nt, D)
        predicted_representations = []
        for pos_ids in target_pos_ids:
            # Predict target representation from context representation
            # Shape: (B, Nt, D)
            predicted_repr = self.predictor(context_representations, pos_ids)
            predicted_representations.append(predicted_repr)
        
        # Compute losses for each target block
        # Each shape: (1,)
        target_losses = []
        for pred_repr, target_repr in zip(predicted_representations, target_representations):
            # Compute MSE loss between predicted and target representations
            # Shape: (1,)
            loss = F.mse_loss(pred_repr, target_repr)
            target_losses.append(loss)
        
        # Compute total loss (average of target losses)
        # Shape: (1,)
        loss = torch.mean(torch.stack(target_losses))
        
        # Update target encoder's weights with EMA of context encoder
        if self.training:
            self._update_target_encoder()
        
        return {
            "loss": loss,
            "target_losses": target_losses,
            "context_representations": context_representations,
            "target_representations": target_representations,
            "predicted_representations": predicted_representations
        }
    
    def generate_masks(
        self, 
        batch_size: int, 
        device: torch.device
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Generate random context and target masks.
        
        Args:
            batch_size: Number of images in batch.
            device: Device to create masks on.
            
        Returns:
            Tuple containing:
                - context_mask: Boolean mask for context blocks. Shape: (B, N)
                - target_masks: List of boolean masks for target blocks. Each mask shape: (B, N)
        """
        # Number of patches
        N = self.grid_size ** 2
        
        # Number of target blocks
        num_target_blocks = 4
        
        # Initialize masks
        context_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)
        target_masks = [torch.zeros(batch_size, N, dtype=torch.bool, device=device) 
                        for _ in range(num_target_blocks)]
        
        # Generate masks for each image in batch
        for b in range(batch_size):
            # Sample target blocks (4 blocks with scale (0.15, 0.2) and aspect ratio (0.75, 1.5))
            for i in range(num_target_blocks):
                # Sample random scale and aspect ratio
                scale = random.uniform(0.15, 0.2)
                aspect_ratio = random.uniform(0.75, 1.5)
                
                # Calculate block height and width
                area = scale * N
                h = int(math.sqrt(area / aspect_ratio))
                w = int(math.sqrt(area * aspect_ratio))
                
                # Ensure h and w are at least 1
                h = max(1, min(h, self.grid_size))
                w = max(1, min(w, self.grid_size))
                
                # Sample random top-left corner
                top = random.randint(0, self.grid_size - h)
                left = random.randint(0, self.grid_size - w)
                
                # Set target mask
                for i_h in range(top, top + h):
                    for i_w in range(left, left + w):
                        idx = i_h * self.grid_size + i_w
                        target_masks[i][b, idx] = True
            
            # Sample context block (scale (0.85, 1.0) and unit aspect ratio)
            scale = random.uniform(0.85, 1.0)
            
            # Calculate block height and width (assuming square)
            size = int(math.sqrt(scale * N))
            
            # Sample random top-left corner
            top = random.randint(0, self.grid_size - size)
            left = random.randint(0, self.grid_size - size)
            
            # Set context mask
            for i_h in range(top, top + size):
                for i_w in range(left, left + size):
                    idx = i_h * self.grid_size + i_w
                    # Only set to True if not in any target block
                    if not any(target_mask[b, idx] for target_mask in target_masks):
                        context_mask[b, idx] = True
        
        return context_mask, target_masks


def create_ijepa_model(
    model_size: str = 'base', 
    img_size: int = 224,
    patch_size: int = 16,
    predictor_dim: int = 384,
    momentum: float = 0.996
) -> IJEPA:
    """
    Create I-JEPA model with specified configuration.
    
    Args:
        model_size: Model size. One of 'base', 'large', 'huge'.
        img_size: Input image size.
        patch_size: Patch size.
        predictor_dim: Dimension of predictor's internal representations.
        momentum: Momentum for updating target encoder's weights.
        
    Returns:
        IJEPA: Initialized I-JEPA model.
    """
    # Configure model based on size
    if model_size == 'base':
        # ViT-B/16 configuration
        embed_dim = 768
        context_encoder_depth = 12
        context_encoder_num_heads = 12
        predictor_depth = 6
    elif model_size == 'large':
        # ViT-L/16 configuration
        embed_dim = 1024
        context_encoder_depth = 24
        context_encoder_num_heads = 16
        predictor_depth = 12
    elif model_size == 'huge':
        # ViT-H/14 configuration
        embed_dim = 1280
        context_encoder_depth = 32
        context_encoder_num_heads = 16
        predictor_depth = 12
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from 'base', 'large', 'huge'.")
    
    # Create and return model
    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        context_encoder_depth=context_encoder_depth,
        context_encoder_num_heads=context_encoder_num_heads,
        predictor_depth=predictor_depth,
        predictor_dim=predictor_dim,
        momentum=momentum
    )
    
    logger.info(f"Created I-JEPA model with size {model_size}")
    logger.info("Model parameters:")
    logger.info(f"  Image size: {img_size}")
    logger.info(f"  Patch size: {patch_size}")
    logger.info(f"  Embedding dimension: {embed_dim}")
    logger.info(f"  Context encoder depth: {context_encoder_depth}")
    logger.info(f"  Context encoder heads: {context_encoder_num_heads}")
    logger.info(f"  Predictor depth: {predictor_depth}")
    logger.info(f"  Predictor dimension: {predictor_dim}")
    
    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
    
    return model




# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# from loguru import logger

# # Import our I-JEPA implementation
# from i_jepa_model import create_ijepa_model
# from i_jepa_utils import MaskGenerator, visualize_masks


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    batch_size = 4
    img_size = 224
    patch_size = 16
    model_size = "base"  # Choose from "base", "large", "huge"

    # Calculate grid size
    grid_size = img_size // patch_size

    # Step 1: Create the I-JEPA model
    logger.info("Creating I-JEPA model...")
    model = create_ijepa_model(
        model_size=model_size,
        img_size=img_size,
        patch_size=patch_size,
        predictor_dim=384,
        momentum=0.996
    )
    model.to(device)
    model.eval()  # Set to evaluation mode for this example

    # Print model architecture summary
    logger.info(f"Model created: {model_size} variant")
    logger.info(f"Grid size: {grid_size}x{grid_size} (patch size: {patch_size})")


    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dummy data for this example
    # In practice, you would load real images from a dataset
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    logger.info(f"Created dummy input tensor with shape: {dummy_input.shape}")

    # Move the input tensor to the device
    dummy_input = dummy_input.to(device)

    # Step 3: Create the mask generator
    logger.info("Creating mask generator...")
    mask_generator = MaskGenerator(
        grid_size=grid_size,
        num_target_blocks=4,
        target_scale_range=(0.15, 0.2),
        target_aspect_ratio_range=(0.75, 1.5),
        context_scale_range=(0.85, 1.0)
    )

    # Step 4: Generate masks for the batch
    logger.info("Generating masks...")
    context_mask, target_masks = mask_generator.generate_masks(
        batch_size=batch_size,
        device=device
    )

    # Log mask shapes
    logger.info(f"Context mask shape: {context_mask.shape}")
    for i, target_mask in enumerate(target_masks):
        logger.info(f"Target mask {i+1} shape: {target_mask.shape}")

    # Step 5: Perform the forward pass
    logger.info("Performing forward pass...")
    with torch.no_grad():  # No need to track gradients for this example
        outputs = model(dummy_input, context_mask, target_masks)

    # Log the outputs
    logger.info(f"Total loss: {outputs['loss'].item():.4f}")
    for i, loss in enumerate(outputs['target_losses']):
        logger.info(f"Target {i+1} loss: {loss.item():.4f}")

    logger.info(f"Context representations shape: {outputs['context_representations'].shape}")
    for i, target_repr in enumerate(outputs['target_representations']):
        logger.info(f"Target {i+1} representations shape: {target_repr.shape}")
    for i, pred_repr in enumerate(outputs['predicted_representations']):
        logger.info(f"Predicted {i+1} representations shape: {pred_repr.shape}")

if __name__ == "__main__":
    main()
