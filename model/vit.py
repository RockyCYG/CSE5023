import torch
import torch.nn as nn

from model.transformer import MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, Encoder, LayerNormalization

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        # [B, d_model, H', W']
        # [B, d_model, num_patches]
        # [B, num_patches, d_model]
        patches = self.patch_proj(x).flatten(2).transpose(1, 2)
        # [B, 1, d_model]
        class_tokens = self.class_token.expand(B, -1, -1)
        # [B, num_patches + 1, d_model]
        x = torch.cat([class_tokens, patches], dim=1)
        x += self.pos_embed
        x = self.dropout(x)
        return x
    

class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, num_classes, num_layers=6, num_heads=8, dropout=0.1, d_ff=2048):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        # Creating EncoderBlocks
        encoder_blocks = [] # Initial list of empty EncoderBlocks
        for _ in range(num_layers): # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout) # Self-Attention
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
            
            # Combine layers into an EncoderBlock
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block) # Appending EncoderBlock to the list of EncoderBlocks
        
        self.encoder = Encoder(nn.ModuleList(encoder_blocks))
        self.norm = LayerNormalization()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        return x
    