"""
models/flat_seq.py
------------------
The FLAT counterpart to models/hierarchical_seq.py, following the Serrano &
Smith (2019) "Flat Attention Network" (FLAN): a SINGLE attention layer over a
contextualised sequence, then a linear classifier (S&S: "inspired by the HAN
... the layer that classifies the document representation is linear").

There is NO beat segmentation here -- the whole 12-lead signal is encoded once.
The per-timestep base representation is produced by PatchCNNEncoder, so flat vs.
hierarchical differ in the feature extractor. On top of
that base, the three variants vary contextualisation REACH exactly as S&S do:

  variant   sequence encoder (reach over 100 ms tokens)
  -------   -----------------------------------------------------------------
  "noenc"   1x1 conv (per-timestep, UPSAMPLES 256->512). Reach 0 -- the
            "no encoder".
  "conv"    S&S dual-width conv: concat of a kernel-5 (sees +/-2 neighbours)
            and a kernel-3 (sees +/-1) convolution. Reach = +/-2 timesteps.
  "bilstm"  BiLSTM over timesteps. Reach = whole sequence (the S&S biRNN).

All three emit a common width (seq_dim=512: noenc/conv directly, bilstm via
2*seq_hidden=512).

Input  : beats-free whole signal, x (B, 12, T)   [PTB-XL 500 Hz -> T=5000]
Forward:
    forward(x)                -> logits                  (B, num_classes)
    forward_with_attention(x) -> logits, alphas          alphas: (B, T_red)
        T_red = T / 50 (100 tokens for T=5000).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the EXACT modules from the hierarchical model so the CNN feature
# extractor, the noenc upsample, the BiLSTM, and the attention are
# dentical across the flat and hierarchical families.
from models.hierarchical_seq import (
    AdditiveAttention, SeqNoEnc, SeqBiLSTM, SeqConv
)

PATCH = (10, 5)     # (a, b): kernel = stride of the two convs -> tokens of a*b = 50 samples

class PatchCNNEncoder(nn.Module):
    """Non-overlapping token extractor: (N,12,T) -> (N,T/(a*b),out_channels).
    Conv 1 turns each 10-sample piece into 64 numbers; conv 2 combines 5
    neighbouring pieces into one 256-number token. Kernel = stride and no
    padding, so each token sees exactly a*b samples and none overlap."""
    def __init__(self, in_channels=12, out_channels=256, dropout=0.1, patch=PATCH):
        super().__init__()
        a, b = patch
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=a, stride=a),          # 10 samples -> 1 piece
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=b, stride=b),          # 5 pieces -> 1 token
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.token_len = a * b
        self.out_dim = out_channels

    def forward(self, x):                       # (N,12,5000)
        x = self.block1(x)                      # (N,64,500)
        x = self.block2(x)                      # (N,out_channels,100)
        return x.permute(0, 2, 1)               # (N,100,out_channels)

class FlatSeqAttention(nn.Module):
    VARIANTS = ("noenc", "conv", "bilstm")

    def __init__(self, num_classes, variant="noenc",
                 conv_channels=256, dropout=0.1,
                 seq_dim=512, seq_hidden=256): # defaults overridden by ARCH settings
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}")
        self.variant = variant

        # Per-100ms encoder: PatchCNNEncoder 
        # applied once over the WHOLE signal (no beat segmentation).
        self.encoder = PatchCNNEncoder(in_channels=12,
                                       out_channels=conv_channels,
                                       dropout=dropout) 
        enc_dim = self.encoder.out_dim

        # CONTEXT REACH
        if variant == "noenc":
            self.seq_encoder = SeqNoEnc(enc_dim, seq_dim, dropout)
        elif variant == "conv":
            self.seq_encoder = SeqConv(enc_dim, seq_dim, dropout)
        else:  # bilstm
            self.seq_encoder = SeqBiLSTM(enc_dim, seq_hidden, dropout)
        seq_out = self.seq_encoder.out_dim

        # SINGLE ATTN + CLASSIFIER
        self.attention  = AdditiveAttention(seq_out)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(seq_out, num_classes)

    def forward(self, x):
        logits, _ = self.forward_with_attention(x)
        return logits

    def forward_with_attention(self, x):
        # x: (B,12,T)
        feats          = self.encoder(x)                 # (B,T_red,enc_dim)
        feats          = self.seq_encoder(feats)         # (B,T_red,seq_out)
        context, alpha = self.attention(feats)           # (B,seq_out),(B,T_red,1)
        logits = self.classifier(self.dropout(context))  # (B,num_classes)
        return logits, alpha.squeeze(-1)                 # alphas: (B,T_red)