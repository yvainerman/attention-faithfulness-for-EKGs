"""
models/hierarchical_seq.py
--------------------------
ONE hierarchical attention model, parameterised by `variant`, implementing the
Serrano & Smith (2019) encoder-CONTEXTUALISATION-REACH axis at the SEQUENCE
(over-beats) level. Every variant has the same two-level HAN shape:

    beat:  beat-encoder -> beat-attention  -> one vector per beat
    seq:   seq-encoder  -> seq-attention   -> one vector per recording -> linear

What changes between variants is ONLY how far the sequence encoder lets a beat
"see" its neighbours -- exactly the axis S&S vary (rnn / conv / noenc):

  variant       beat encoder         sequence encoder (reach over beats)
  -----------   ------------------   ------------------------------------------
  "noenc"       CNN over samples     1x1 conv (per-beat, UPSAMPLES channels) --
                                     reach = 0 beats (each beat sees only itself)
  "conv"        CNN over samples     dual-width conv over beats (concat of
                                     kernel-5 and kernel-3) -- reach = +/-2 beats
  "bilstm_cnn"  CNN over samples     BiLSTM over beats -- reach = whole sequence
  "bilstm_full" BiLSTM over samples  BiLSTM over beats -- reach = whole sequence
                                     (S&S "HANrnn": recurrent at BOTH levels)

The first three share an IDENTICAL beat encoder (CNN), so noenc -> conv ->
bilstm_cnn is a clean controlled sweep of sequence-encoder reach with everything
else held fixed. "bilstm_full" additionally makes the BEAT encoder recurrent, so
it is the closest analogue to S&S's original recurrent HAN.

All variants emit a common sequence-feature width `seq_dim` (default 512) so the
sequence attention and classifier are identical across variants -- this is the
point of UPSAMPLING in the noenc variant: a 1x1 conv 256 -> 512 gives the
no-context model the same feature budget as the BiLSTM (2*seq_hidden=512), so a
faithfulness/accuracy difference cannot be blamed on width.

Forward signatures (identical across variants and to the older HCNN models):
    forward(beats)                -> logits                       (B, num_classes)
    forward_with_attention(beats) -> logits, seq_alphas, beat_alphas
        seq_alphas:  (B, S)            one weight per beat (S preserved)
        beat_alphas: (B, S, T_red)     within-beat weights (T_red = 75 for CNN
                                       beat encoder, 300 for bilstm_full)

`beats` is (B, S, T=300, C=12); length-bucketed batching gives a uniform S per
batch, so no masking is needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# SHARED MODULES
class CNNBeatEncoder(nn.Module):
    """CNN over a single beat: (N,12,300) -> (N,75,out_channels).
    Identical to the encoder in the existing hierarchical_cnn*.py models.
    Takes a single beat (1D time series signal with 12 channels) and
    compresses it into feature maps over time."""
    def __init__(self, in_channels=12, out_channels=256, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), 
            nn.MaxPool1d(2, 2), nn.Dropout(dropout), 
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2), nn.Dropout(dropout),
        )
        self.out_dim = out_channels # output width

    def forward(self, x):                       # (N,12,300)
        x = self.block1(x)
        x = self.block2(x)
        return x.permute(0, 2, 1)               # (N,75,out_channels)


class BiLSTMBeatEncoder(nn.Module):
    """BiLSTM over the RAW beat samples: (N,300,12) -> (N,300,2*hidden).
    The recurrent analogue of S&S's word-level BiGRU (here the per-sample
    12-lead vector plays the role of a word embedding)."""
    def __init__(self, in_channels=12, hidden=128, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, num_layers=1,
                        batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)  # applied after the LSTM, like the CNN's end-of-block dropout
        self.out_dim = 2 * hidden

    def forward(self, x):                       # (N,300,12)
        out, _ = self.lstm(x)
        return self.drop(out)                   # (N,300,2*hidden)


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention. (B,T,D) -> context (B,D), weights (B,T,1)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim) # multiply input by learned weights and add offset
        self.v = nn.Linear(hidden_dim, 1) # turn into single score

    def forward(self, features):
        scores  = self.v(torch.tanh(self.W(features)))    # (B,T,1)
        weights = F.softmax(scores, dim=1)                 # (B,T,1), dim=1 means across pos axis
        context = torch.sum(weights * features, dim=1)     # (B,D) weighted average of all pos vector
        return context, weights


# SEQUENCE ENCODERS (CONTEXT-REACH)
class SeqNoEnc(nn.Module):
    """noenc reach = 0. A 1x1 convolution (== per-beat Linear) that UPSAMPLES
    in_dim -> out_dim. Needed only to match widths, so that width does not confound comparison.
    Each output beat is a function of ONLY its own input beat, so no information
    crosses beats; the only thing it does is give the no-context model the 
    same feature width as the conv / BiLSTM variants.
    
    NOTE: lc: 256 + 256 from two convs, gl: 256 per direction.
    We adjust here with the linear layer to match these dimensions.
    To match S&S, we would have to swap to pass through, but this would alter
    parameter counts further and introduce different widths.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)    # 1x1 conv over the beat axis, turn from 256 to 512
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x):                          # (B,S,in_dim)
        return self.drop(self.act(self.fc(x)))     # (B,S,out_dim)


class SeqConv(nn.Module):
    """S&S convolutional encoder over the beat sequence: the
    concatenation of two convolutions of different widths: one kernel-5 (each
    output beat sees +/-2 neighbours) and one kernel-3 (+/-1) -- giving a fixed
    +/-2-beat contextualisation reach. Each branch produces out_dim//2 channels;
    they are concatenated to out_dim. Applied here over beats rather than timesteps."""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even (split across two convs)" # if 512 can't split into equal halves
        half = out_dim // 2 # 256
        self.conv_wide   = nn.Conv1d(in_dim, half, kernel_size=5, padding=2)  # kernel 5, so 2 beats per side
        self.conv_narrow = nn.Conv1d(in_dim, half, kernel_size=3, padding=1)  # kernel 3, 1 beat per side
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x):                          # (B,S,in_dim)
        x = x.transpose(1, 2)                      # (B,in_dim,S)
        x = torch.cat([self.conv_wide(x), self.conv_narrow(x)], dim=1)  # (B,out_dim,S)
        x = x.transpose(1, 2)                      # (B,S,out_dim)
        return self.drop(self.act(x))

class SeqBiLSTM(nn.Module):
    """rnn reach = whole sequence. BiLSTM over beats: (B,S,in_dim) ->
    (B,S,2*hidden). The S&S rnn encoder."""
    def __init__(self, in_dim, hidden, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)     # applied after the LSTM, like SeqNoEnc / SeqConv
        self.out_dim = 2 * hidden

    def forward(self, x):                          # (B,S,in_dim)
        out, _ = self.lstm(x)
        return self.drop(out)                      # (B,S,2*hidden)


# UNIFIED HIERARCHICAL MODEL
class HierarchicalSeqAttention(nn.Module):
    VARIANTS = ("noenc", "conv", "bilstm_cnn", "bilstm_full") # all possible hierarchical variants

    def __init__(self, num_classes, variant="noenc",
                 conv_channels=256, dropout=0.1,
                 seq_dim=512, beat_hidden=128, seq_hidden=256): # trainer overrides these with ARCH values.
        super().__init__()
        if variant not in self.VARIANTS: # asking for a version that doesnt exist causes error
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}")
        self.variant = variant

        # BEAT ENCODER
        if variant == "bilstm_full":
            self.beat_encoder = BiLSTMBeatEncoder(in_channels=12, hidden=beat_hidden,
                                      dropout=dropout)
            self._beat_is_cnn = False
        else:
            self.beat_encoder = CNNBeatEncoder(in_channels=12,
                                               out_channels=conv_channels,
                                               dropout=dropout)
            self._beat_is_cnn = True
        beat_dim = self.beat_encoder.out_dim
        self.beat_attention = AdditiveAttention(beat_dim)

        # SEQUENCE ENCODER
        if variant == "noenc":
            self.seq_encoder = SeqNoEnc(beat_dim, seq_dim, dropout)
        elif variant == "conv":
            self.seq_encoder = SeqConv(beat_dim, seq_dim, dropout)
        else:  # bilstm_cnn or bilstm_full
            self.seq_encoder = SeqBiLSTM(beat_dim, seq_hidden, dropout)
        seq_out = self.seq_encoder.out_dim
        self.seq_attention = AdditiveAttention(seq_out)

        # CLASSIFIER
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(seq_out, num_classes)

    def forward(self, beats):
        logits, _, _ = self.forward_with_attention(beats)
        return logits

    def forward_with_attention(self, beats):
        B, S, T, C = beats.shape

        # PER-BEAT PROCESSING (EACH BEAT ENCODED SEPARATELY)
        if self._beat_is_cnn:
            flat = beats.reshape(B * S, T, C).permute(0, 2, 1)   # (B*S,12,300)
        else:
            flat = beats.reshape(B * S, T, C)                    # (B*S,300,12)
        beat_feats             = self.beat_encoder(flat)         # (B*S,T_red,beat_dim)
        beat_vecs, beat_alphas = self.beat_attention(beat_feats)

        beat_dim = beat_vecs.shape[-1]
        T_red    = beat_alphas.shape[1]
        beat_vecs   = beat_vecs.view(B, S, beat_dim)             # (B,S,beat_dim)
        beat_alphas = beat_alphas.squeeze(-1).view(B, S, T_red)  # (B,S,T_red)

        # SEQ LEVEL ENCODER + ATTN
        seq_feats             = self.seq_encoder(beat_vecs)      # (B,S,seq_out)
        final_vec, seq_alphas = self.seq_attention(seq_feats)
        seq_alphas = seq_alphas.squeeze(-1)                      # (B,S)

        logits = self.classifier(self.dropout(final_vec))        # (B,num_classes)
        return logits, seq_alphas, beat_alphas