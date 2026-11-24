"""Text-conditioned aligner head (V1_LARGE_PLAN.md §5j, design C: late fusion, whole-song lyrics).

The audio encoder (omniASR + MuQ fusion) runs once, as for the CTC head. This head reads its frame
embeddings and the whole song's lyric tokens and scores every frame against every token:

    lead[t, 0]  = "no lead lyric here",  lead[t, 1 + j] = frame t sings lead token j
    back[t, 0]  = "no backing lyric",    back[t, 1 + k] = frame t sings backing token k

Each row is a log-softmax over [none, tokens of that voice], so the existing monotonic Viterbi decodes
it with token columns in place of letters (one "letter" per token, blank = column 0). Lead and backing
are two paths over the same frames, so they may overlap.

Text side: one embedding per acoustic token (its romanised letters through a character embedding and a
small convolution) plus a voice embedding (lead / backing) and a sinusoidal position over the song's
text order, then a small Transformer over the whole song. Repeated words differ by their context, so
the "lonely" of "(so lonely)" is not the lead's "lonely".

Frame side: a projection of the encoder's frame embeddings, then ``layers`` x (frame self-attention,
cross-attention to the song's tokens, feed-forward). Frames attend to every token of the song, so a
chunk needs no prior idea of which lines it holds; the global Viterbi over the song resolves repeats.

Shared by the trainer (``train_text_aligner.py``), the harness and inference; nothing here imports
transformers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

LEAD, BACKING = 0, 1
IMPOSSIBLE = -1e4  # score of a token column that does not belong to the voice


@dataclass
class TextAlignerConfig:
    frame_dim: int = 1024  # encoder hidden size (omniASR-300M)
    letter_dim: int = (
        30  # the encoder's CTC letter log-probabilities, appended to each frame (0: off)
    )
    dim: int = 256
    heads: int = 4
    text_layers: int = 4
    cross_layers: int = 2
    ff_mult: int = 4
    letters: int = 32  # >= the CTC vocabulary (30): letter ids index the character embedding
    max_letters: int = 16  # letters kept per token (longer tokens are truncated for the embedding)
    dropout: float = 0.1
    # Round 2 (2026-10-20; defaults keep the 6 M prototype loadable):
    # factorized: per voice P(singing) x P(token | singing) instead of one softmax over [none, tokens],
    # whose "none" won every frame and turned decoded words into 1-2 frame spikes.
    factorized: bool = False
    # line_embeddings: line index and position in the line on the text side (repeat disambiguation).
    line_embeddings: bool = False
    max_lines: int = 256
    max_line_positions: int = 64
    # qk_norm (2026-10-25): LayerNorm on each head's queries and keys in the cross layers' attention. Without
    # it the cross-attention query/key weights grew without bound (cross.0 query norm 17 -> 410 by step 34k)
    # until the softmax saturated and training diverged. Off keeps older checkpoints exact.
    qk_norm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# The head adds an absolute sinusoidal position to every frame and is trained on 20-27 s windows: past
# that length it has never seen the positions and its output degrades (on a 45 s window it named the
# sung line on 61 % of a song's frames, against 87 % at 25 s). Always run it in windows this long.
TRAINING_WINDOW_FRAMES = 1250  # 25 s at 20 ms
WINDOW_OVERLAP_FRAMES = 250


def frame_windows(
    frames: int, window: int = TRAINING_WINDOW_FRAMES, overlap: int = WINDOW_OVERLAP_FRAMES
):
    """Windows of at most ``window`` frames covering ``frames``, neighbours sharing ``overlap``:
    ``[(piece, keep), ...]`` with ``piece`` the slice to run and ``keep`` the slice of its output to
    use (half the overlap on each side), so the kept parts concatenate to exactly ``frames``."""
    step = window - overlap
    starts = list(range(0, max(1, frames - overlap), step)) or [0]
    windows = []
    for index, start in enumerate(starts):
        stop = min(frames, start + window)
        left = overlap // 2 if index else 0
        right = overlap - overlap // 2 if index + 1 < len(starts) else 0
        windows.append((slice(start, stop), slice(left, stop - start - right)))
    return windows


def sinusoid(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device) / half)
    angle = positions.float()[..., None] * scale
    return torch.cat((angle.sin(), angle.cos()), -1)


class TokenEmbedding(nn.Module):
    """Letters of one acoustic token -> one vector (character embedding, conv, masked mean)."""

    def __init__(self, config: TextAlignerConfig):
        super().__init__()
        self.letters = nn.Embedding(config.letters + 1, config.dim, padding_idx=config.letters)
        self.conv = nn.Conv1d(config.dim, config.dim, kernel_size=3, padding=1)
        self.pad = config.letters

    def forward(self, letters: torch.Tensor) -> torch.Tensor:
        """letters [B, N, L] (pad = config.letters) -> [B, N, dim]."""
        batch, tokens, length = letters.shape
        x = self.letters(letters.view(batch * tokens, length))  # [BN, L, dim]
        x = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        mask = (letters.view(batch * tokens, length) != self.pad).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return pooled.view(batch, tokens, -1)


class CrossLayer(nn.Module):
    def __init__(self, config: TextAlignerConfig):
        super().__init__()
        d, h = config.dim, config.heads
        self.self_norm, self.cross_norm, self.ff_norm = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            nn.LayerNorm(d),
        )
        self.self_attention = nn.MultiheadAttention(d, h, dropout=config.dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d, h, dropout=config.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, d * config.ff_mult),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * config.ff_mult, d),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.qk_norm = config.qk_norm
        if config.qk_norm:
            self.self_q_norm, self.self_k_norm = nn.LayerNorm(d // h), nn.LayerNorm(d // h)
            self.cross_q_norm, self.cross_k_norm = nn.LayerNorm(d // h), nn.LayerNorm(d // h)

    def attend(self, attention, query, key, value, key_padding_mask, q_norm=None, k_norm=None):
        if not self.qk_norm:
            return attention(
                query, key, value, key_padding_mask=key_padding_mask, need_weights=False
            )[0]
        heads, dim = attention.num_heads, attention.embed_dim
        q, k, v = F._in_projection_packed(
            query, key, value, attention.in_proj_weight, attention.in_proj_bias
        )
        split = lambda x: x.view(x.shape[0], x.shape[1], heads, dim // heads).transpose(
            1, 2
        )  # [B, H, T, dh]
        q, k, v = q_norm(split(q)), k_norm(split(k)), split(v)
        mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=attention.dropout if self.training else 0.0
        )
        return attention.out_proj(out.transpose(1, 2).reshape(query.shape[0], query.shape[1], dim))

    def forward(self, frames, frame_padding, text, text_padding):
        x = self.self_norm(frames)
        frames = frames + self.dropout(
            self.attend(
                self.self_attention,
                x,
                x,
                x,
                frame_padding,
                getattr(self, "self_q_norm", None),
                getattr(self, "self_k_norm", None),
            )
        )
        x = self.cross_norm(frames)
        frames = frames + self.dropout(
            self.attend(
                self.cross_attention,
                x,
                text,
                text,
                text_padding,
                getattr(self, "cross_q_norm", None),
                getattr(self, "cross_k_norm", None),
            )
        )
        return frames + self.dropout(self.ff(self.ff_norm(frames)))


class TextAligner(nn.Module):
    """Scores frames against the song's lead and backing tokens (module docstring)."""

    def __init__(self, config: TextAlignerConfig | None = None):
        super().__init__()
        self.config = config = config or TextAlignerConfig()
        d = config.dim
        self.tokens = TokenEmbedding(config)
        self.voice = nn.Embedding(2, d)
        if config.line_embeddings:
            self.line = nn.Embedding(config.max_lines, d)
            self.line_position = nn.Embedding(config.max_line_positions, d)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d,
                config.heads,
                d * config.ff_mult,
                config.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            config.text_layers,
            enable_nested_tensor=False,
        )
        self.text_norm = nn.LayerNorm(d)
        self.frame_in = nn.Sequential(
            nn.LayerNorm(config.frame_dim + config.letter_dim),
            nn.Linear(config.frame_dim + config.letter_dim, d),
        )
        self.cross = nn.ModuleList(CrossLayer(config) for _ in range(config.cross_layers))
        self.frame_norm = nn.LayerNorm(d)
        # Per voice: query/key projections for the token scores and a "none" / "singing" logit per frame.
        self.query = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.key = nn.ModuleList(nn.Linear(d, d) for _ in range(2))
        self.none = nn.ModuleList(nn.Linear(d, 1) for _ in range(2))
        # Letter-evidence bias (bootstrap): + scale x log(mean CTC probability of the token's letters at
        # the frame), so from the first step a frame that sounds like l-o-n-e favours tokens spelled so;
        # attention then learns which of them (context, voice).
        self.letter_scale = nn.Parameter(torch.ones(2))

    def encode_text(self, letters, voice, text_padding, line=None, position=None):
        """letters [B, N, L], voice [B, N] (0 lead / 1 backing), text_padding [B, N] (True = pad);
        ``line`` / ``position`` [B, N] (line index, position in the line) with ``line_embeddings``."""
        positions = torch.arange(letters.shape[1], device=letters.device).expand(letters.shape[:2])
        x = self.tokens(letters) + self.voice(voice) + sinusoid(positions, self.config.dim)
        if self.config.line_embeddings:
            x = (
                x
                + self.line(line.clamp(0, self.config.max_lines - 1))
                + self.line_position(position.clamp(0, self.config.max_line_positions - 1))
            )
        return self.text_norm(self.text_encoder(x, src_key_padding_mask=text_padding))

    def forward_parts(
        self,
        hidden,
        frame_padding,
        letters,
        voice,
        text_padding,
        letter_log_probs=None,
        line=None,
        position=None,
    ):
        """Per voice: ``combined`` [B, T, 1 + N] log-probabilities over [none, every token] (what the
        decoders read), and with ``factorized`` also ``singing`` logits [B, T] and ``token`` log-probs
        [B, T, N] (normalised over that voice's tokens only)."""
        text = self.encode_text(letters, voice, text_padding, line, position)
        if self.config.letter_dim:
            hidden = torch.cat((hidden, letter_log_probs.to(hidden.dtype)), -1)
        frames = self.frame_in(hidden)
        frames = frames + sinusoid(
            torch.arange(frames.shape[1], device=frames.device), self.config.dim
        )
        for layer in self.cross:
            frames = layer(frames, frame_padding, text, text_padding)
        frames = self.frame_norm(frames)
        evidence = None
        if self.config.letter_dim:
            pad = self.config.letters
            hot = F.one_hot(letters.clamp_max(self.config.letter_dim), self.config.letter_dim + 1)[
                ..., :-1
            ].float()
            hot = hot * (letters != pad).unsqueeze(-1).float()  # [B, N, L, letter_dim]
            mix = hot.sum(2) / hot.sum((2, 3)).clamp_min(1.0).unsqueeze(
                -1
            )  # [B, N, letter_dim], rows sum to 1
            evidence = torch.log(
                torch.einsum("btc,bnc->btn", letter_log_probs.float().exp(), mix).clamp_min(1e-6)
            )
        parts = []
        for v in (LEAD, BACKING):
            scores = torch.einsum(
                "btd,bnd->btn", self.query[v](frames), self.key[v](text)
            ).float() / math.sqrt(self.config.dim)
            if evidence is not None:
                scores = scores + self.letter_scale[v] * evidence
            usable = (~text_padding) & (voice == v)
            # Finite "impossible": CTC's backward turns -inf emissions into NaN gradients even for
            # columns the target never uses.
            scores = scores.masked_fill(~usable[:, None, :], IMPOSSIBLE)
            none = self.none[v](
                frames
            ).float()  # [B, T, 1]; with factorized it is the "singing" logit
            if self.config.factorized:
                token = F.log_softmax(scores, -1)
                singing = none[..., 0]
                combined = torch.cat((F.logsigmoid(-none), F.logsigmoid(none) + token), -1)
                parts.append({"combined": combined, "singing": singing, "token": token})
            else:
                parts.append({"combined": F.log_softmax(torch.cat((none, scores), -1), -1)})
        return parts

    def forward(
        self,
        hidden,
        frame_padding,
        letters,
        voice,
        text_padding,
        letter_log_probs=None,
        line=None,
        position=None,
    ):
        """(lead, backing) log-probabilities [B, T, 1 + N] over [none, every token of the song]; a voice's
        column j + 1 is ~IMPOSSIBLE unless token j belongs to that voice (and is not padding)."""
        parts = self.forward_parts(
            hidden, frame_padding, letters, voice, text_padding, letter_log_probs, line, position
        )
        return parts[LEAD]["combined"], parts[BACKING]["combined"]


def forward_sum_loss(log_probs, sequences, lengths, frames):
    """CTC over token columns (blank = column 0, "none"), mean over windows with targets (RAD-TTS
    forward-sum). ``sequences`` [B, S] padded token columns, ``lengths`` [B]."""
    rows = (lengths > 0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return log_probs.new_zeros(())
    targets = torch.cat([sequences[r, : lengths[r]] for r in rows.tolist()])
    return F.ctc_loss(
        log_probs[rows].transpose(0, 1),
        targets,
        frames[rows],
        lengths[rows],
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )


def voice_loss(
    part,
    targets,
    sequences,
    lengths,
    frames,
    fs_weight,
    fs_mode="ctc",
    regions=None,
    region_loss=None,
):
    """One voice's loss: factorized = BCE(singing) + CE(token | singing frames) + fs_weight x forward-sum;
    otherwise CE over [none, tokens] + fs_weight x forward-sum. ``targets`` [B, T]: -100 ignore, 0 none,
    1 + j token column. Returns (loss, parts dict of floats).

    ``fs_mode="region"`` (factorized heads, with ``regions`` [B, R, 3 + L] = first frame, stop frame, token count,
    token columns): instead of the CTC forward-sum, each group of contiguous tokens fills its measured frames
    blank-free and in order, boundaries free (``region_loss``: the trainer's monotonic_alignment_loss), per frame,
    the same objective the letter path uses and decoder A assumes."""
    combined = part["combined"]
    targets = targets[:, : combined.shape[1]]
    graded = targets != -100
    stats = {}
    if not graded.any():
        return combined.new_zeros(()), stats
    if fs_mode == "region" and "token" in part and regions is not None and region_loss is not None:
        segments, columns, covered = [], [], 0
        for batch, region in (regions[..., 2] > 0).nonzero().tolist():
            first, stop, count = regions[batch, region, :3].tolist()
            stop = min(stop, int(frames[batch]))
            if stop - first >= count > 0:
                segments.append(part["token"][batch, first:stop])
                columns.append(regions[batch, region, 3 : 3 + count] - 1)
                covered += stop - first
        fs = (
            region_loss(segments, columns) / max(covered, 1) if segments else combined.new_zeros(())
        )
        if "singing" in part:
            sung = graded & (targets > 0)
            bce = F.binary_cross_entropy_with_logits(part["singing"][graded], sung[graded].float())
            ce = (
                F.nll_loss(part["token"][sung], targets[sung] - 1)
                if sung.any()
                else combined.new_zeros(())
            )
            stats.update(bce=bce.detach(), ce=ce.detach(), fs=fs.detach())
            return bce + ce + fs_weight * fs, stats
    if "singing" in part:
        # The forward-sum shapes only WHICH token: its blank is "not singing", and CTC rewards spiky paths
        # (blank nearly everywhere), which pulled the singing logit down on sung frames (BCE ~1.4, worse than
        # a constant guess; fa-300m-muq-100k-v2-text to step 8000). The frame BCE alone owns "singing?".
        singing = part["singing"].detach()[..., None]
        fs_input = torch.cat((F.logsigmoid(-singing), F.logsigmoid(singing) + part["token"]), -1)
    else:
        fs_input = combined
    fs = forward_sum_loss(fs_input, sequences, lengths, frames)
    if "singing" in part:
        sung = graded & (targets > 0)
        bce = F.binary_cross_entropy_with_logits(part["singing"][graded], sung[graded].float())
        ce = (
            F.nll_loss(part["token"][sung], targets[sung] - 1)
            if sung.any()
            else combined.new_zeros(())
        )
        loss = bce + ce + fs_weight * fs
        stats.update(bce=bce.detach(), ce=ce.detach())
    else:
        ce = F.nll_loss(combined.transpose(1, 2), targets, ignore_index=-100)
        loss = ce + fs_weight * fs
        stats.update(ce=ce.detach())
    stats["fs"] = fs.detach()
    return loss, stats


def pack_song_tokens(tokens: list[dict], config: TextAlignerConfig) -> dict:
    """One song's tokens in text order -> tensors for a batch of one (padding added by ``collate_text``).

    ``tokens``: dicts with ``ids`` (letter ids of the CTC vocabulary), ``voice`` (0 lead / 1 backing) and
    ``line``, already in text order (line, then position in the line)."""
    letters = torch.full((len(tokens), config.max_letters), config.letters, dtype=torch.long)
    for index, token in enumerate(tokens):
        ids = list(token["ids"])[: config.max_letters]
        if ids:
            letters[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    voice = torch.tensor([int(token.get("voice", LEAD)) for token in tokens], dtype=torch.long)
    line, position, previous, count = [], [], None, 0
    for token in tokens:
        number = int(token.get("line", 0))
        count = count + 1 if number == previous else 0
        previous = number
        line.append(number)
        position.append(count)
    return {
        "letters": letters,
        "voice": voice,
        "line": torch.tensor(line, dtype=torch.long),
        "position": torch.tensor(position, dtype=torch.long),
    }


def collate_text(songs: list[dict], config: TextAlignerConfig) -> dict:
    """Pad packed songs to the longest: letters [B, N, L], voice / line / position [B, N], text_padding [B, N]."""
    most = max(1, max(song["letters"].shape[0] for song in songs))
    letters = torch.full((len(songs), most, config.max_letters), config.letters, dtype=torch.long)
    voice = torch.zeros((len(songs), most), dtype=torch.long)
    line = torch.zeros((len(songs), most), dtype=torch.long)
    position = torch.zeros((len(songs), most), dtype=torch.long)
    padding = torch.ones((len(songs), most), dtype=torch.bool)
    for row, song in enumerate(songs):
        n = song["letters"].shape[0]
        letters[row, :n], voice[row, :n], padding[row, :n] = song["letters"], song["voice"], False
        if "line" in song:
            line[row, :n], position[row, :n] = song["line"], song["position"]
    return {
        "letters": letters,
        "voice": voice,
        "line": line,
        "position": position,
        "text_padding": padding,
    }


def voice_columns(voice: torch.Tensor, v: int) -> torch.Tensor:
    """Indices (into the token axis, without the none column) of one voice's tokens, in text order."""
    return (voice == v).nonzero(as_tuple=True)[0]


def decode_voice(
    log_probs: torch.Tensor,
    columns: list[int],
    *,
    line_ranges: list[tuple[int, int]] | None = None,
    minimum_frames: list[int] | None = None,
    in_line_gap_cost: float = 0.0,
):
    """Token spans of one voice from its output [T, 1 + N] (one song): the existing monotonic Viterbi
    (``ctc_segmentation.align_with_minimum_durations``) with one "letter" per token, the token's own
    column as its emission and column 0 ("none") as the blank. ``columns`` are the voice's token columns
    (1 + text index) in text order; ``line_ranges`` index into ``columns`` (for the in-line gap cost).
    Returns one TokenSpan per token (start/end frames, mean probability)."""
    from ctc_segmentation import align_with_minimum_durations

    groups = [[int(column)] for column in columns]
    spans = align_with_minimum_durations(
        log_probs.float(),
        groups,
        blank=0,
        minimum_frames=minimum_frames or [1] * len(groups),
        line_ranges=line_ranges,
        in_line_gap_cost=in_line_gap_cost,
    )
    return [group[0] for group in spans]
