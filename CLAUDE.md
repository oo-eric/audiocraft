# AudioCraft - Claude Instructions

## Project Overview

AudioCraft is a PyTorch library from Meta for audio generation. It includes MusicGen (text-to-music), AudioGen (text-to-sound), and EnCodec (audio codec).

## Local Modifications

This repo has been patched to run on macOS ARM (Apple Silicon) without xformers:

### Modified Files

1. **audiocraft/modules/transformer.py**
   - Made xformers import optional (try/except wrapper)
   - Without xformers, `memory_efficient` attention runs on torch's `scaled_dot_product_attention`
     (~1.6-2x faster on MPS than the einsum path, same outputs)
   - `_get_mask` returns `None` for single-step decoding whether or not xformers is installed. The
     old "garbled audio" bug came from building a mask tensor there: the SDPA branch then ran
     `is_causal=True` with 1 query vs N keys, which torch aligns top-left, so each new token only
     attended to the first key
   - `set_efficient_attention_backend('xformers')` falls back to `'torch'` when xformers is missing.
     The backend also picks the q/k/v layout, so leaving it on `'xformers'` (as the MAGNeT loader
     does) would hand SDPA a `b t h d` tensor where it expects `b h t d`
   - Uses `torch.unbind` instead of `xformers.ops.unbind`

2. **audiocraft/utils/profiler.py**
   - Made xformers profiler import optional with try/except

3. **audiocraft/utils/autocast.py**
   - Disables autocast for MPS device. Originally because torch 2.1 didn't support it; torch 2.14
     does, but fp16/bf16 autocast gave no speed or memory gain (weights stay fp32), so it stays off

## Setup (macOS ARM)

Python 3.12 virtualenv managed with `uv`, torch 2.14.1 + torchaudio 2.11.0:

```bash
source .venv/bin/activate
```

See NOTES.md for full installation instructions.

## MPS (Metal GPU) Acceleration

Use Apple's Metal GPU instead of CPU for faster generation:

```bash
# Enable MPS fallback for unsupported ops. MusicGen no longer needs this on torch 2.14;
# AudioGen/MAGNeT/JASCO haven't been re-checked, so keep setting it.
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

```python
# Use device='mps' instead of 'cpu'
model = MusicGen.get_pretrained('facebook/musicgen-small', device='mps')
```

### MPS Performance

torch 2.14.1, generation time only (after model load + one warm-up generate). "Before" is
`memory_efficient` forced off (the old patch), "after" is torch SDPA:

| Model | Before | After |
|-------|--------|-------|
| MusicGen-small (5s) | 5.8s | 3.0s |
| AudioGen-medium (5s) | 11.9s | 7.3s |
| MAGNeT-small (10s) | 4.8s | 4.4s |
| JASCO-400M (10s, Euler 50 steps) | 6.5s | 6.3s |

Outputs match: MusicGen/AudioGen tokens are bit-identical, JASCO latents differ by ~1e-6
relative, and MAGNeT forward logits differ by ~1e-6 with every argmax agreeing (its sampled tokens
drift apart over the iterative decode, which is expected).

Mixed precision, measured before the SDPA fix:

| Model | fp32 (default) | fp16 autocast | bf16 autocast |
|-------|----------------|---------------|---------------|
| MusicGen-small | 5.8s | 6.3s | — |
| MusicGen-medium | 11.6s | 11.7s | 14.6s |

MusicGen-medium peaks at ~5.4 GB of MPS memory with or without autocast. At batch size 1 the
decode loop is bound by kernel launches, not by math, so lower precision doesn't help. If memory
becomes the bottleneck, try casting the LM weights to fp16 rather than enabling autocast.

Older torch 2.1 numbers (MPS / CPU), not re-measured: MusicGen-small 15s / 28s,
AudioGen-medium 29s / 82s, MAGNeT-small (10s) 8s / 29s.

## Key Files

- `demos/musicgen_app.py` - Gradio UI (was pinned to gradio==3.50.2; untested on torch 2.14)

## Common Tasks

### Generate Music

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-small', device='mps')  # or 'cpu'
model.set_generation_params(
    duration=10,
    use_sampling=True,
    top_k=250,
    top_p=0.0,
    temperature=1.0,
    cfg_coef=3.0
)
wav = model.generate(['lo-fi hip hop beats to study to'])
audio_write(
    'output',
    wav[0].cpu(),
    model.sample_rate,
    strategy='loudness',
    loudness_headroom_db=16,
    loudness_compressor=True
)
```

### Available Models

- `facebook/musicgen-small` - Fast, lower quality
- `facebook/musicgen-medium` - Better quality, slower
- `facebook/musicgen-large` - Best quality, needs lots of RAM

## Known Issues

- xformers has no macOS wheels
- Gradio demo was pinned to 3.50.2 and hasn't been tested on the torch 2.14 setup
- `torchaudio.load`/`save` need `torchcodec` on torchaudio 2.9+. Inference is unaffected, but the
  mp3/aac augmentation in `audiocraft/data/audio_utils.py` and the visqol metric will fail without it
- JASCO on MPS fails with its default dopri5 ODE solver: torchdiffeq builds its tolerances in
  float64, which MPS doesn't support. Use `set_generation_params(euler=True, euler_steps=50)`
- Installing xformers on macOS buys nothing: there's no wheel, the source build needs `-fopenmp`
  stripped for Apple clang, and every attention kernel is CUDA/ROCm only (MAGNeT crashes on MPS
  with it installed, since its loader asks for the xformers backend)

## Dependencies Note

Tested versions: torch 2.14.1, torchaudio 2.11.0, av 19.0.0, numpy 1.26.4, spacy 3.8.16,
transformers 5.18.0. xformers isn't installed (it's commented out in requirements.txt; the patches
make it optional). `torchvision` and `torchtext` were dropped because nothing imports them.

## Upstream PR Notes

When submitting a PR to facebookresearch/audiocraft, reference these related issues:

**macOS/Apple Silicon (long-standing requests):**
- #230 - xFormers error on M1
- #43 - Apple Silicon M1 support request
- #31 - Apple Silicon feature request
- #13 - Running on M1 help
- #587 - CPU-only Docker support for macOS ARM64
- #573 - Training on Mac M4 memory issues

**xformers issues:**
- #407 - MAGNeT models require xformers backend
- #362 - xformers/torch module conflict

**Related PRs:**
- #487 - Update dependencies for torch v2.2.x, 2.3.x (similar compatibility goal)
