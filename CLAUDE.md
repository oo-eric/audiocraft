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
     does, but the transformer weights are already fp16, so autocast adds nothing (see
     "Precision on MPS")

4. **audiocraft/models/flow_matching.py**
   - On MPS, passes `dtype=torch.float32` to torchdiffeq's dopri5 solver. It keeps step sizes and
     tolerances in float64 by default, which MPS doesn't support, so JASCO crashed unless you
     switched to `euler=True`. On CPU, float32 vs float64 time handling gave the same 122 solver
     steps and latents within 4e-6 relative

5. **audiocraft/models/loaders.py**
   - `_get_state_dict` allowlists the omegaconf classes some checkpoints pickle (via
     `torch.serialization.safe_globals`). torch 2.6+ defaults to `weights_only=True`, which made the
     MultiBand Diffusion checkpoint fail to load. Avoids `weights_only=False`, which would let a
     checkpoint run arbitrary code

6. **demos/musicgen_app.py**
   - Picks cuda > mps > cpu (it used to fall back to CPU on anything without CUDA) and defaults
     `PYTORCH_ENABLE_MPS_FALLBACK=1`
   - Ported to Gradio 6: `gr.make_waveform` is gone, so the video outputs were dropped (the audio
     player draws its own waveform), and the 3.x `source=` became `sources=[...]`

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
| JASCO-400M (10s, default dopri5) | crashed | 19.4s |

Outputs match: MusicGen/AudioGen tokens are bit-identical, JASCO latents differ by ~1e-6
relative, and MAGNeT forward logits differ by ~1e-6 with every argmax agreeing (its sampled tokens
drift apart over the iterative decode, which is expected).

### Precision on MPS

On any non-CPU device the loader sets `cfg.dtype = 'float16'`, so the LM transformer weights are
stored in fp16 (embeddings, output heads and the conditioner projection stay fp32). The input
activations are fp32, and MPS silently upcasts fp16 weights in mixed-dtype ops (CPU raises
instead), so the math runs in fp32. That gives fp16 memory with fp32 numerics: tokens are
bit-identical to an all-fp32 LM.

MusicGen-medium, 5s, with the SDPA fix:

| LM precision | Time | Peak MPS memory | Tokens |
|---|---|---|---|
| Default (fp16 storage, fp32 compute) | 6.6s | 5.3 GiB | — |
| `model.lm.half()` (fp16 compute) | 6.3s | 5.3 GiB | diverge from step 0; no NaNs |
| `model.lm.float()` (all fp32) | 8.1s | 11.3 GiB | identical to default |

Autocast, measured before the SDPA fix with the default weights: fp16 autocast matched the
default (11.7s vs 11.6s for medium) and bf16 was slower (14.6s), because it re-casts the fp16
weights on every op.

Older torch 2.1 numbers (MPS / CPU), not re-measured: MusicGen-small 15s / 28s,
AudioGen-medium 29s / 82s, MAGNeT-small (10s) 8s / 29s.

## Key Files

- `demos/musicgen_app.py` - Gradio UI (gradio 6.29): `python -m demos.musicgen_app`, then open the URL it prints

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
- Gradio sends usage analytics by default; set `GRADIO_ANALYTICS_ENABLED=False` to turn that off
- `torchaudio.load`/`save` need `torchcodec` on torchaudio 2.9+. Inference is unaffected, but the
  mp3/aac augmentation in `audiocraft/data/audio_utils.py` and the visqol metric will fail without it
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
