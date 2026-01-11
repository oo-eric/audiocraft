# AudioCraft - Claude Instructions

## Project Overview

AudioCraft is a PyTorch library from Meta for audio generation. It includes MusicGen (text-to-music), AudioGen (text-to-sound), and EnCodec (audio codec).

## Local Modifications

This repo has been patched to run on macOS ARM (Apple Silicon) without xformers:

### Modified Files

1. **audiocraft/modules/transformer.py**
   - Made xformers import optional (try/except wrapper)
   - Auto-disables `memory_efficient` attention when xformers unavailable (the PyTorch fallback doesn't work correctly and produces garbled audio)
   - Falls back to standard PyTorch MultiheadAttention instead
   - Uses `torch.unbind` instead of `xformers.ops.unbind`

2. **audiocraft/utils/profiler.py**
   - Made xformers profiler import optional with try/except

## Setup (macOS ARM)

```bash
conda activate audiocraft
```

See NOTES.md for full installation instructions.

## Key Files

- `demos/test.py` - Simple generation test script
- `demos/codec_test.py` - EnCodec encode/decode test
- `demos/musicgen_app.py` - Gradio UI (has version issues)

## Common Tasks

### Generate Music

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('facebook/musicgen-small', device='cpu')
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

- xformers doesn't work on macOS (incompatible with PyTorch 2.1.0 via conda)
- Gradio demo requires version 3.50.2 (run `pip install gradio==3.50.2`)
- Generation is slow on CPU (several minutes for medium model)

## Dependencies Note

Some packages have version mismatches from requirements.txt but work fine:
- av: 14.2.0 (requires 11.0.0)
- spacy: 3.8.7 (requires 3.7.6)
- xformers: not installed (required but made optional via patches)

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
