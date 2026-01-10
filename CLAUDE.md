# AudioCraft - Claude Instructions

## Project Overview

AudioCraft is a PyTorch library from Meta for audio generation. It includes MusicGen (text-to-music), AudioGen (text-to-sound), and EnCodec (audio codec).

## Local Modifications

This repo has been patched to run on macOS ARM (Apple Silicon) without xformers:

### Modified Files

1. **audiocraft/modules/transformer.py**
   - Made xformers import optional (try/except wrapper)
   - Falls back to PyTorch's `scaled_dot_product_attention` when xformers unavailable
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

model = MusicGen.get_pretrained('facebook/musicgen-medium', device='cpu')
model.set_generation_params(duration=10)
wav = model.generate(['acoustic guitar melody'])
audio_write('output', wav[0].cpu(), model.sample_rate, strategy='loudness')
```

### Available Models

- `facebook/musicgen-small` - Fast, lower quality
- `facebook/musicgen-medium` - Better quality, slower
- `facebook/musicgen-large` - Best quality, needs lots of RAM

## Known Issues

- xformers doesn't work on macOS (incompatible with PyTorch 2.1.0 via conda)
- Gradio demo has version compatibility issues
- Generation is slow on CPU (several minutes for medium model)

## Dependencies Note

Some packages have version mismatches from requirements.txt but work fine:
- av: 14.2.0 (requires 11.0.0)
- spacy: 3.8.7 (requires 3.7.6)
- xformers: not installed (required but made optional via patches)
