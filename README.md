# AudioCraft

![docs badge](https://github.com/oo-eric/audiocraft/workflows/audiocraft_docs/badge.svg)
![linter badge](https://github.com/oo-eric/audiocraft/workflows/audiocraft_linter/badge.svg)
![tests badge](https://github.com/oo-eric/audiocraft/workflows/audiocraft_tests/badge.svg)

AudioCraft is a PyTorch library for deep learning research on audio generation. AudioCraft contains inference and training code
for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.

## Installation

This fork targets PyTorch 2.14.1 and has been tested with Python 3.12 on macOS Apple Silicon
(see [below](#macos-apple-silicon)); CUDA hasn't been tested on this version. Upstream AudioCraft
requires Python 3.9 and PyTorch 2.1.0. To install AudioCraft, you can run the following:

```shell
# Best to make sure you have torch installed first.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.14.1'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
# Then proceed to one of the following
python -m pip install -U audiocraft  # stable release
python -m pip install -U git+https://git@github.com/oo-eric/audiocraft@torch_upgrade#egg=audiocraft  # bleeding edge
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
python -m pip install -e '.[wm]'  # if you want to train a watermarking model
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:

```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

### macOS Apple Silicon

AudioCraft runs on Apple Silicon (M1–M4) and uses the Metal GPU through PyTorch's `mps` device. You
don't need xformers: without it, attention runs on PyTorch's `scaled_dot_product_attention`.
Installing xformers on macOS doesn't help, since its attention kernels are CUDA-only.

```shell
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.14.1 torchaudio==2.11.0
uv pip install av einops "flashy>=0.0.1" "hydra-core>=1.1" hydra_colorlog julius num2words \
    "numpy<2.0.0" sentencepiece spacy huggingface_hub tqdm "transformers>=4.31.0" demucs librosa \
    soundfile torchmetrics encodec protobuf torchdiffeq
uv pip install --no-deps -e .
uv pip install gradio  # optional, for the web demo
```

Pass `device='mps'` when loading a model:

```python
from audiocraft.models import MusicGen
model = MusicGen.get_pretrained('facebook/musicgen-small', device='mps')
```

Some ops have no MPS kernel yet, so set `PYTORCH_ENABLE_MPS_FALLBACK=1` to let them fall back to the
CPU. All of MusicGen (including melody conditioning), AudioGen, MAGNeT, JASCO and the MultiBand
Diffusion decoder run on MPS. Generation times on MPS with PyTorch 2.14.1 (after a warm-up run):

| Model           | Audio | Time                                                               |
| --------------- | ----- | ------------------------------------------------------------------ |
| MusicGen-small  | 5s    | 3.0s                                                               |
| MusicGen-medium | 5s    | 6.6s                                                               |
| AudioGen-medium | 5s    | 7.3s                                                               |
| MAGNeT-small    | 10s   | 4.4s                                                               |
| JASCO-400M      | 10s   | 19.4s (default dopri5 solver), 6.3s (`euler=True, euler_steps=50`) |

The Gradio demo (`python -m demos.musicgen_app`) picks MPS automatically.

## Models

At the moment, AudioCraft contains the training code and inference code for:

- [MusicGen](./docs/MUSICGEN.md): A state-of-the-art controllable text-to-music model.
- [AudioGen](./docs/AUDIOGEN.md): A state-of-the-art text-to-sound model.
- [EnCodec](./docs/ENCODEC.md): A state-of-the-art high fidelity neural audio codec.
- [Multi Band Diffusion](./docs/MBD.md): An EnCodec compatible decoder using diffusion.
- [MAGNeT](./docs/MAGNET.md): A state-of-the-art non-autoregressive model for text-to-music and text-to-sound.
- [AudioSeal](./docs/WATERMARKING.md): A state-of-the-art audio watermarking.
- [MusicGen Style](./docs/MUSICGEN_STYLE.md): A state-of-the-art text-and-style-to-music model.
- [JASCO](./docs/JASCO.md): "High quality text-to-music model conditioned on chords, melodies and drum tracks"

## Training code

AudioCraft contains PyTorch components for deep learning research in audio and training pipelines for the developed models.
For a general introduction of AudioCraft design principles and instructions to develop your own training pipeline, refer to
the [AudioCraft training documentation](./docs/TRAINING.md).

For reproducing existing work and using the developed training pipelines, refer to the instructions for each specific model
that provides pointers to configuration, example grids and model/task-specific information and FAQ.

## API documentation

We provide some [API documentation](https://oo-eric.github.io/audiocraft/api_docs/audiocraft/index.html) for AudioCraft.

## FAQ

#### Is the training code available?

Yes! We provide the training code for [EnCodec](./docs/ENCODEC.md), [MusicGen](./docs/MUSICGEN.md),[Multi Band Diffusion](./docs/MBD.md) and [JASCO](./docs/JASCO.md).

#### Where are the models stored?

Hugging Face stored the model in a specific location, which can be overridden by setting the `AUDIOCRAFT_CACHE_DIR` environment variable for the AudioCraft models.
In order to change the cache location of the other Hugging Face models, please check out the [Hugging Face Transformers documentation for the cache setup](https://huggingface.co/docs/transformers/installation#cache-setup).
Finally, if you use a model that relies on Demucs (e.g. `musicgen-melody`) and want to change the download location for Demucs, refer to the [Torch Hub documentation](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved).

## License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
- The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).

## Citation

For the general framework of AudioCraft, please cite the following.

```
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

When referring to a specific model, please cite as mentioned in the model specific README, e.g
[./docs/MUSICGEN.md](./docs/MUSICGEN.md), [./docs/AUDIOGEN.md](./docs/AUDIOGEN.md), etc.
