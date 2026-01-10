import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Force CPU to avoid any MPS issues on Apple Silicon
device = 'cpu'
print(f'Using device: {device}')

print('Loading model...')
model = MusicGen.get_pretrained('facebook/musicgen-small', device=device)
print('Model sample rate:', model.sample_rate)

model.set_generation_params(
    duration=10,
    use_sampling=True,
    top_k=250,
    top_p=0.0,
    temperature=1.0,
    cfg_coef=3.0
)
print('Generating...')
wav = model.generate(['lo-fi hip hop beats to study to'])

print('Output shape:', wav.shape)

# Use audiocraft's audio_write with proper normalization
audio_write(
    'output',
    wav[0].cpu(),
    model.sample_rate,
    strategy='loudness',
    loudness_headroom_db=16,
    loudness_compressor=True
)
print('Done! Saved output.wav')