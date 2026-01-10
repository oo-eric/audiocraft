import scipy.io.wavfile as wavfile
import numpy as np
import torch
from audiocraft.models import MusicGen

# Force CPU to avoid any MPS issues on Apple Silicon
device = 'cpu'
print(f'Using device: {device}')

print('Loading model...')
# Try medium model for better quality (uses more RAM)
model = MusicGen.get_pretrained('facebook/musicgen-medium', device=device)
print('Model sample rate:', model.sample_rate)

model.set_generation_params(duration=10)
print('Generating (this will take a few minutes on CPU)...')
wav = model.generate(['acoustic folk song with guitar and soft vocals'])

print('Output shape:', wav.shape)

# Get audio as numpy
audio = wav[0, 0].cpu().numpy()
print('Audio numpy shape:', audio.shape)
print('Audio min/max:', audio.min(), audio.max())
print('Audio dtype:', audio.dtype)

# Check if audio looks reasonable
print('First 10 samples:', audio[:10])
print('Audio std:', audio.std())

# Normalize to [-1, 1] then convert to int16
audio = np.clip(audio, -1.0, 1.0)
audio_int16 = (audio * 32767).astype(np.int16)

# Save as 32kHz WAV
wavfile.write('output.wav', model.sample_rate, audio_int16)
print('Done! Saved output.wav at', model.sample_rate, 'Hz')