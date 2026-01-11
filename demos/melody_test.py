"""Test script for MusicGen melody conditioning on macOS ARM."""
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

device = 'cpu'

# Load the melody model (required for melody conditioning)
print("Loading musicgen-melody model...")
model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
model.set_generation_params(
    duration=10,
    use_sampling=True,
    top_k=250,
    temperature=1.0,
    cfg_coef=3.0
)

# Load your melody/reference audio
# Replace 'melody.wav' with your audio file
melody_path = 'melody.wav'
print(f"Loading melody from {melody_path}...")
melody, sr = torchaudio.load(melody_path)

# Generate conditioned on the melody
print("Generating with melody conditioning...")
wav = model.generate_with_chroma(
    descriptions=['lo-fi hip hop beats to study to'],
    melody_wavs=melody,
    melody_sample_rate=sr
)

# Save output
audio_write(
    'melody_output',
    wav[0].cpu(),
    model.sample_rate,
    strategy='loudness',
    loudness_headroom_db=16,
    loudness_compressor=True
)
print("Saved to melody_output.wav")
