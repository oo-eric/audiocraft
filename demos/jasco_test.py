"""Test JASCO model on macOS ARM without xformers."""

from audiocraft.models import JASCO
from audiocraft.data.audio import audio_write
import os

# Force CPU
device = 'cpu'
print(f'Using device: {device}')

# Path to chord mapping file
assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
chords_mapping_path = os.path.join(assets_dir, 'chord_to_index_mapping.pkl')

print('Loading JASCO model...')
model = JASCO.get_pretrained(
    'facebook/jasco-chords-drums-400M',
    chords_mapping_path=chords_mapping_path,
    device=device
)
print('Model sample rate:', model.sample_rate)

model.set_generation_params(
    cfg_coef_all=5.0,
    cfg_coef_txt=0.0
)

# Text description
text = "Upbeat jazz piano with drums"

# Chord progression: (chord, start_time_in_seconds)
chords = [
    ('C', 0.0),
    ('Am', 2.0),
    ('F', 4.0),
    ('G', 6.0),
    ('C', 8.0)
]

print('Generating music with chord conditioning...')
print(f'  Text: {text}')
print(f'  Chords: {chords}')

output = model.generate_music(descriptions=[text], chords=chords, progress=True)

print('Output shape:', output.shape)

audio_write(
    'jasco_output',
    output.cpu().squeeze(0),
    model.sample_rate,
    strategy='loudness',
    loudness_compressor=True
)
print('Done! Saved jasco_output.wav')
