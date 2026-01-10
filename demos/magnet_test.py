"""Test MAGNeT model on macOS ARM without xformers.

Note: MAGNeT docs say "you must have a GPU" - this test may fail.
Issue #407 suggests MAGNeT has a hard xformers requirement.
"""

from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

# Force CPU
device = 'cpu'
print(f'Using device: {device}')

print('Loading MAGNeT model...')
try:
    model = MAGNeT.get_pretrained('facebook/magnet-small-10secs', device=device)
    print('Model sample rate:', model.sample_rate)

    print('Generating music...')
    wav = model.generate(['upbeat disco beat'])

    print('Output shape:', wav.shape)

    audio_write(
        'magnet_output',
        wav[0].cpu(),
        model.sample_rate,
        strategy='loudness',
        loudness_compressor=True
    )
    print('Done! Saved magnet_output.wav')

except Exception as e:
    print(f'MAGNeT failed: {type(e).__name__}: {e}')
    print('\nThis is expected - MAGNeT may require GPU/xformers (see issue #407)')
