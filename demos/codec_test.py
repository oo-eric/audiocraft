from audiocraft.models import CompressionModel
import torch
import scipy.io.wavfile as wavfile
import numpy as np

# Load encodec
model = CompressionModel.get_pretrained('facebook/encodec_32khz')
model.eval()

# Create a simple test tone (440Hz sine wave)
sr = 32000
t = np.linspace(0, 2, sr * 2)
tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Save original tone
tone_int16 = (tone * 32767).astype(np.int16)
wavfile.write('tone_original.wav', sr, tone_int16)
print('Saved tone_original.wav')

# Encode then decode through EnCodec
tone_tensor = torch.from_numpy(tone).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    codes, scale = model.encode(tone_tensor)
    decoded = model.decode(codes, scale)

# Save decoded tone
decoded_np = decoded[0, 0].numpy()
decoded_int16 = (np.clip(decoded_np, -1, 1) * 32767).astype(np.int16)
wavfile.write('tone_decoded.wav', sr, decoded_int16)
print('Saved tone_decoded.wav')

print('Listen to both files - they should sound like a 440Hz tone (A note)')
