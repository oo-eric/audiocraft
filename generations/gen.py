"""Run test generations and save them for listening.

Each run writes to generations/runs/<timestamp>_<label>/: one WAV per prompt and variant, named
so the variants of a prompt sort next to each other, plus manifest.json with the settings,
timings and environment. Every variant of a prompt uses the same seed, so A/B pairs differ only
by the variant.

Examples:
    python generations/gen.py --label fp16-ab --variants default half
    python generations/gen.py --model facebook/musicgen-small --prompt "sea shanty" --duration 5

Variants:
    default  LM as loaded (on MPS: fp16 weights, fp32 math)
    half     model.lm.half() (fp16 math)
    float    model.lm.float() (fp32 weights and math)
"""
import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch  # noqa: E402

from audiocraft.data.audio import audio_write  # noqa: E402
from audiocraft.models import AudioGen, MusicGen  # noqa: E402

DEFAULT_PROMPTS = [
    'lo-fi hip hop beats to study to',
    'solo acoustic guitar, fingerpicked folk ballad',
    'orchestral film score with soaring strings and brass, epic',
    'driving techno with a heavy kick and acid bassline',
    'smoky late-night jazz trio, upright bass and brushed drums',
]
VARIANTS = {
    'default': lambda lm: lm,
    'half': lambda lm: lm.half(),
    'float': lambda lm: lm.float(),
}


def slug(text: str, n: int = 40) -> str:
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:n].rstrip('-')


def sync(device: str):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='facebook/musicgen-medium')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--variants', nargs='+', default=['default'], choices=list(VARIANTS))
    parser.add_argument('--prompt', action='append', help='repeatable; defaults to a built-in genre set')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=250)
    parser.add_argument('--cfg-coef', type=float, default=3.0)
    parser.add_argument('--label', default='', help='appended to the run directory name')
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = Path(__file__).parent / 'runs' / (f'{stamp}_{slug(args.label)}' if args.label else stamp)
    run_dir.mkdir(parents=True)

    cls = AudioGen if 'audiogen' in args.model else MusicGen
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    manifest = {'model': args.model, 'device': args.device, 'duration': args.duration, 'seed': args.seed,
                'top_k': args.top_k, 'cfg_coef': args.cfg_coef, 'torch': torch.__version__,
                'commit': commit, 'outputs': []}

    for variant in args.variants:
        # Variants cast the LM in place, so each one starts from a freshly loaded model.
        model = cls.get_pretrained(args.model, device=args.device)
        model.set_generation_params(duration=args.duration, use_sampling=True, top_k=args.top_k,
                                    cfg_coef=args.cfg_coef)
        VARIANTS[variant](model.lm)
        torch.manual_seed(args.seed)
        model.generate(['warmup'], progress=False)  # keep one-time kernel setup out of the timings
        for i, prompt in enumerate(prompts):
            torch.manual_seed(args.seed + i)
            sync(args.device)
            start = time.time()
            wav = model.generate([prompt], progress=False)
            sync(args.device)
            elapsed = time.time() - start
            w = wav[0].float().cpu()
            stem = f'{i:02d}_{slug(prompt)}__{variant}'
            audio_write(str(run_dir / stem), w, model.sample_rate, strategy='loudness',
                        loudness_headroom_db=16, loudness_compressor=True)
            entry = {'file': stem + '.wav', 'prompt': prompt, 'variant': variant,
                     'seed': args.seed + i, 'seconds': round(elapsed, 2),
                     'nan': bool(torch.isnan(w).any())}
            manifest['outputs'].append(entry)
            print(f'{stem}: {elapsed:.1f}s')

    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'\nWrote {len(manifest["outputs"])} files to {run_dir}')


if __name__ == '__main__':
    main()
