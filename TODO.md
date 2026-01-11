# TODO - macOS ARM Support PR

## Before PR Submission

### Testing
- [x] MusicGen - works
- [x] AudioGen - works
- [ ] MAGNeT - fails with tensor shape mismatch, needs investigation (test script at demos/magnet_test.py)
- [x] JASCO - works

### Clean Up for Upstream
- [ ] Remove `demos/test.py` - personal test script
- [ ] Remove `demos/codec_test.py` - personal test script
- [ ] Remove `.python-version` - pyenv local file
- [ ] Remove `CLAUDE.md` - keep in fork only
- [ ] Remove `NOTES.md` - keep in fork only
- [ ] Remove `TODO.md` - keep in fork only

### Code Cleanup
- [ ] Review transformer.py - remove unused fallback code paths
- [ ] Consider simplifying warning messages (currently verbose)

### Documentation
- [ ] Add macOS ARM section to README.md
- [ ] Document that xformers is optional on macOS

### Optional
- [ ] Add CI smoke test that works without GPU
- [ ] Test on Linux without xformers to ensure backwards compatibility

## PR Description Draft

**Title:** Make xformers optional for macOS ARM support

**Description:**
This PR enables AudioCraft to run on macOS ARM64 (Apple Silicon) by making xformers optional.

**Changes:**
- Make xformers import optional in `transformer.py`
- Auto-disable `memory_efficient` attention when xformers unavailable
- Make xformers profiler optional in `profiler.py`

**Fixes/Addresses:**
- #230 - xFormers error on M1
- #43 - Apple Silicon M1 support request
- #31 - Apple Silicon feature request
- #13 - Running on M1 help
- #587 - CPU-only Docker support for macOS ARM64
- #362 - xformers/torch module conflict

**Tested on:**
- macOS ARM64 (Apple Silicon)
- MusicGen (small, medium)
- AudioGen (medium)
- JASCO
