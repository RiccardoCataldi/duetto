# duetto

[![tests](https://github.com/RiccardoCataldi/duetto/actions/workflows/tests.yml/badge.svg)](https://github.com/RiccardoCataldi/duetto/actions/workflows/tests.yml) [![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)](LICENSE)

![You in orange, the model in blue, on the piano roll](samples/duo/duetto.gif)

Play a MIDI keyboard and a transformer plays a second piano with you, in real time. It listens like a partner: your key, the chord you play, your pulse, your phrases. It comps while you play and takes the melody when you rest.

▶ **[Watch a full improvisation on YouTube](https://www.youtube.com/watch?v=0qvRMhN8qa4)**

https://github.com/user-attachments/assets/41a18857-2421-4c9a-b8b0-83cfdad1ff84

The model is the pretrained [Anticipatory Music Transformer](https://github.com/jthickstun/anticipation) (Lakh MIDI), running on a Jetson; the laptop handles MIDI, sound and the piano roll.

https://github.com/user-attachments/assets/bd7c846d-0ae0-4ff4-a644-22b97c01049c

Above: you play low chords, the model improvises over them (the **it solos** preset). Files in [samples/duo](samples/duo).

## Its memory

What you play becomes a corpus, in the manner of [Somax2](https://github.com/DYCI2/Somax2) (written anew from its published model, in [music_transformer/memory.py](music_transformer/memory.py)). Your notes wake peaks in that corpus, on three layers: your line, your harmony, its own line. Peaks move on with time, fade, and add up when they agree, in any key, and the strongest says what comes next and when. AMT plays it, as closely as `recall` asks: at 0 it only invents, at the top it quotes you. It also knows the harmony it expects next.

Silence comes the same way: the rests you play are in the corpus; `silence` leaves slices out; `timeout` is how long it plays on after your last note; with `sparse` on it plays only when it remembers. With nothing to remember it invents, in phrases of about `phrase` notes with a `breath` between them.

`./run_duet.sh` gives it your saved duos in `generated/` (your track of each); by hand, `python -m music_transformer.server --corpus generated`.

## Run it

```bash
./run_duet.sh                      # on the laptop
```

`run_duet.sh` restarts everything: it copies the model's code to the Jetson and maxes its clocks, then starts the model server there, the SSH tunnel, FluidSynth, `main.py` on the keyboard and the piano roll. Closing the piano roll stops the duo and saves it in `generated/`.

### Without a Jetson

Everything runs on one Linux laptop too. The server downloads the model and picks CUDA, Apple Silicon or the CPU (with the smaller model on the CPU):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m music_transformer.server                    # terminal 1: the model
fluidsynth -a pipewire -m alsa_seq -i -s /usr/share/sounds/sf2/FluidR3_GM.sf2 &
aconnect "Midi Through" "FLUID Synth"                            # the duo's sound
.venv/bin/python main.py                                         # terminal 2: pick your keyboard (Ctrl+C stops and saves)
.venv/bin/python pianoroll.py                                    # terminal 3: the piano roll and the knobs
```

No MIDI keyboard? A virtual one works: `sudo apt install vmpk`, open it, and pick it in `main.py`.

<details>
<summary>Setup</summary>

- **Jetson** (Orin Nano, JetPack 6): the repo in `~/Ai-Duet` with a venv made with `--system-site-packages` (NVIDIA's torch), the model as safetensors in `~/models/music-medium-800k`, power mode MAXN_SUPER, and `ssh jetson` working with a key. Then once `ssh -t jetson sudo bash Ai-Duet/jetson_setup.sh`: it lets `run_duet.sh` max the clocks, and frees the memory the model needs by running without a desktop and turning off the services the duo does not use (the script says how to undo it).
- **Laptop**: FluidSynth with FluidR3_GM, and a venv with `pip install mido python-rtmidi pygame imageio-ffmpeg`. A sampled grand at `~/sf2/SalamanderGrandPiano.sf2` ([Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html), or another with `SOUNDFONT=path/to.sf2 ./run_duet.sh`) plays the piano over FluidR3_GM's.
- **Code changes**: `./run_duet.sh` copies `music_transformer/`, `tests/` and `jetson_setup.sh` to the Jetson. The model's tests need torch: run `python -m unittest tests.test_duo_session tests.test_graphed` there.
- **Speed**: on CUDA the model feeds each token through a CUDA graph ([music_transformer/graphed.py](music_transformer/graphed.py)), the same GPT-2 about three times sooner than in eager mode.
- **Logs**: `~/amt_server.log` on the Jetson; `duet.log`, `fluidsynth.log`, `pianoroll.log` in `/tmp` on the laptop.

</details>

## Piano roll and knobs

![The piano roll with the knobs](samples/duo/pianoroll-knobs.png)

The top of the piano roll says what the model hears (key, chord, tempo) and what it is doing. Keys: F fullscreen, K show or hide the controls, Esc close.

Controls are sliders (drag them; hold Shift to drag finely) and steppers (click the arrows). The arrow keys or Ctrl and the wheel over a control turn it; the wheel alone scrolls the panel. A double click resets a control, and the bottom of the panel says what the control under the mouse does. A control is dimmed while what it shapes is not happening.

- **AI mind**
  - Presets: **duet**, **call & response**, **it comps** (it plays the harmony, you improvise), **it solos** (you play the harmony, it improvises), **free**
  - Musical controls: density, harmony, role, dialogue, groove, risk, focus, memory, key follow
  - Range: `lowest`, `highest` octave
  - Scale: `root` and `scale` force it into a scale (C major, D harmonic minor, F minor pentatonic...) instead of your key
  - Tempo: `bpm` fixes its tempo instead of following yours, `metronome` clicks it
- **Your sound**, **AI sound**: instrument, volume, pan, envelope, reverb and delay, in FluidSynth
- **Advanced**: every knob of the model, described in `KNOBS` in [music_transformer/duo.py](music_transformer/duo.py), where the defaults are too

Knobs and sound are kept in `~/.config/ai-duet/`.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): use it, change it and share it freely for anything noncommercial (personal, study, research, education, hobby). Commercial use is not allowed. The model itself is the [Anticipatory Music Transformer](https://github.com/jthickstun/anticipation), under its own license.
