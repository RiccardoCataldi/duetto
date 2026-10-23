

https://github.com/user-attachments/assets/d64728c6-4b59-4d35-81a4-7908be2604a1

# AI Music Generator

An interactive AI music generation system that responds to MIDI keyboard input in real-time. The system uses a transformer-based model to generate music based on your playing, creating a collaborative experience between human and AI.

## Features

- Real-time MIDI input processing
- AI-powered music generation based on your playing
- A real-time duo of two pianos: the model improvises with two hands of its own while you play. It listens like a partner: it follows your key and your modulations, hears the jazz chord you play (maj7, m7b5, sus, 6 told apart by the bass) and colours it with its 9th, 11th and 13th, keeps to your pulse, stays under you while you play and answers your phrases when you rest
- It plays with dynamics, as a pianist does: as loud as you play, softer while it accompanies you, its top voice singing over its inner voices and its left hand; and your sustain pedal sounds and is heard
- Different instruments for input and generated music
- MIDI file saving and playback capabilities

## Listen

The video at the top: a one-minute improvisation, played live on the Jetson duo: a human on piano (amber), the model on a warm electric piano (teal). The model heard it in C major.

- the video with sound, at the top of this page ([duo-c-major.mp4](samples/duo/duo-c-major.mp4) in the repo)
- [duo-c-major.ogg](samples/duo/duo-c-major.ogg): the audio alone (open *View raw* to play it in the browser)
- [duo-c-major.mid](samples/duo/duo-c-major.mid): the MIDI (track 1 is the human, track 2 the model)

The audio is the saved MIDI rendered with the same FluidSynth settings as `run_duet.sh`, and the video is the piano roll drawn over it. `sound.py bake` first puts in the sound the duo was played with (instruments, envelopes, reverb sends and the delay's echoes), which `main.py` saves next to it:
```bash
python sound.py bake generated/<time>.mid duo-baked.mid
eval "$(grep -E '^(REVERB|CHORUS)=' run_duet.sh)"
fluidsynth -ni -g 0.7 $REVERB $CHORUS -T oga -F duo.ogg /usr/share/sounds/sf2/FluidR3_GM.sf2 duo-baked.mid
python pianoroll.py video generated/<time>.mid --audio duo.ogg -o duo.mp4
```

## Piano roll

`pianoroll.py` shows the duo as you play: your notes and the model's rise from the keyboard, each in its own colour, and the keys light up while they sound. `run_duet.sh` opens it next to the duo (F: fullscreen, K: show or hide the knobs, Esc: close). It listens to what `main.py` sends to FluidSynth on Midi Through; to open it by hand, run `python pianoroll.py`.

![The piano roll with the knobs](samples/duo/pianoroll-knobs.png)

At the top it says what the model hears and does: your key, the chord you play, your tempo once it has found your pulse, and whether it is accompanying you, answering your phrase or playing on its own. Beside it are knobs (`knobs.py`) to turn while you play, in four tabs. Drag a knob up or down (hold Shift to turn it finely) or use the wheel over it; a double click puts it back, and hovering says what it does. A knob is dimmed while what it shapes is not happening: `dialogue` while you play, `role` while you rest, `groove` until it has found your pulse.

- **AI mind**: musical controls, each with its two ends; each turns some of the model's knobs on the Jetson along a line whose middle is the duo's default:

  | Control | From ... to | Turns |
  |---|---|---|
  | density | sparse ... dense | `min_step`, `space`, `max_rest` |
  | harmony | inside ... outside | `out_of_key`, `chord_tone`, `tension` |
  | role | accompanies ... leads (while you play) | `under` |
  | dialogue | ignores you ... answers you (when your phrase ends) | `answer`, `echo` |
  | groove | loose ... tight | `grid` |
  | risk | safe ... adventurous | `temperature`, `top_p` |
  | focus | on you ... on itself | `memory` |
  | key follow | steady ... quick | `stay` |

  and `lowest`, `highest`: the octaves it may play. The model's knobs are kept in `~/.config/ai-duet/mind.json` and given back to the server whenever it restarts; the server log shows each turn (`duo: min_step = 32`).
- **Your sound** and **AI sound** (`sound.py`, in FluidSynth on the laptop): the General MIDI instrument, volume, where it sits from left to right (pan: you on the left, the model on the right, as two pianos on a record), the envelope (attack, decay, sustain and release, as how many times longer than the instrument's own, through SoundFont NRPNs), the reverb send, and a delay: its time, how loud the first echo is (0 turns it off) and how much each echo keeps of the one before. FluidSynth has no delay, so the echoes are MIDI notes, played on channels 2 (yours) and 3 (the model's). The sound is kept in `~/.config/ai-duet/sound.json` across sessions.
- **Advanced**: every knob of the model on its own (see *Tuning by ear* below; with Aria, its own knobs, see *Duo with Aria*).

## Requirements

- Python 3.8+
- MIDI keyboard/controller
- MIDI output device (virtual or physical)
- TensorFlow (for the AI model)

## Dependencies

The project requires several Python packages listed in `requirements.txt`. See the Setup section for installation instructions.

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Connect your MIDI keyboard/controller to your computer
4. Make sure you have a MIDI output device configured
5. The AI uses the pretrained [Anticipatory Music Transformer](https://github.com/jthickstun/anticipation) (trained on the Lakh MIDI dataset). It is downloaded from Hugging Face on first run: `music-medium-800k` on GPU/Apple Silicon, `music-small-800k` on CPU. On an NVIDIA GPU it can be [Aria](#duo-with-aria-on-a-pc-with-a-gpu) instead, a model of expressive piano

## Usage

1. Start the model server (`python -m music_transformer.server`, or on a Jetson: see below)
2. Run the program (`AMT_SERVER` defaults to `localhost:8765`):
```bash
python main.py
```

3. Select your MIDI input device when prompted
4. Start playing on your MIDI keyboard: you play piano, the model plays a second piano at the same time
5. Press Ctrl+C to stop the program and save the duo to `generated/`

## Duo on a Jetson

The model runs on a Jetson Orin Nano (JetPack 6) and the laptop handles MIDI and sound; `run_duet.sh` connects the two.

**On the Jetson, once:**
- The repo in `~/Ai-Duet` with a venv `.venv` made with `--system-site-packages` (it uses NVIDIA's torch)
- The model converted to safetensors in `~/models/music-medium-800k` (transformers 5 refuses `.bin` files with torch < 2.6)
- Power mode MAXN_SUPER (it persists across reboots)
- `ssh jetson` works from the laptop with a key

**On the laptop, once:** FluidSynth with the `FluidR3_GM` soundfont (or, much closer to a real grand, a sampled piano such as the Salamander Grand Piano in SF2: `SOUNDFONT=path/to/it.sf2 ./run_duet.sh`), and the venv `.venv` of this repo with `pip install mido python-rtmidi pygame imageio-ffmpeg` (imageio-ffmpeg brings its own ffmpeg, for the videos).

**Every time the Jetson boots:**
```bash
ssh -t jetson sudo jetson_clocks   # the clocks go back to defaults after every reboot
./run_duet.sh                      # from this repo, on the laptop
```
`run_duet.sh` restarts everything: the server on the Jetson, the SSH tunnel on port 8765, FluidSynth, `main.py` on the KeyStep and the piano roll.

**Logs:** `~/amt_server.log` on the Jetson (`duo: started`, the key it hears you in each time it changes, then how many notes it heard and played, how long a note took to sample and how soon after your note its first one went out);
`duet.log` (at the end, how many of the model's notes came later than `LATE_S` and were skipped), `fluidsynth.log` and `pianoroll.log` on the laptop in `$LOG_DIR` (default `/tmp`).
If its first note after yours takes longer than `MARGIN` (150 ms) to go out, or many notes are skipped, the model is too slow for the duo: it answers you late.

**How it hears you:** each of your notes enters its context when you played it, even if notes of its own already on their way come after it. Its duration is a guess when you press the key; when the note stops sounding (when you let go, or the pedal comes up), `main.py` says how long it lasted (`off <onset> <pitch> <duration>`) and the model hears that from its next note on, as Aria learnt the pedal: as longer notes. From then on a chord you hold counts as sounding until you let go.

**Changing the code on the Jetson:** it does not pull from git; copy the files over, e.g.
`scp music_transformer/duo.py jetson:~/Ai-Duet/music_transformer/`, then run `./run_duet.sh` again.
The server now says which hand plays each note (`note <onset> <duration> <pitch> <hand>`): copy `music_transformer/` to the Jetson whenever you update `main.py` on the laptop.
The session tests need torch, so run them there: `python -m unittest tests.test_duo_session`.

**The model's sound** (top of `main.py`): `RING`, `RING_S` (how long its notes ring past what the model wrote), its dynamics (`SOFTER`: how much softer than you it plays, `ACCOMPANY`: softer still while you play, `LEFT_HAND`, `TOP` and `INNER`: its left hand and its inner voices under its top voice, `CONTOUR`: how much its line swells as it rises, `HUMAN`: how much each velocity varies), `GENERATED_CONTROLS` (volume, pan, reverb and chorus sends, until the piano roll sets its own); a sound saved before the duo became two pianos keeps its electric piano: double-click the **AI sound** knobs to put them back; the room itself is `REVERB` and `CHORUS` in `run_duet.sh`; the rest is in the **Your sound** and **AI sound** knobs.

**Tuning by ear:** turn the knobs in the **Advanced** tab of the piano roll while you play (they are `KNOBS` in `music_transformer/duo.py`, with their ranges); to change a default, edit it at the top of `music_transformer/duo.py`:
- `MIN_STEP`, `SPACE`: time between the model's onsets, when you rest and while you play
- `OUT_OF_KEY`, `CHORD_TONE`: how strongly it keeps to your key and leans on the chord you are playing
- `TENSION`: how much it colours your chord with its tensions (`TENSIONS` in `music_transformer/listener.py`: the 9th, #11 and 13th of a major seventh...); a tension is never taken away as out of key
- `HANDS` (the `hands` knob): one hand, or its line over a left hand of its own. The left hand is another instrument in the model's context (`LEFT`, a bass, or `--left` on the server), so the model hears two parts; it keeps under `LEFT_TOP` and the right hand over `RIGHT_BOTTOM`, with its own rhythm (`LEFT_STEP`) and chords of up to `LEFT_CHORD` notes. Both sound as the model's piano
- `MAX_REST`: the longest silence it may leave (longer lets it lay out, as a partner does)
- `UNDER`: how firmly it stays below you while you play
- `ANSWER`, `ECHO`: how long after your phrase it answers it, and how closely it follows your intervals
- `GRID`: how tightly its onsets sit on your pulse
- `STAY` (top of `music_transformer/listener.py`): how soon it believes you modulated (lower = sooner, more flicker)
- `NO_REPEAT`: the length of runs it may not repeat
- `MEMORY`: how much of its own playing it remembers (shorter = listens to you more)
- `temperature` and `top_p` in `DuoSession`: how adventurous its choices are
- `lowest`, `highest` (knobs only, in `KNOBS`): the octaves it may play, from C of the lowest to B of the highest (C4 is middle C); the piano roll draws them as a line over the keys

## Duo with Aria on a PC with a GPU

[Aria](https://github.com/EleutherAI/aria) (EleutherAI) is a model of expressive solo piano: about 1B parameters, trained on ~60k hours of piano recordings transcribed to MIDI. Where AMT plays one line and leaves the dynamics to `main.py`, Aria plays both its hands and says how hard it strikes each key; it hears your notes and its own as one performance, four hands on one keyboard, and is told the music is jazz (`--genre`). It needs an NVIDIA GPU with ~4 GB free (an RTX 3080 Ti is plenty); the Jetson Orin Nano is too slow for it.

**On the PC, once** (Python 3.11+, the repo in `~/Ai-Duet`, `ssh pc` working from the laptop with a key):
```bash
cd ~/Ai-Duet && python3 -m venv .venv && . .venv/bin/activate
pip install torch safetensors numpy                               # torch with CUDA (driver CUDA 12.8: --index-url https://download.pytorch.org/whl/cu128)
pip install git+https://github.com/EleutherAI/aria-utils.git
git clone https://github.com/EleutherAI/aria ~/aria && pip install --no-deps -e ~/aria   # --no-deps: mlx is for Macs
mkdir -p ~/models && wget -O ~/models/aria-medium-gen.safetensors \
  https://huggingface.co/loubb/aria-medium-base/resolve/main/model-gen.safetensors
```

**Every time:** from the laptop,
```bash
DUET_HOST=pc SERVER_ARGS="--engine aria --checkpoint ~/models/aria-medium-gen.safetensors" ./run_duet.sh
```
Over a VPN, the network's delay adds to the model's: the model plays `LOOKAHEAD` (1 s) ahead and a new note of its sounds no sooner than `MARGIN` (150 ms); if `ping pc` is above ~60 ms or jumps around, raise `MARGIN` in `music_transformer/aria_duo.py` and `LATE_S` in `main.py`. The server log says, when a duo ends, how fast it was: if its first note after yours goes out well within `MARGIN`, `MARGIN` can come down and it answers you sooner.

**Its knobs** (`KNOBS` in `music_transformer/aria_duo.py`, in the **Advanced** tab): Aria is a pianist already, so they nudge it less than AMT's and there are no rules against loops: `chord` (most notes it starts at once), `space` and `min_step` (time between its onsets while you play and when you rest), `out_of_key`, `chord_tone`, `tension`, `under`, `grid` (at its highest, 20, its own expressive timing), `max_rest`, `temperature` and `min_p` (how adventurous, as Aria's authors sample it), `lowest` and `highest`. The **AI mind** controls turn those it has.

**Its tests** need torch and aria-utils, not the model: `python -m unittest tests.test_aria_duo`.

## Instrument Selection

When changing instruments, enter a number between 0-127. Some interesting combinations to try:

- Keyboard: Piano (0), Generated: Strings (48)
- Keyboard: Electric Piano (4), Generated: Synth Pad (88)
- Keyboard: Acoustic Guitar (24), Generated: Choir (52)
- Keyboard: Trumpet (56), Generated: Synth Brass (62)

The program supports all General MIDI instruments (0-127), including:
- Pianos and Keyboards (0-7)
- Chromatic Percussion (8-15)
- Organs (16-23)
- Guitars (24-31)
- Basses (32-39)
- Strings (40-47)
- Ensembles (48-55)
- Brass (56-63)
- Reeds (64-71)
- Pipes (72-79)
- Synth Leads (80-87)
- Synth Pads (88-95)
- Synth Effects (96-103)
- Ethnic (104-111)
- Percussive (112-119)
- Sound Effects (120-127)

## Project Structure

```
ai-music/
├── CK_rec/              # MIDI recording and playback module
├── music_transformer/   # Music generation model
├── trained_models/      # Place for your trained models
├── generated/           # Directory for saved generated music
├── main.py             # Main application
├── pianoroll.py        # Piano roll of the duo, live or as a video
├── knobs.py            # The knobs beside the live piano roll
├── sound.py            # Your sound and the model's: instruments, envelope, reverb, delay
└── requirements.txt    # Project dependencies
```

## MIDI Configuration

The program supports:
- Multiple MIDI input devices
- MIDI output for playback
- Real-time MIDI message processing
- MIDI file saving and loading
- Different instruments on separate MIDI channels
- Program change messages for instrument selection

## Troubleshooting

If you encounter issues:
1. Check your MIDI device connections
2. Verify the model file exists in the correct location
3. Ensure all dependencies are installed
4. Check the console output for error messages
5. Make sure your MIDI output device is properly configured

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Feel free to submit issues and enhancement requests!
