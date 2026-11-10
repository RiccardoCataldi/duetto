#!/bin/bash
# Restart the duet: the model's server on the Jetson (or another machine), SSH tunnel, FluidSynth, main.py and the
# piano roll on the laptop. On another ssh host, with the left hand on a piano in the context, for example:
#   DUET_HOST=pc SERVER_ARGS="--model ~/models/music-medium-800k --left 0" ./run_duet.sh
#   DUET_HOST=pc SERVER_ARGS="--engine aria --checkpoint ~/models/aria-medium-gen.safetensors" ./run_duet.sh
cd "$(dirname "$0")"
LOG=${LOG_DIR:-/tmp}
HOST=${DUET_HOST:-jetson}                                            # an ssh host with this repo in ~/Ai-Duet
SERVER_ARGS=${SERVER_ARGS:---model ~/models/music-medium-800k}       # AMT by default

# the watcher of the old window first, or it would stop what we start below
[ -f "$LOG/duet-window.pid" ] && kill "$(cat "$LOG/duet-window.pid")" 2>/dev/null
pkill -INT -f "^\.venv/bin/python main\.py"; sleep 2; pkill -f "^\.venv/bin/python main\.py"
pkill -f "^\.venv/bin/python pianoroll\.py"
pkill -x timidity
pkill -x fluidsynth
pkill -f "^ssh -N .*-L 8765:localhost:8765"
sleep 1

# the model's code as it is here: the Jetson does not pull from git
mkdir -p generated
rsync -a --exclude __pycache__ music_transformer tests jetson_setup.sh generated "$HOST":Ai-Duet/ || echo "could not copy the code to $HOST"

# the clocks at their highest (jetson_setup.sh lets this run without a password), then the model's server
ssh "$HOST" 'if [ -x /usr/bin/jetson_clocks ]; then
    sudo -n /usr/bin/jetson_clocks 2>/dev/null || echo "clocks not maxed: run jetson_setup.sh there once"
  fi
  pkill -f "^python -u -m music_transformer\.server"; sleep 2
  cd ~/Ai-Duet || exit 1
  . .venv/bin/activate
  HF_HUB_OFFLINE=1 setsid nohup python -u -m music_transformer.server --corpus generated '"$SERVER_ARGS"' > ~/amt_server.log 2>&1 < /dev/null &
  for i in $(seq 60); do grep -q -E "listening|Traceback" ~/amt_server.log && break; sleep 2; done
  grep -E "listening|Traceback" ~/amt_server.log'

setsid nohup ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 8765:localhost:8765 "$HOST" > /dev/null 2>&1 < /dev/null &
# small audio buffers (-z 128 -c 2) keep the keyboard responsive; a big, dark room and a slow chorus warm up the sound
REVERB="-o synth.reverb.room-size=0.75 -o synth.reverb.damp=0.6 -o synth.reverb.width=1.0 -o synth.reverb.level=0.8"
CHORUS="-o synth.chorus.depth=6 -o synth.chorus.speed=0.25 -o synth.chorus.level=1.5"
# the instruments are FluidR3_GM's; a sampled grand sounds far more like a real piano, and is loaded over it when
# there is one (FluidSynth takes a preset from the soundfont loaded last): ~/sf2/SalamanderGrandPiano.sf2, or
# SOUNDFONT=path/to.sf2 ./run_duet.sh
GM=/usr/share/sounds/sf2/FluidR3_GM.sf2
SOUNDFONT=${SOUNDFONT:-$HOME/sf2/SalamanderGrandPiano.sf2}
[ -f "$SOUNDFONT" ] || SOUNDFONT=
setsid nohup fluidsynth -a pipewire -m alsa_seq -i -s -z 128 -c 2 -g 0.7 $REVERB $CHORUS "$GM" ${SOUNDFONT:+"$SOUNDFONT"} > "$LOG/fluidsynth.log" 2>&1 < /dev/null &
sleep 3
SYNTH=$(aconnect -o | awk -F'[ :]' '/FLUID Synth/ {print $2; exit}')
aconnect 14:0 "$SYNTH:0" && echo "Midi Through -> FluidSynth ($SYNTH:0)"

# your MIDI keyboard, the first one plugged in; without one, the laptop's keyboard in the piano roll
KEYSTEP=$(.venv/bin/python -c "import rtmidi; print(next((i for i, p in enumerate(rtmidi.MidiIn().get_ports()) if not any(n in p for n in ('Midi Through', 'FLUID', 'RtMidi', 'duetto'))), ''))")
echo "$KEYSTEP" | AMT_SERVER=localhost:8765 PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python main.py > "$LOG/duet.log" 2>&1 &
sleep 4
grep -E "chosen|output opened|Playing with" "$LOG/duet.log"

# the piano roll of you and the model (F: fullscreen); closing it stops the duet (main.py saves it) and the sound
setsid nohup bash -c '.venv/bin/python pianoroll.py > "$1/pianoroll.log" 2>&1
  pkill -INT -f "^\.venv/bin/python main\.py"; sleep 2; pkill -x fluidsynth' duet-window "$LOG" < /dev/null > /dev/null 2>&1 &
echo $! > "$LOG/duet-window.pid"
