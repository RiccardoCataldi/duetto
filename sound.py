"""
The sound of the duo in FluidSynth, yours and the model's three voices (its solo, its chords, its bass: a band, each
voice on an instrument of its own): instrument, volume, envelope (ADSR), reverb and a delay.
The piano roll's knobs turn it while you play (see knobs.py) and it is kept in SOUND_FILE; main.py saves a copy
next to each duo, and a saved duo is rendered with it:

    python sound.py bake generated/<time>.mid duo.mid    # the duo with its sound, ready for fluidsynth -F

FluidSynth has no delay, so the delay is played as MIDI echoes on a channel of their own (ECHO), quieter each time.
The envelope offsets the instrument's own through SoundFont NRPNs, so its knobs read as how many times longer.
"""
import argparse
import heapq
import json
import os
import threading
import time
from collections import Counter

import mido

YOU, AI = 0, 1                       # MIDI channels, as main.py plays them: you, and the model's solo
CHORDS, BASS = 4, 5                  # its chords and its bass
PLAYERS = (YOU, AI, CHORDS, BASS)
VOICES = {'solo': AI, 'chords': CHORDS, 'bass': BASS}   # the channel of each of the model's voices (server.py's)
ECHO = {YOU: 2, AI: 3, CHORDS: 6, BASS: 7}   # the channel each one's echoes play on (9 is the metronome's)
SOUND_FILE = os.path.expanduser('~/.config/ai-duet/sound.json')
MAX_ECHOES = 12
QUIETEST = 4                         # velocity under which an echo is not played

# name: default for you, default for the model's voices, lowest, highest, unit, what it does
SOUND = {
    'instrument': (0, 0, 0, 127, 'program', "the General MIDI instrument"),
    'volume': (100, 100, 0, 127, 'level', "how loud it plays"),
    'pan': (40, 88, 0, 127, 'pan', "where it sits, from left to right: two pianos, as on a record"),
    'attack': (0, 0, -2400, 4800, 'times', "how slowly a note starts, against the instrument's own"),
    'decay': (0, 0, -2400, 4800, 'times', "how slowly a held note falls to its sustain"),
    'sustain': (0, 0, 0, 600, 'amount', "how much quieter a held note settles (an organ more than a piano, which dies away anyway)"),
    'release': (0, 0, -2400, 4800, 'times', "how long a note rings after it is let go"),
    'reverb': (40, 70, 0, 127, 'level', "how much of it goes to the reverb"),
    'delay': (350, 350, 50, 1000, 'ms', "the time between its echoes"),
    'echo': (0.0, 0.0, 0.0, 1.0, 'level', "how loud the first echo is (0: no delay)"),
    'feedback': (0.4, 0.4, 0.0, 0.85, 'level', "how much each echo keeps of the one before"),
}
# where its chords and its bass differ from its solo: a piano a little behind it, a double bass in the middle
THEIRS = {CHORDS: {'volume': 90, 'pan': 76}, BASS: {'instrument': 32, 'volume': 115, 'pan': 64, 'reverb': 30}}
# bands: name, the instruments of its solo, its chords and its bass
BANDS = [('piano trio', (0, 0, 32)), ('vibes', (11, 0, 32)), ('two pianos', (0, 0, 0))]
ENVELOPE = {'attack': 34, 'decay': 36, 'sustain': 37, 'release': 38}   # SoundFont generators of the volume envelope
NRPN_SCALE = {'attack': 2, 'decay': 2, 'sustain': 1, 'release': 2}     # FluidSynth multiplies NRPN data by these

GM = ["Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1",
      "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
      "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
      "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Nylon Guitar", "Steel Guitar", "Jazz Guitar",
      "Clean Guitar", "Muted Guitar", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass",
      "Finger Bass", "Pick Bass", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
      "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
      "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs",
      "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section",
      "Synth Brass 1", "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn",
      "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle",
      "Ocarina", "Square Lead", "Saw Lead", "Calliope Lead", "Chiff Lead", "Charang Lead", "Voice Lead", "Fifths Lead",
      "Bass + Lead", "New Age Pad", "Warm Pad", "Polysynth Pad", "Choir Pad", "Bowed Pad", "Metallic Pad", "Halo Pad",
      "Sweep Pad", "Rain", "Soundtrack", "Crystal", "Atmosphere", "Brightness", "Goblins", "Echoes", "Sci-fi", "Sitar",
      "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums",
      "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise", "Breath Noise",
      "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause", "Gunshot"]


def defaults():
    return {player: {**{name: spec[player != YOU] for name, spec in SOUND.items()}, **THEIRS.get(player, {})}
            for player in PLAYERS}


def load(path=SOUND_FILE):
    """The sound as last turned, or the defaults."""
    sound = defaults()
    try:
        with open(path) as f:
            saved = json.load(f)
        for player in PLAYERS:
            sound[player].update({k: v for k, v in saved.get(str(player), {}).items() if k in SOUND})
    except (OSError, ValueError):
        pass
    return sound


def save(sound, path=SOUND_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({str(player): values for player, values in sound.items()}, f, indent=1)


def messages(player, name, value):
    """The MIDI messages that set `name` for `player`, on its channel and its echoes' (none for the delay itself)."""
    out = []
    for channel in (player, ECHO[player]):
        if name == 'instrument':
            out.append([0xC0 | channel, int(value)])
        elif name == 'volume':
            out.append([0xB0 | channel, 7, int(value)])
        elif name == 'pan':
            out.append([0xB0 | channel, 10, int(value)])
        elif name == 'reverb':
            out.append([0xB0 | channel, 91, int(value)])
        elif name in ENVELOPE:
            data = 8192 + round(value / NRPN_SCALE[name])
            out += [[0xB0 | channel, 99, 120], [0xB0 | channel, 98, ENVELOPE[name]],
                    [0xB0 | channel, 6, data >> 7], [0xB0 | channel, 38, data & 127]]
    return out


def setup(sound):
    """All the messages that put FluidSynth in `sound`."""
    return [m for player, values in sound.items() for name, value in values.items() for m in messages(player, name, value)]


class Echoes:
    """Plans the echoes of each note as it is played, and lets a note off through only when no echo still holds."""

    def __init__(self):
        self.planned = {}          # (channel, pitch) -> (delay, count) of the echoes of the note now held
        self.holding = Counter()   # (echo channel, pitch) -> echoes sounding

    def plan(self, message, now, sound):
        """The echoes of `message` (yours or the model's note): [(when, message)]."""
        if len(message) < 3 or message[0] & 0x0F not in ECHO:
            return []
        kind, channel, pitch, velocity = message[0] & 0xF0, message[0] & 0x0F, message[1], message[2]
        if kind == 0x90 and velocity > 0:
            values = sound[channel]
            delay, level, levels = values['delay'] / 1000, velocity * values['echo'], []
            while level >= QUIETEST and len(levels) < MAX_ECHOES:
                levels.append(min(round(level), 127))
                level *= values['feedback']
            self.planned[(channel, pitch)] = (delay, len(levels))
            return [(now + k * delay, [0x90 | ECHO[channel], pitch, v]) for k, v in enumerate(levels, 1)]
        if kind in (0x80, 0x90):
            delay, count = self.planned.pop((channel, pitch), (0, 0))
            return [(now + k * delay, [0x80 | ECHO[channel], pitch, 0]) for k in range(1, count + 1)]
        return []

    def gate(self, message):
        """`message` if it should reach FluidSynth: a note off of an echo only once no other echo holds the key."""
        key = (message[0] & 0x0F, message[1])
        if message[0] & 0xF0 == 0x90:
            self.holding[key] += 1
            return message
        self.holding[key] -= 1
        if self.holding[key] > 0:
            return None
        del self.holding[key]
        return message


class Engine:
    """Puts FluidSynth in the sound and plays the echoes, live: `send` takes one raw MIDI message."""

    def __init__(self, send, path=SOUND_FILE):
        self.send, self.path = send, path
        self.sound = load(path)
        self.echoes = Echoes()
        self.due = []              # heap of (when, n, message)
        self.n = 0
        self.lock = threading.Condition()
        for message in setup(self.sound):
            send(message)
        threading.Thread(target=self._play, daemon=True).start()

    def set(self, player, name, value):
        spec = SOUND[name]
        value = round(value) if isinstance(spec[0], int) else float(value)
        self.sound[player][name] = min(max(value, spec[2]), spec[3])
        for message in messages(player, name, self.sound[player][name]):
            self.send(message)
        save(self.sound, self.path)

    def heard(self, message, now):
        """A note main.py played; its echoes follow."""
        with self.lock:
            for when, echo in self.echoes.plan(message, now, self.sound):
                heapq.heappush(self.due, (when, self.n, echo))
                self.n += 1
            self.lock.notify()

    def _play(self):
        with self.lock:
            while True:
                wait = self.due[0][0] - time.monotonic() if self.due else None
                if wait is None or wait > 0:
                    self.lock.wait(wait)
                    continue
                _, _, message = heapq.heappop(self.due)
                message = self.echoes.gate(message)
                if message:
                    self.send(message)


def bake(path, out, sound):
    """A saved duo with `sound`: its setup first, then your notes and the model's with their echoes."""
    mid = mido.MidiFile(path)
    events, now, echoes = [], 0.0, Echoes()
    for message in mid:
        now += message.time
        if message.type in ('note_on', 'note_off'):
            raw = message.bytes()
            events.append((now, raw))
            events += echoes.plan(raw, now, sound)
        elif message.type == 'control_change' and message.control not in (7, 10, 91):
            events.append((now, message.bytes()))
    played, gate = [], Echoes()
    for when, raw in sorted(events, key=lambda e: e[0]):
        if raw[0] & 0x0F in ECHO.values() and raw[0] & 0xF0 in (0x80, 0x90):
            raw = gate.gate(raw)
        if raw:
            played.append((when, raw))
    baked = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    track = mido.MidiTrack([mido.Message.from_bytes(m) for m in setup(sound)])
    ticks = lambda seconds: round(mido.second2tick(seconds, baked.ticks_per_beat, mido.bpm2tempo(120)))
    last = 0
    for when, raw in played:
        track.append(mido.Message.from_bytes(raw, time=ticks(when) - last))
        last = ticks(when)
    track.append(mido.MetaMessage('end_of_track', time=ticks(3)))
    baked.tracks.append(track)
    baked.save(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('bake',))
    parser.add_argument('midi', help='the saved duo (generated/<time>.mid)')
    parser.add_argument('out', help='the duo with its sound')
    parser.add_argument('--sound', help='the sound to use (default: the one saved next to the duo, or the current one)')
    args = parser.parse_args()
    beside = os.path.splitext(args.midi)[0] + '.sound.json'
    bake(args.midi, args.out, load(args.sound or (beside if os.path.exists(beside) else SOUND_FILE)))
    print(f"Duo with its sound saved as {args.out}")
