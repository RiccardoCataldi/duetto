import math
import os
import random
import signal
import socket
import statistics
import sys
import threading
import time
from collections import deque

import mido
import rtmidi

import sound
from CK_rec.setup import Setup
from CK_rec.rec_classes import CK_rec

LATE_S = 0.15            # a note of the model that arrives later than this is skipped, not played out of time
GUESS_S = 0.3            # how long we guess your note will last before you have played any
INPUT_PROGRAM = 0        # Acoustic Grand Piano
GENERATED_PROGRAM = 0    # Acoustic Grand Piano too: a duo of two pianos, you on the left and the model on the right
INPUT_CONTROLS = ((10, 40),)                                  # pan
GENERATED_CONTROLS = ((7, 100), (10, 88), (91, 70), (93, 0))  # volume, pan, reverb send, chorus send
PEDAL = 64               # the sustain pedal's control
KEYS = 'duetto keys'     # our MIDI port for the laptop's keyboard (pianoroll.py plays into it)
PLUG_S = 1.0             # how often we look whether your keyboard was unplugged or plugged back in
RING = 1.6               # its notes ring this much longer than the model wrote them, as if with a light pedal
RING_S = 0.4             # and at least this many seconds more
MELODIC_S = 0.5          # how often we look whether you made a voice melodic, or gave it a pedal (sound.py)
PEDAL_S = 4.0            # with its pedal, a chord rings on until the next, and this long at most
CHORD_S = 0.08           # notes this close are struck together: one chord
# its dynamics, shaped as a pianist shapes them; it follows how loud you play (LEVEL: how much your last note counts)
LEVEL = 0.3
QUIETEST = 80            # the softest it follows you down to: some keyboards never send more than 70
SOFTER = 0.8             # it plays a little softer than you
ACCOMPANY = 0.85         # and softer still while you play
LEFT_HAND = 0.85         # its left hand under its right
TOP, INNER = 1.08, 0.82  # in a chord the top voice sings and the inner voices fall back
CONTOUR = 0.2            # its line swells as it rises and falls back as it descends: this much from lowest to highest
CONTOUR_NOTES = 16       # its last notes the contour is measured against
# and its solo's phrases have a shape: they begin softly, their held notes fall away, the notes off the beat are lighter
BEGINS = (0.88, 0.94)    # how loud the first and the second note of a phrase are, of what they would be
PHRASE_BEATS = 1.5       # beats of silence before a note that make it a phrase's first
HELD = 0.9               # a note a beat long or more, after the phrase's first: where it comes to rest
GHOST = 0.92             # a note off the beat, shorter than a beat
HUMAN = 4                # velocity a little different every time, as a hand never strikes twice the same
# the metronome of the tune (the bpm and metronome knobs): General MIDI percussion, a bar of BAR beats
DRUMS = 9
CLICK = (76, 77)         # hi wood block on the first beat of a bar, low wood block on the others
CLICK_VELOCITY = 70
BAR = 4


def dynamics(pitch, hand, together, line, level, accompanying, rng=random, phrase=None):
    """The velocity of one of the model's notes, of its voice `hand` ('solo', 'chords' or 'bass'). `together`: the
    pitches that voice starts with it, `line`: its solo's last pitches, `level`: how loud you play, `accompanying`:
    whether you are playing. `phrase`, for a note of its solo: (the notes of its phrase before it, the beats it
    lasts, how far into its beat it begins, from 0 to 1): a phrase begins softly, comes to rest on a held note that
    falls away, and its notes off the beat are lighter."""
    velocity = max(level, QUIETEST) * SOFTER * (ACCOMPANY if accompanying else 1)
    if hand in ('left', 'bass'):
        velocity *= LEFT_HAND
    elif len(together) > 1:
        velocity *= TOP if pitch == max(together) else INNER
    if hand == 'solo' and len(line) >= 4 and max(line) > min(line):
        height = min(max((pitch - min(line)) / (max(line) - min(line)), 0), 1)
        velocity *= 1 + CONTOUR * (height - 0.5)
    if hand == 'solo' and phrase:
        before, lasts, into = phrase
        if before < len(BEGINS):
            velocity *= BEGINS[before]
        elif lasts >= 1:
            velocity *= HELD
        elif 0.2 < into < 0.9:
            velocity *= GHOST
    velocity += rng.uniform(-HUMAN, HUMAN)
    return min(max(round(velocity), 1), 127)


class Duo:
    """
    You and the model play at the same time. Every key you press goes to the model straight away, and how long it
    lasted once it stops sounding; it sends back its own line a little ahead of time, and redoes the part that has
    not sounded yet whenever you play.
    `send` takes one protocol line (see music_transformer/server.py); `started` is when the connection opened.
    """

    def __init__(self, send, midi_rec, started, rng=random, sound_file=sound.SOUND_FILE):
        self.send = send
        self.sound_file = sound_file   # where the piano roll keeps the sound: which voices are melodic
        self.melodic, self.pedals, self.checked, self.stamp = set(), set(), float('-inf'), False
        self.midi_rec = midi_rec
        self.started = started
        self.rng = rng
        self.lock = threading.Lock()
        self.notes = []                # (onset, offset, pitch, velocity, the model's channel or False for yours) in
                                       # time.monotonic() seconds
        self.held = {}                 # pitch -> (onset, velocity), the keys you hold down
        self.pedaled = {}              # pitch -> (onset, velocity), the keys you let go that the pedal holds
        self.pedal = False
        self.pedalings = []            # (time, value) of your sustain pedal
        self.durations = deque(maxlen=16)
        self.level = 80                # how loud you play
        self.played_at = None          # when you last pressed a key
        self.pending = []              # the model's notes not started yet: (onset, duration, pitch, voice)
        self.sounding = {}             # (channel, pitch) -> (onset, offset, velocity) of the model's notes: each of
                                       # its voices (solo, chords, bass) on its own channel, its own instrument
        self.line = deque(maxlen=CONTOUR_NOTES)   # its solo's last pitches
        self.sung = (None, 0)          # when its solo's last note began, and the notes of its phrase so far
        self.late = 0                  # the model's notes that came later than LATE_S, skipped
        self.bpm, self.metronome = 0, False   # the tempo the model keeps to and whether we click it
        self.click_at = None           # when the metronome's next beat falls
        self.bar_at = started          # when a bar of the metronome begins: its beats are counted from there

    def note_on(self, pitch, velocity, now):
        with self.lock:
            lines = self._sounded(pitch, now)   # a key the pedal still holds, struck again
            self.held[pitch] = (now, velocity)
            self.level += LEVEL * (velocity - self.level)
            self.played_at = now
            guess = statistics.median(self.durations) if self.durations else GUESS_S
        for line in lines + [f"on {now - self.started:.3f} {guess:.3f} {pitch} {velocity}\n"]:
            self.send(line)

    def note_off(self, pitch, now):
        with self.lock:
            if pitch not in self.held:
                return
            self.pedaled[pitch] = self.held.pop(pitch)
            lines = [] if self.pedal else self._sounded(pitch, now)
        for line in lines:
            self.send(line)

    def sustain(self, value, now):
        """Your sustain pedal: while it is down, the keys you let go ring on until it comes up."""
        with self.lock:
            self.pedalings.append((now, value))
            self.pedal = value >= 64
            lines = [] if self.pedal else [line for pitch in list(self.pedaled) for line in self._sounded(pitch, now)]
        for line in lines:
            self.send(line)

    def _sounded(self, pitch, now):
        """Your note on `pitch`, held by the pedal, has stopped sounding at `now`: the lines that tell the model how
        long it lasted (the server guessed when you pressed it)."""
        if pitch not in self.pedaled:
            return []
        onset, velocity = self.pedaled.pop(pitch)
        self.durations.append(now - onset)   # how long your notes sound, the pedal's ring and all
        self.notes.append((onset, now, pitch, velocity, False))
        return [f"off {onset - self.started:.3f} {pitch} {now - onset:.3f}\n"]

    def receive(self, line):
        kind, *values = line.split()
        with self.lock:
            if kind == 'note':
                onset, duration, pitch, *how = values   # its voice, if the model says
                voice = how[0] if how and how[0] in sound.VOICES else 'solo'
                self.pending.append((self.started + float(onset), float(duration), int(pitch), voice))
            elif kind == 'cut':
                cut = self.started + float(values[0])
                self.pending = [n for n in self.pending if n[0] < cut]
            elif kind == 'tempo':
                self.bpm, self.metronome, self.click_at = float(values[0]), values[1] == '1', None
                self.bar_at = self.started + (float(values[2]) if len(values) > 2 else 0)

    def tick(self, now):
        """Starts and stops the model's notes that are due, and clicks the metronome."""
        with self.lock:
            if self.metronome and self.bpm:
                self._click(now)
            for key, (onset, offset, _) in list(self.sounding.items()):
                if offset <= now:
                    self._release(key, now)
            due = [n for n in self.pending if n[0] <= now]
            self.pending = [n for n in self.pending if n[0] > now]
            accompanying = self.played_at is not None and now - self.played_at < 1.0
            melodic, pedals = self._melodic(now), self.pedals
            for onset, duration, pitch, hand in sorted(due, key=lambda n: n[:3]):
                if now - onset > LATE_S:
                    self.late += 1
                    continue
                channel = sound.VOICES[hand]
                together = [n[2] for n in due if n[0] == onset and n[3] == hand]
                # its bass, as a double bass, and a melodic voice, as a flute: one note at a time, the highest of
                # those struck together; each stops the one before, and none rings on as with a pedal
                one = hand == 'bass' or channel in melodic
                if one and pitch != max(together):
                    continue
                # with its pedal the chord before rings until this one is struck: then the pedal is changed
                pedal = channel in pedals and not one
                for key, (struck, *_) in list(self.sounding.items()):
                    if key[0] == channel and (one or key[1] == pitch or pedal and struck < now - CHORD_S):
                        self._release(key, now)
                phrase = None
                if hand == 'solo' and self.bpm:   # where the note is in its phrase and in its beat
                    beat = 60 / self.bpm
                    last, before = self.sung
                    before = 0 if last is None or onset - last > PHRASE_BEATS * beat else before
                    phrase = (before, duration / beat, (onset - self.bar_at) / beat % 1)
                    self.sung = (onset, before + (onset != last))
                velocity = dynamics(pitch, hand, together, self.line, self.level, accompanying, self.rng, phrase)
                if hand == 'solo':
                    self.line.append(pitch)
                ring = duration if one else max(duration * RING, duration + RING_S, PEDAL_S if pedal else 0)
                self.midi_rec.play_midi_message([0x90, pitch, velocity], is_generated=True, channel=channel)
                self.sounding[(channel, pitch)] = (now, now + ring, velocity)

    def _melodic(self, now):
        """The channels of its voices that play one note at a time (sound.py's `melodic`), as the piano roll's knobs
        left them in the sound file: read again, every MELODIC_S, when it has changed; and, in `pedals`, those that
        have their sustain pedal down (`pedal`)."""
        if now >= self.checked:
            self.checked = now + MELODIC_S
            try:
                stamp = os.path.getmtime(self.sound_file)
            except OSError:
                stamp = None
            if stamp != self.stamp:
                self.stamp, values = stamp, sound.load(self.sound_file)
                self.melodic = {player for player in sound.MELODIC_VOICES if values[player]['melodic']}
                self.pedals = {player for player in sound.MELODIC_VOICES if values[player]['pedal']}
        return self.melodic

    def _click(self, now):
        """Clicks the beat that is due: the model's beats fall on k * 60 / bpm seconds after the connection opened."""
        beat = 60 / self.bpm
        if self.click_at is not None and now >= self.click_at:
            if now - self.click_at < LATE_S:
                click = CLICK[round((self.click_at - self.bar_at) / beat) % BAR > 0]
                self.midi_rec.play_midi_message([0x90, click, CLICK_VELOCITY], channel=DRUMS)
                self.midi_rec.play_midi_message([0x80, click, 0], channel=DRUMS)
            self.click_at = None
        if self.click_at is None:
            self.click_at = self.bar_at + math.floor((now - self.bar_at) / beat + 1) * beat

    def stop(self, now):
        with self.lock:
            self.pending = []
            for pitch in list(self.pedaled):
                self._sounded(pitch, now)
            for key in list(self.sounding):
                self._release(key, now)

    def _release(self, key, now):
        channel, pitch = key
        onset, _, velocity = self.sounding.pop(key)
        self.midi_rec.play_midi_message([0x80, pitch, 0], is_generated=True, channel=channel)
        self.notes.append((onset, now, pitch, velocity, channel))

    def save(self, path):
        if not self.notes:
            return
        mid = mido.MidiFile()
        ticks = lambda seconds: int(mido.second2tick(seconds, mid.ticks_per_beat, mido.bpm2tempo(120)))
        origin = min(n[0] for n in self.notes)
        played_with = sound.load()   # a track for you and one for each of its voices, on the instrument it sounded as
        for channel in sound.PLAYERS:
            generated, program = channel != sound.YOU, played_with[channel]['instrument']
            messages = []
            for onset, offset, pitch, velocity, voice in self.notes:
                if (voice or sound.YOU) == channel:
                    messages.append((onset - origin, 'note_on', pitch, velocity))
                    messages.append((offset - origin, 'note_off', pitch, 0))
            if not messages and channel not in (sound.YOU, sound.AI):
                continue
            track = mido.MidiTrack([mido.Message('program_change', channel=channel, program=program)])
            track += [mido.Message('control_change', channel=channel, control=c, value=v)
                      for c, v in (GENERATED_CONTROLS if generated else INPUT_CONTROLS)]
            if not generated:
                messages += [(at - origin, 'pedal', PEDAL, value) for at, value in self.pedalings if at >= origin]
            last = 0
            for at, kind, pitch, velocity in sorted(messages):
                if kind == 'pedal':
                    track.append(mido.Message('control_change', channel=channel, control=pitch, value=velocity,
                                              time=ticks(at) - last))
                else:
                    track.append(mido.Message(kind, channel=channel, note=pitch, velocity=velocity, time=ticks(at) - last))
                last = ticks(at)
            mid.tracks.append(track)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mid.save(path)
        # the sound it was played with, as the piano roll's knobs left it: python sound.py bake renders the duo with it
        sound.save(played_with, os.path.splitext(path)[0] + '.sound.json')
        print(f"Duet saved as {path}")


def connect(address):
    host, port = address.rsplit(':', 1)
    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall(b"hello\n")   # the duo begins now, not at your first key: over a tune it may start by itself
    print(f"Playing with the model's server at {address} (its log says which model)")
    return sock, time.monotonic()


if __name__ == "__main__":
    codeK = Setup()
    myPort = codeK.perform_setup()
    keyboard = None                # no MIDI keyboard: the laptop's, in the piano roll
    if myPort is not None:
        codeK.open_port(myPort)
        keyboard = codeK.port_name(myPort)

    midiRec = CK_rec()
    midiRec.set_input_instrument(INPUT_PROGRAM)
    midiRec.set_generated_instrument(GENERATED_PROGRAM)
    for channel, controls in ((midiRec.input_channel, INPUT_CONTROLS), (midiRec.generated_channel, GENERATED_CONTROLS)):
        for control, value in controls:
            midiRec.play_midi_message([0xB0 | channel, control, value])
    for message in sound.setup(sound.load()):   # each voice on its instrument, even with no piano roll beside us
        midiRec.play_midi_message(message)
    sock, started = connect(os.environ.get('AMT_SERVER', 'localhost:8765'))
    duo = Duo(lambda line: sock.sendall(line.encode()), midiRec, started)

    def listen():
        for line in sock.makefile('r'):
            duo.receive(line)
        print("The model's server closed the duo")

    threading.Thread(target=listen, daemon=True).start()

    def on_midi(event, data=None):
        message, _ = event
        if len(message) < 3:
            return
        kind, pitch, velocity = message[0] & 0xF0, message[1], message[2]
        if kind == 0x90 and velocity > 0:
            midiRec.play_midi_message(message)
            duo.note_on(pitch, velocity, time.monotonic())
        elif kind in (0x80, 0x90):
            midiRec.play_midi_message(message)
            duo.note_off(pitch, time.monotonic())
        elif kind == 0xB0 and pitch == PEDAL:   # your sustain pedal, on your channel
            midiRec.play_midi_message([0xB0 | midiRec.input_channel, PEDAL, velocity])
            duo.sustain(velocity, time.monotonic())

    if keyboard:
        codeK.set_callback(on_midi)
    keys = rtmidi.MidiIn()         # the laptop's keyboard, which the piano roll plays into this port of ours (M)
    keys.open_virtual_port(KEYS)
    keys.set_callback(on_midi)
    # run_duet.sh starts us in the background, where SIGINT is ignored: stop and save on SIGINT or SIGTERM anyway
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.default_int_handler)
    print("Play: the model improvises with you. Ctrl+C to stop.")

    plugged, looked = True, time.monotonic()
    try:
        while True:
            now = time.monotonic()
            duo.tick(now)
            if now - looked > PLUG_S:
                looked = now
                if not keyboard:
                    pass
                elif plugged and not codeK.present(keyboard):
                    plugged = False
                    print("Keyboard unplugged: waiting for it to come back")
                    # its last note-offs never came: let go of the keys and the pedal it was holding
                    on_midi(([0xB0, PEDAL, 0], 0))
                    for pitch in list(duo.held):
                        on_midi(([0x80, pitch, 0], 0))
                elif not plugged and codeK.present(keyboard):
                    codeK.reopen(keyboard)
                    plugged = True
            time.sleep(0.002)
    except KeyboardInterrupt:
        pass
    finally:
        keys.close_port()
        codeK.end()
        duo.stop(time.monotonic())
        sock.close()
        print(f"The model played {sum(bool(n[4]) for n in duo.notes)} notes; {duo.late} came later than {LATE_S} s and "
              f"were skipped")
        duo.save(f"generated/{int(time.time())}.mid")
        print("Recording Stopped")
        sys.exit(0)
