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
PLUG_S = 1.0             # how often we look whether your keyboard was unplugged or plugged back in
RING = 1.6               # its notes ring this much longer than the model wrote them, as if with a light pedal
RING_S = 0.4             # and at least this many seconds more
# its dynamics, shaped as a pianist shapes them; it follows how loud you play (LEVEL: how much your last note counts)
LEVEL = 0.3
QUIETEST = 80            # the softest it follows you down to: some keyboards never send more than 70
SOFTER = 0.8             # it plays a little softer than you
ACCOMPANY = 0.85         # and softer still while you play
LEFT_HAND = 0.85         # its left hand under its right
TOP, INNER = 1.08, 0.82  # in a chord the top voice sings and the inner voices fall back
CONTOUR = 0.2            # its line swells as it rises and falls back as it descends: this much from lowest to highest
CONTOUR_NOTES = 16       # its last notes the contour is measured against
HUMAN = 4                # velocity a little different every time, as a hand never strikes twice the same
# the metronome of a tempo you fix (the bpm and metronome knobs): General MIDI percussion, a bar of BAR beats
DRUMS = 9
CLICK = (76, 77)         # hi wood block on the first beat of a bar, low wood block on the others
CLICK_VELOCITY = 70
BAR = 4


def dynamics(pitch, hand, together, line, level, accompanying, rng=random):
    """The velocity of one of the model's notes. `together`: the pitches its right hand starts with it,
    `line`: its right hand's last pitches, `level`: how loud you play, `accompanying`: whether you are playing."""
    velocity = max(level, QUIETEST) * SOFTER * (ACCOMPANY if accompanying else 1)
    if hand == 'left':
        velocity *= LEFT_HAND
    elif len(together) > 1:
        velocity *= TOP if pitch == max(together) else INNER
    if hand != 'left' and len(line) >= 4 and max(line) > min(line):
        height = min(max((pitch - min(line)) / (max(line) - min(line)), 0), 1)
        velocity *= 1 + CONTOUR * (height - 0.5)
    velocity += rng.uniform(-HUMAN, HUMAN)
    return min(max(round(velocity), 1), 127)


class Duo:
    """
    You and the model play at the same time. Every key you press goes to the model straight away, and how long it
    lasted once it stops sounding; it sends back its own line a little ahead of time, and redoes the part that has
    not sounded yet whenever you play.
    `send` takes one protocol line (see music_transformer/server.py); `started` is when the connection opened.
    """

    def __init__(self, send, midi_rec, started, rng=random):
        self.send = send
        self.midi_rec = midi_rec
        self.started = started
        self.rng = rng
        self.lock = threading.Lock()
        self.notes = []                # (onset, offset, pitch, velocity, generated) in time.monotonic() seconds
        self.held = {}                 # pitch -> (onset, velocity), the keys you hold down
        self.pedaled = {}              # pitch -> (onset, velocity), the keys you let go that the pedal holds
        self.pedal = False
        self.pedalings = []            # (time, value) of your sustain pedal
        self.durations = deque(maxlen=16)
        self.level = 80                # how loud you play
        self.played_at = None          # when you last pressed a key
        self.pending = []              # the model's notes not started yet: (onset, duration, pitch, hand, velocity)
        self.sounding = {}             # pitch -> (onset, offset, velocity) of the model's notes
        self.line = deque(maxlen=CONTOUR_NOTES)   # its right hand's last pitches
        self.late = 0                  # the model's notes that came later than LATE_S, skipped
        self.bpm, self.metronome = 0, False   # the tempo the model keeps to (0: yours) and whether we click it
        self.click_at = None           # when the metronome's next beat falls

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
                onset, duration, pitch, *how = values   # its hand, and how hard if the model says (Aria does)
                hand, velocity = (how + ['right'])[0], int(how[1]) if len(how) > 1 else None
                self.pending.append((self.started + float(onset), float(duration), int(pitch), hand, velocity))
            elif kind == 'cut':
                cut = self.started + float(values[0])
                self.pending = [n for n in self.pending if n[0] < cut]
            elif kind == 'tempo':
                self.bpm, self.metronome, self.click_at = float(values[0]), values[1] == '1', None

    def tick(self, now):
        """Starts and stops the model's notes that are due, and clicks the metronome."""
        with self.lock:
            if self.metronome and self.bpm:
                self._click(now)
            for pitch, (onset, offset, _) in list(self.sounding.items()):
                if offset <= now:
                    self._release(pitch, now)
            due = [n for n in self.pending if n[0] <= now]
            self.pending = [n for n in self.pending if n[0] > now]
            accompanying = self.played_at is not None and now - self.played_at < 1.0
            for onset, duration, pitch, hand, given in sorted(due, key=lambda n: n[:3]):
                if now - onset > LATE_S:
                    self.late += 1
                    continue
                if pitch in self.sounding:
                    self._release(pitch, now)
                together = [n[2] for n in due if n[0] == onset and n[3] == hand]
                velocity = given or dynamics(pitch, hand, together, self.line, self.level, accompanying, self.rng)
                if hand != 'left':
                    self.line.append(pitch)
                self.midi_rec.play_midi_message([0x90, pitch, velocity], is_generated=True)
                self.sounding[pitch] = (now, now + max(duration * RING, duration + RING_S), velocity)

    def _click(self, now):
        """Clicks the beat that is due: the model's beats fall on k * 60 / bpm seconds after the connection opened."""
        beat = 60 / self.bpm
        if self.click_at is not None and now >= self.click_at:
            if now - self.click_at < LATE_S:
                click = CLICK[round((self.click_at - self.started) / beat) % BAR > 0]
                self.midi_rec.play_midi_message([0x90, click, CLICK_VELOCITY], channel=DRUMS)
                self.midi_rec.play_midi_message([0x80, click, 0], channel=DRUMS)
            self.click_at = None
        if self.click_at is None:
            self.click_at = self.started + math.floor((now - self.started) / beat + 1) * beat

    def stop(self, now):
        with self.lock:
            self.pending = []
            for pitch in list(self.pedaled):
                self._sounded(pitch, now)
            for pitch in list(self.sounding):
                self._release(pitch, now)

    def _release(self, pitch, now):
        onset, _, velocity = self.sounding.pop(pitch)
        self.midi_rec.play_midi_message([0x80, pitch, 0], is_generated=True)
        self.notes.append((onset, now, pitch, velocity, True))

    def save(self, path):
        if not self.notes:
            return
        mid = mido.MidiFile()
        ticks = lambda seconds: int(mido.second2tick(seconds, mid.ticks_per_beat, mido.bpm2tempo(120)))
        origin = min(n[0] for n in self.notes)
        for generated, channel, program in ((False, 0, INPUT_PROGRAM), (True, 1, GENERATED_PROGRAM)):
            messages = []
            for onset, offset, pitch, velocity, is_generated in self.notes:
                if is_generated == generated:
                    messages.append((onset - origin, 'note_on', pitch, velocity))
                    messages.append((offset - origin, 'note_off', pitch, 0))
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
        sound.save(sound.load(), os.path.splitext(path)[0] + '.sound.json')
        print(f"Duet saved as {path}")


def connect(address):
    host, port = address.rsplit(':', 1)
    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"Playing with the AMT model at {address}")
    return sock, time.monotonic()


if __name__ == "__main__":
    codeK = Setup()
    myPort = codeK.perform_setup()
    codeK.open_port(myPort)
    keyboard = codeK.port_name(myPort)

    midiRec = CK_rec(myPort, 0x90, debug=False)
    midiRec.set_input_instrument(INPUT_PROGRAM)
    midiRec.set_generated_instrument(GENERATED_PROGRAM)
    for channel, controls in ((midiRec.input_channel, INPUT_CONTROLS), (midiRec.generated_channel, GENERATED_CONTROLS)):
        for control, value in controls:
            midiRec.play_midi_message([0xB0 | channel, control, value])
    sock, started = connect(os.environ.get('AMT_SERVER', 'localhost:8765'))
    duo = Duo(lambda line: sock.sendall(line.encode()), midiRec, started)

    def listen():
        for line in sock.makefile('r'):
            duo.receive(line)
        print("The AMT server closed the duo")

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

    codeK.set_callback(on_midi)
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
                if plugged and not codeK.present(keyboard):
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
        codeK.end()
        duo.stop(time.monotonic())
        sock.close()
        print(f"The model played {sum(n[4] for n in duo.notes)} notes; {duo.late} came later than {LATE_S} s and "
              f"were skipped")
        duo.save(f"generated/{int(time.time())}.mid")
        print("Recording Stopped")
        sys.exit(0)
