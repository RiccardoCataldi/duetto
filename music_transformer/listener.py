"""
Listens to your playing the way a duo partner does: the key you are in and when you modulate, the chord under your
hands, your pulse and your last phrase. Times are seconds since the connection opened.
"""

import cmath
import math

import numpy as np

NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]   # Krumhansl-Kessler, from C
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
SCALES = ((0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10, 11))                      # major, minor with both sevenths
# the chords of jazz harmony: C6 is Am7 and Cm6 is Am7b5 (and dim7 is itself four times), told apart by the bass
CHORDS = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10), 'dim': (0, 3, 6),
          'maj7': (0, 4, 7, 11), '6': (0, 4, 7, 9), 'm6': (0, 3, 7, 9), 'm7b5': (0, 3, 6, 10), 'dim7': (0, 3, 6, 9),
          'mMaj7': (0, 3, 7, 11), 'sus4': (0, 5, 7), '7sus4': (0, 5, 7, 10), '+': (0, 4, 8)}
# the tensions a jazz pianist colours each chord with, in semitones above its root: 9, #11 or 11, 13
TENSIONS = {'': (2, 6, 9), 'maj7': (2, 6, 9), '6': (2, 6), 'm': (2, 5), 'm7': (2, 5, 9), 'm6': (2, 5),
            'mMaj7': (2, 5, 9), '7': (2, 9), 'm7b5': (5, 8), 'dim': (2, 5, 8, 11), 'dim7': (2, 5, 8, 11),
            'sus4': (2, 9), '7sus4': (2, 9), '+': (2, 6)}
BASS = 0.15      # how much the lowest sounding note says the chord's root is

STAY = 0.97      # chance per note that the key stays: lower follows a modulation sooner but flickers on passing notes
SHARPNESS = 2.0  # how much one note says about the key
FAR = 0.6        # each step round the circle of fifths makes a modulation there about 1.8 times less likely
CHORD_S = 0.4    # half-life of the notes that make the chord
PULSE_S = 6.0    # seconds of your onsets the pulse is read from
PHRASE = 1.5     # a rest of this many beats ends your phrase
KEEP_S = 20.0    # seconds of your notes kept


MEMBER = np.array([[(pc - k % 12) % 12 in SCALES[k >= 12] for pc in range(12)] for k in range(24)], float)


def _transitions(stay):
    """How likely each key is to move to each other one, at each note."""
    signature = [7 * (k % 12 + (3 if k >= 12 else 0)) % 12 for k in range(24)]   # a minor key shares its relative's
    far = np.array([[min((a - b) % 12, (b - a) % 12) + 0.5 * ((i < 12) != (j < 12))
                     for j, b in enumerate(signature)] for i, a in enumerate(signature)])
    move = np.exp(-FAR * far)
    np.fill_diagonal(move, 0)
    move = (1 - stay) * move / move.sum(1, keepdims=True)
    np.fill_diagonal(move, stay)
    return move


def _keys():
    profiles = np.array([np.roll(MAJOR, k) for k in range(12)] + [np.roll(MINOR, k) for k in range(12)])
    emit = profiles / profiles.sum(1, keepdims=True)
    return np.log(emit) * SHARPNESS


EMIT, MOVE = _keys(), _transitions(STAY)


class Listener:
    def __init__(self):
        self.belief = np.full(24, 1 / 24)
        self.notes = []   # (onset, duration, pitch), oldest first
        self.held = set() # (onset, pitch) of your notes still sounding: their duration is only a guess until release()
        self.releases = False   # whether you say when your notes stop sounding (main.py does)
        self.move = MOVE

    def stay(self, stay):
        """Sets STAY for this listener."""
        self.move = _transitions(stay)

    def hear(self, onset, duration, pitch):
        """Your note. Once release() has been heard, it sounds on past `duration` until its own release() comes."""
        belief = (self.belief @ self.move) * np.exp(EMIT[:, pitch % 12])
        self.belief = belief / belief.sum()
        self.notes = [n for n in self.notes if n[0] > onset - KEEP_S] + [(onset, duration, pitch)]
        self.held = {n for n in self.held if n[0] > onset - KEEP_S} | ({(onset, pitch)} if self.releases else set())

    def release(self, onset, pitch, duration):
        """Your note pressed at `onset` has stopped sounding: it lasted `duration`."""
        self.releases = True
        self.held.discard((onset, pitch))
        self.notes = [(o, duration if (o, p) == (onset, pitch) else d, p) for o, d, p in self.notes]

    def key(self):
        """(key, how likely): 0-11 major from C, 12-23 minor."""
        k = int(self.belief.argmax())
        return k, float(self.belief[k])

    @staticmethod
    def name(key):
        return f"{NAMES[key % 12]} {'minor' if key >= 12 else 'major'}"

    def in_key(self):
        """For each pitch class, how likely it belongs to your key: while you modulate, both keys count."""
        return self.belief @ MEMBER

    def chord(self, now):
        """Pitch classes of the chord you are playing now, or None when nothing is sounding."""
        harmony = self.harmony(now)
        return harmony[2] if harmony else None

    def harmony(self, now):
        """(root, kind, pitch classes) of the chord you are playing now, e.g. (7, '7', {7, 11, 2, 5}); or None."""
        weight, bass = np.zeros(12), None
        for onset, duration, pitch in self.notes:
            if onset <= now:
                if (onset, pitch) in self.held:
                    duration = max(duration, now - onset)
                weight[pitch % 12] += 0.5 ** (max(now - onset - duration, 0) / CHORD_S)
                if onset + duration >= now and (bass is None or pitch < bass):
                    bass = pitch
        if weight.sum() < 0.5:
            return None
        scale = set(np.flatnonzero(self.in_key() > 0.5))
        norm = np.linalg.norm(weight)
        best, harmony = -1, None
        for root in range(12):
            for kind, shape in CHORDS.items():
                tones = {(root + step) % 12 for step in shape}
                score = sum(weight[p] for p in tones) / math.sqrt(len(tones)) / norm \
                    + 0.1 * tones.issubset(scale) + BASS * (bass is not None and bass % 12 == root)
                if score > best:
                    best, harmony = score, (root, kind, tones)
        return harmony

    def chord_label(self, now):
        """The name of the chord you are playing now, e.g. G7 (C6 rather than Am7 when C is in the bass)."""
        harmony = self.harmony(now)
        return NAMES[harmony[0]] + harmony[1] if harmony else None

    def tensions(self, now):
        """Pitch classes that colour the chord you are playing now (its 9, 11 or #11, 13), or an empty set."""
        harmony = self.harmony(now)
        if not harmony:
            return set()
        root, kind, tones = harmony
        return {(root + step) % 12 for step in TENSIONS[kind]} - tones

    def bias(self, now, chord_tone, tension, out_of_key):
        """For each of the 128 pitches, how much more a partner would play it now: the tones of your chord and its
        tensions, less the pitches outside your key (a tension is not taken as one)."""
        in_key, harmony = self.in_key(), self.harmony(now)
        chord = harmony[2] if harmony else set()
        colours = {(harmony[0] + step) % 12 for step in TENSIONS[harmony[1]]} - chord if harmony else set()
        return np.array([chord_tone * (p % 12 in chord) + tension * (p % 12 in colours)
                         - out_of_key * (1 - in_key[p % 12]) * (p % 12 not in colours) for p in range(128)])

    @staticmethod
    def chord_name(tones):
        """The name of a chord from chord(), e.g. G7."""
        for root in range(12):
            for suffix, shape in CHORDS.items():
                if {(root + step) % 12 for step in shape} == tones:
                    return NAMES[root] + suffix
        return None

    def pulse(self, now):
        """(tatum, phase) in seconds: your onsets fall on phase + k * tatum. None until your rhythm is regular."""
        onsets = sorted({round(n[0], 2) for n in self.notes if n[0] > now - PULSE_S})
        if len(onsets) < 6:
            return None
        fit = {p / 100: abs(sum(cmath.exp(2j * math.pi * t * 100 / p) for t in onsets)) / len(onsets)
               for p in range(15, 101)}
        best = max(fit.values())
        if best < 0.6:
            return None
        tatum = max(p for p, r in fit.items() if r >= 0.9 * best)
        phase = cmath.phase(sum(cmath.exp(2j * math.pi * t / tatum) for t in onsets)) / (2 * math.pi) * tatum
        return tatum, phase % tatum

    @staticmethod
    def beat(tatum):
        return tatum * math.ceil(0.4 / tatum)

    def phrase(self, now):
        """Pitches of your last phrase, once a rest has ended it; [] while you are still playing it."""
        pulse = self.pulse(now)
        rest = PHRASE * (self.beat(pulse[0]) if pulse else 0.4)
        if not self.notes or now - self.notes[-1][0] < rest:
            return []
        start = len(self.notes) - 1
        while start > 0 and self.notes[start][0] - self.notes[start - 1][0] < rest:
            start -= 1
        return [n[2] for n in self.notes[start:]]
