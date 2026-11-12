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
# the scales you may force it into instead of your key, in semitones from their root
FORCED = {'major': (0, 2, 4, 5, 7, 9, 11), 'natural_minor': (0, 2, 3, 5, 7, 8, 10),
          'harmonic_minor': (0, 2, 3, 5, 7, 8, 11), 'melodic_minor': (0, 2, 3, 5, 7, 9, 11),
          'dorian': (0, 2, 3, 5, 7, 9, 10), 'phrygian': (0, 1, 3, 5, 7, 8, 10), 'lydian': (0, 2, 4, 6, 7, 9, 11),
          'mixolydian': (0, 2, 4, 5, 7, 9, 10), 'locrian': (0, 1, 3, 5, 6, 8, 10),
          'major_pentatonic': (0, 2, 4, 7, 9), 'minor_pentatonic': (0, 3, 5, 7, 10), 'blues': (0, 3, 5, 6, 7, 10)}
FORCE = 5.0      # the least logit taken from a pitch class outside a forced scale
# the chords of jazz harmony: C6 is Am7 and Cm6 is Am7b5 (and dim7 is itself four times), told apart by the bass
CHORDS = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10), 'dim': (0, 3, 6),
          'maj7': (0, 4, 7, 11), '6': (0, 4, 7, 9), 'm6': (0, 3, 7, 9), 'm7b5': (0, 3, 6, 10), 'dim7': (0, 3, 6, 9),
          'mMaj7': (0, 3, 7, 11), 'sus4': (0, 5, 7), '7sus4': (0, 5, 7, 10), '+': (0, 4, 8)}
# the tensions a jazz pianist colours each chord with, in semitones above its root: 9, #11 or 11, 13
TENSIONS = {'': (2, 6, 9), 'maj7': (2, 6, 9), '6': (2, 6), 'm': (2, 5), 'm7': (2, 5, 9), 'm6': (2, 5),
            'mMaj7': (2, 5, 9), '7': (2, 9), 'm7b5': (5, 8), 'dim': (2, 5, 8, 11), 'dim7': (2, 5, 8, 11),
            'sus4': (2, 9), '7sus4': (2, 9), '+': (2, 6)}
# over a chart: the scales that go with each chord, from its root; the one nearest the tune's key is played
CHORD_SCALES = {
    'maj7': ('major', 'lydian'), '': ('major', 'lydian', 'mixolydian'), '6': ('major', 'lydian'),
    '7': ('mixolydian', (0, 1, 4, 5, 7, 8, 10), (0, 1, 3, 4, 6, 8, 10)),   # or phrygian dominant, to a minor; altered
    'sus4': ('mixolydian', 'major'), '7sus4': ('mixolydian',),
    'm7': ('dorian', 'natural_minor', 'phrygian'), 'm': ('dorian', 'natural_minor', 'phrygian'),
    'm6': ('melodic_minor', 'dorian'), 'mMaj7': ('melodic_minor', 'harmonic_minor'),
    'm7b5': ('locrian', (0, 2, 3, 5, 6, 8, 10)), 'dim': ((0, 2, 3, 5, 6, 8, 9, 11),), 'dim7': ((0, 2, 3, 5, 6, 8, 9, 11),),
    '+': ((0, 2, 4, 6, 8, 10),)}
CHART_FORCE = 10.0   # logit taken from a pitch outside the chord's scale over a chart: its bass's, and at the default
                     # `out_of_key` its line's
BASS_TOP = 52    # over a chart, under E3 it is a bass: the chord's root, its fifth and its other tones only
ROOT = 3.0       # logit added to the root there, and half of it to the fifth
AVOID = 3.0      # logit taken from a chord's avoid notes: a semitone over one of its tones (E over your C minor)
BASS = 0.15      # how much the lowest sounding note says the chord's root is

STAY = 0.97      # chance per note that the key stays: lower follows a modulation sooner but flickers on passing notes
# over a chart the same stay says how long you must hold a chord of your own before it leaves the chart's for it: no
# time at QUICK, for ever at STEADY, (stay - QUICK) / (STEADY - stay) beats in between: at STAY about five, over a bar
QUICK, STEADY = 0.85, 0.995
SHARPNESS = 2.0  # how much one note says about the key
SURE = 0.6       # how likely a key must be before it is the one it says you are in: lower flickers between neighbours
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


HAND_GAP = 12     # semitones between two of your notes that make them two hands, with room between them


def hands(pitches):
    """Where your hands are: (lowest, highest) of each, from the bottom. Notes an octave or more apart are two hands,
    with room between them; closer, they are one hand, the keys between them included."""
    out = []
    for pitch in sorted(set(pitches)):
        if out and pitch - out[-1][1] < HAND_GAP:
            out[-1] = (out[-1][0], pitch)
        else:
            out.append((pitch, pitch))
    return out


class Listener:
    def __init__(self):
        self.belief = np.full(24, 1 / 24)
        self.notes = []   # (onset, duration, pitch), oldest first
        self.held = set() # (onset, pitch) of your notes still sounding: their duration is only a guess until release()
        self.releases = False   # whether you say when your notes stop sounding (main.py does)
        self.move, self.stays = MOVE, STAY
        self.forced = None      # (root, scale in FORCED) it keeps to instead of your key
        self.tempo = None       # seconds a beat of a tempo you fix lasts, a beat on `origin`; None: your pulse
        self.origin = 0.0
        self.said = None        # the key it says you are in (said_key)
        self.chart = None       # (chord(now) -> (root, kind), key) of a chart it follows instead of your chord and key

    def stay(self, stay):
        """Sets STAY for this listener; over a chart, how long you hold a chord before it follows it (harmony)."""
        self.move, self.stays = _transitions(stay), stay

    def force(self, root, scale):
        """Keeps it to `scale` (in FORCED) from `root` (0 = C) instead of your key; None for your key again."""
        self.forced = (root % 12, scale) if scale else None

    def fix(self, bpm, origin=0.0):
        """Keeps it to `bpm` instead of your pulse, a beat falling on `origin`; 0 for your pulse again."""
        self.tempo, self.origin = 60 / bpm if bpm else None, origin

    def hear(self, onset, duration, pitch):
        """Your note. Once release() has been heard, it sounds on past `duration` until its own release() comes."""
        belief = (self.belief @ self.move) * np.exp(EMIT[:, pitch % 12])
        self.belief = belief / belief.sum()
        if self.belief.max() > SURE:
            self.said = int(self.belief.argmax())
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

    def said_key(self):
        """The key it says you are in: the likeliest once it is surer than SURE, kept while none is; None before.
        What it plays leans on all the keys you may be in (in_key), this is only what it tells you."""
        return self.said

    @staticmethod
    def name(key):
        return f"{NAMES[key % 12]} {'minor' if key >= 12 else 'major'}"

    def forced_name(self):
        """The scale it is forced into, e.g. 'F minor pentatonic', or None."""
        return f"{NAMES[self.forced[0]]} {self.forced[1].replace('_', ' ')}" if self.forced else None

    def in_key(self):
        """For each pitch class, how likely it belongs to your key (while you modulate, both keys count), or 1 or 0
        to the scale it is forced into, or to the key of the chart it follows."""
        if self.chart and self.chart[1] is not None:   # a tune comes before a scale you force
            return MEMBER[self.chart[1]]
        if self.forced:
            root, scale = self.forced
            return np.array([(pc - root) % 12 in FORCED[scale] for pc in range(12)], float)
        return self.belief @ MEMBER

    def chord(self, now):
        """Pitch classes of the chord you are playing now, or None when nothing is sounding."""
        harmony = self.harmony(now)
        return harmony[2] if harmony else None

    def harmony(self, now):
        """(root, kind, pitch classes) of the chord you are playing now, e.g. (7, '7', {7, 11, 2, 5}), or None; the
        chart's chord at `now` when it follows one, unless you reharmonise it: then the chord you are holding, once
        you have held it as long as its stay asks (QUICK, STEADY), until you let it go."""
        if self.chart:
            root, kind = self.chart[0](now)
            written = {(root + step) % 12 for step in CHORDS[kind]}
            yours = {pitch % 12 for onset, duration, pitch in self.notes
                     if onset <= now and ((onset, pitch) in self.held or onset + duration >= now)
                     and (now - onset) * (STEADY - self.stays) >= (self.stays - QUICK) * self.tempo}
            # a line, a shell, or a voicing of the chart's chord with its tensions and no root, is no other chord: its
            # bass keeps the chart's root under it
            if len(yours) < 3 or yours <= written | {(root + step) % 12 for step in TENSIONS[kind]}:
                return root, kind, written
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

    def chord_scale(self, now):
        """Over a chart: for each pitch class, 1 if it is in the scale of the chord at `now`, the one of
        CHORD_SCALES nearest the tune's key (D7 in G minor: with its b9 and b13), else 0."""
        root, kind, tones = self.harmony(now)
        key = MEMBER[self.chart[1]] if self.chart[1] is not None else np.zeros(12)
        scales = [np.array([(pc - root) % 12 in (FORCED[s] if isinstance(s, str) else s) for pc in range(12)], float)
                  for s in CHORD_SCALES[kind]]
        scale = max(scales, key=lambda scale: scale @ key)   # the first of the nearest
        scale[list(tones)] = 1
        return scale

    def palette(self, now, name):
        """Over a chart: for each pitch class, 1 if a soloist playing `name` plays it over the chord at `now`.
        'bebop': anything (the session weighs the notes); 'scale': the chord's scale; 'arpeggio': the chord; 'pentatonic': the pentatonic on the chord's root; 'blues'
        and 'major_pentatonic': the blues scale and the major pentatonic of the tune's key, whatever the chord."""
        root, kind, tones = self.harmony(now)
        tonic = self.chart[1] % 12 if self.chart[1] is not None else root
        if name == 'arpeggio':
            steps, root = (), root
        elif name == 'pentatonic' and kind not in ('dim', 'dim7', '+'):
            steps = (0, 3, 5, 6, 10) if kind == 'm7b5' else FORCED['minor_pentatonic'] if kind.startswith('m') and \
                kind != 'maj7' else FORCED['major_pentatonic']
        elif name in ('blues', 'major_pentatonic'):
            steps, root, tones = FORCED[name], tonic, ()
        elif name == 'bebop':   # any note: the session weighs them as bebop soloists do (bebop.py)
            return np.ones(12)
        else:
            return self.chord_scale(now)
        scale = np.array([(pc - root) % 12 in steps for pc in range(12)], float)
        scale[list(tones)] = 1
        return scale

    def bias(self, now, chord_tone, tension, out_of_key, palette=None):
        """For each of the 128 pitches, how much more a partner would play it now: the tones of your chord and its
        tensions (over a chart: within what a soloist playing `palette` plays, see palette()), less its avoid notes (a semitone over a chord tone: they clash with it whatever the key) and the
        pitches outside your key (a tension is not taken as one); in a forced scale, by FORCE at least, its tensions
        included."""
        in_key, harmony = self.in_key(), self.harmony(now)
        chord = harmony[2] if harmony else set()
        colours = {(harmony[0] + step) % 12 for step in TENSIONS[harmony[1]]} - chord if harmony else set()
        if self.forced and not self.chart:
            colours &= set(np.flatnonzero(in_key))
            out_of_key = max(out_of_key, FORCE)
        elif self.chart:   # the scale of the chord of the bar, and nothing else; a bass under it
            in_key = self.palette(now, palette)
            colours &= set(np.flatnonzero(in_key))
            bass = np.array([p < BASS_TOP for p in range(128)])
            root = np.array([p % 12 == harmony[0] for p in range(128)]) + 0.5 * np.array(
                [p % 12 == (harmony[0] + 7) % 12 and p % 12 in chord for p in range(128)])
            low = bass * (ROOT * root - CHART_FORCE * np.array([p % 12 not in chord for p in range(128)]))
            avoid = set() if palette == 'bebop' else {(tone + 1) % 12 for tone in chord} - chord - colours
            # over its bass as firmly as `out_of_key` says, from none (outside, as without a chart) to more than twice
            # CHART_FORCE (inside); its bass keeps to it
            force = np.where(bass, CHART_FORCE, CHART_FORCE * out_of_key / 2)
            return low + np.array([chord_tone * (p % 12 in chord) + tension * (p % 12 in colours)
                                   - AVOID * (p % 12 in avoid) - force[p] * (1 - in_key[p % 12]) for p in range(128)])
        avoid = {(tone + 1) % 12 for tone in chord} - chord - colours
        free = colours if self.forced and not self.chart else colours | chord   # the chord you play is never out of your key
        return np.array([chord_tone * (p % 12 in chord) + tension * (p % 12 in colours) - AVOID * (p % 12 in avoid)
                         - out_of_key * (1 - in_key[p % 12]) * (p % 12 not in free) for p in range(128)])

    @staticmethod
    def chord_name(tones):
        """The name of a chord from chord(), e.g. G7."""
        for root in range(12):
            for suffix, shape in CHORDS.items():
                if {(root + step) % 12 for step in shape} == tones:
                    return NAMES[root] + suffix
        return None

    def pulse(self, now):
        """(tatum, phase) in seconds: your onsets fall on phase + k * tatum. None until your rhythm is regular; the
        eighths of a tempo you fix."""
        if self.tempo:
            return self.tempo / 2, self.origin % (self.tempo / 2)
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

    def beat(self, tatum):
        return self.tempo or tatum * math.ceil(0.4 / tatum)

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
