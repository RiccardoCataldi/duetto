"""
A memory in the manner of Somax2 (Borg, Assayag, Malt), written from its published model: what you play becomes a
corpus of slices; what you play next wakes points of that corpus, peaks, that move on with time and fade, so several
ideas of where you are can grow and die together; the strongest says what comes next, and when. It listens on three
layers (your line, your harmony, its own line) and recognises you in any key. Rests are in the corpus: a slice lasts
until the next onset, however short its notes.

Times are seconds on the session's clock: a corpus loaded from files lies before zero.
"""

import math
import os
from collections import defaultdict, deque

import mido
import numpy as np

SLICE = 0.03         # seconds within which notes make one slice
NGRAM = 3            # intervals of your line that must match
NEAR = 4.0           # seconds: your harmony is not matched against the moments just gone, which always resemble it
ALIKE = 0.85         # how alike two harmonies must be to wake a peak
WAKE = 64            # peaks your harmony wakes at most at each slice
CHROMA_S = 0.4       # half-life of a note in the harmony of a slice
WIDTH = 0.2          # seconds: peaks this close, in the same key, add up
MOST = 256           # peaks kept
FADED = 0.1          # how much of a peak's strength is left when it dies
LONGEST = 4.0        # seconds a slice may last
GAP = 2.0            # seconds of silence between the corpus's files and your playing
TRUST = 1.0          # how much a harmony that agrees or disagrees with a peak moves its strength
PHASE = 3.0          # how strongly a peak must fall on the same place of the beat
TRANSPOSED = np.array([[(pc - shift) % 12 for pc in range(12)] for shift in range(12)])


class Memory:
    def __init__(self, decay=30.0, least=1.5, continuity=1.5, jump=8, melodic=1.0, harmonic=1.0, itself=0.5,
                 probabilistic=False):
        self.decay, self.least, self.continuity, self.jump = decay, least, continuity, jump
        self.melodic, self.harmonic, self.itself, self.probabilistic = melodic, harmonic, itself, probabilistic
        self.times, self.phases, self.chroma = np.zeros(1024), np.zeros(1024), np.zeros((1024, 12))
        self.slices, self.tops = [], []        # [(pitch, duration)] of each slice, and its highest pitch
        self.index = defaultdict(list)         # NGRAM intervals -> the slices that end them
        self.yours, self.mine = deque(maxlen=NGRAM + 1), deque(maxlen=NGRAM + 1)   # top pitches of the two lines
        self.colour, self.coloured = np.zeros(12), None   # your harmony now, and when it was last faded
        self.open, self.listening = None, True # the slice still taking notes; loading a corpus wakes no peaks
        self.at = None                         # the time the peaks stand at
        self.positions, self.scores = np.zeros(0), np.zeros(0)
        self.shifts = np.zeros(0, dtype=int)
        self.previous, self.run = None, 0      # the slice last played, its transposition, and how long it has run
        self.shift = 0

    @property
    def size(self):
        return len(self.slices)

    def load(self, paths):
        """Takes the first track with notes of each MIDI file (in a saved duo, yours) into the corpus."""
        pieces = [notes for notes in map(_first_track, sorted(paths)) if notes]
        time_, self.listening = -sum(max(o + d for o, d, _ in notes) + GAP for notes in pieces) - GAP, False
        for notes in pieces:
            for onset, duration, pitch in notes:
                self._take(time_ + onset, pitch, duration, float('nan'))
            self._close()
            time_ += max(o + d for o, d, _ in notes) + GAP
            self.yours.clear()
        self.colour[:], self.coloured, self.at, self.listening = 0, None, None, True

    def hear(self, onset, pitch, duration, phase=None):
        """Your note: it joins the corpus and, once its slice is complete, wakes the peaks it matches."""
        self.update(onset)
        self._take(onset, pitch, duration, float('nan') if phase is None else phase)

    def release(self, onset, pitch, duration):
        """Your note pressed at `onset` lasted `duration`."""
        for i in range(self.size - 1, -1, -1):
            if self.times[i] < onset - SLICE:
                return
            self.slices[i] = [(p, duration if p == pitch else d) for p, d in self.slices[i]]

    def own(self, at, pitch):
        """Its own note at `at`: it wakes the peaks of the third layer, its memory of itself."""
        self._line(self.mine, pitch, self.itself, at)

    def update(self, now):
        """Moves the peaks on to `now`, fading them, and hears the slice that is complete."""
        if self.open is not None and now - self.times[self.open] >= SLICE:
            self._close()
        if self.at is not None and now > self.at:
            self.positions += now - self.at
            self.scores *= math.exp(-(now - self.at) * math.log(1 / FADED) / self.decay)
        self.at = now if self.at is None else max(self.at, now)
        alive = (self.scores > FADED) & (self.positions < self._known())
        self._keep(alive)

    def step(self, at, beat_phase=None):
        """The slice to play at `at`, ahead of now: (slice, transposition, matched), or None with nothing to play.
        `matched` is false when no peak is strong enough and it carries on along the corpus from the last slice."""
        if self.size == 0:
            return None
        ahead = at - self.at
        slices = np.clip(np.searchsorted(self.times[:self.size], self.positions + ahead, 'right') - 1, 0, self.size - 1)
        scores = self.scores.copy()
        scores[self.positions + ahead >= self._known()] = 0
        if self.previous is not None:
            next_ = (slices == self.previous + 1) & (self.shifts == self.shift)
            if self.run >= self.jump:
                scores[next_] = 0       # long enough along one line: it must jump
            else:
                scores[next_] *= self.continuity * 0.5 ** max(self.run - 2, 0)
        if beat_phase is not None:
            known = ~np.isnan(self.phases[slices])
            scores[known] *= np.exp(PHASE * (np.cos(2 * np.pi * (beat_phase - self.phases[slices[known]])) - 1))
        scores[scores < self.least] = 0
        if scores.any():
            i = int(np.random.choice(len(scores), p=scores / scores.sum()) if self.probabilistic else scores.argmax())
            return self._played(int(slices[i]), int(self.shifts[i])), int(self.shifts[i]), True
        if self.previous is not None and self.previous + 1 < self.size:
            return self._played(self.previous + 1, self.shift), self.shift, False
        return None

    def notes(self, slice_, shift):
        """[(pitch, duration)] of a slice, transposed, lowest first."""
        return sorted((p + shift, d) for p, d in self.slices[slice_])

    def duration(self, slice_):
        """Seconds until the next slice: its notes and the rest after them."""
        if slice_ + 1 >= self.size:
            return LONGEST
        return min(self.times[slice_ + 1] - self.times[slice_], LONGEST)

    def expect(self, at):
        """(chroma, strength) of the harmony it expects at `at`, or (None, 0) with no peak strong enough."""
        if len(self.scores) == 0 or self.scores.max() < self.least:
            return None, 0.0
        i = int(self.scores.argmax())
        position = self.positions[i] + at - self.at
        slice_ = int(np.searchsorted(self.times[:self.size], position, 'right') - 1)
        if slice_ < 0 or position >= self._known():
            return None, 0.0
        return self.chroma[slice_][TRANSPOSED[self.shifts[i] % 12]], float(self.scores[i])

    def _played(self, slice_, shift):
        same = self.previous is not None and slice_ == self.previous + 1 and shift == self.shift
        self.run = self.run + 1 if same else 0
        self.previous, self.shift = slice_, shift
        return slice_

    def _known(self):
        """The end of what it has heard: a peak has nothing past it to say."""
        return self.times[self.size - 1] if self.size else 0.0

    def _take(self, time_, pitch, duration, phase):
        self._fade(time_)
        if self.open is not None and time_ - self.times[self.open] < SLICE:
            self.colour[pitch % 12] += 1
            self.slices[self.open].append((pitch, duration))
            self.tops[self.open] = max(self.tops[self.open], pitch)
            return
        self._close()
        self.colour[pitch % 12] += 1
        if self.size == len(self.times):
            self.times, self.phases = np.resize(self.times, 2 * self.size), np.resize(self.phases, 2 * self.size)
            self.chroma = np.resize(self.chroma, (2 * self.size, 12))
        self.open = self.size
        self.times[self.size], self.phases[self.size] = time_, phase
        self.slices.append([(pitch, duration)])
        self.tops.append(pitch)

    def _close(self):
        """The open slice is complete: it is matched against what came before, then remembered."""
        if self.open is None:
            return
        i, self.open = self.open, None
        norm = np.linalg.norm(self.colour)
        self.chroma[i] = self.colour / norm if norm else self.colour
        if self.listening:
            self.at = self.times[i] if self.at is None else self.at
            self._harmony(i)
        self._line(self.yours, self.tops[i], self.melodic if self.listening else 0.0, self.times[i], i)

    def _line(self, history, pitch, weight, at, learn=None):
        history.append(pitch)
        if len(history) <= NGRAM:
            return
        key = tuple(np.diff(list(history)).tolist())
        if weight and self.index[key]:
            matches = np.array(self.index[key])
            self._wake(self.times[matches] + self.at - at, pitch - np.array(self.tops)[matches], weight)
        if learn is not None:
            self.index[key].append(learn)

    def _harmony(self, i):
        earlier = int(np.searchsorted(self.times[:i], self.times[i] - NEAR))
        if earlier == 0 or not self.harmonic:
            return
        if len(self.scores):      # each harmony you play is evidence for the peaks that expected it, against the others
            there = np.clip(np.searchsorted(self.times[:i], self.positions, 'right') - 1, 0, i - 1)
            expected = self.chroma[there[:, None], TRANSPOSED[self.shifts % 12]]
            self.scores = self.scores * (1 + TRUST * (expected @ self.chroma[i] - ALIKE))
        alike = self.chroma[:earlier] @ self.chroma[i][TRANSPOSED].T    # [slice, transposition]
        slices, shifts = np.nonzero(alike >= ALIKE)
        if len(slices) > WAKE:
            strongest = np.argpartition(alike[slices, shifts], -WAKE)[-WAKE:]
            slices, shifts = slices[strongest], shifts[strongest]
        self._wake(self.times[slices] + self.at - self.times[i], -shifts, self.harmonic * alike[slices, shifts])

    def _wake(self, positions, shifts, scores):
        """New peaks, and those in one place and key add up: what agrees grows."""
        if len(positions) == 0:
            return
        shifts = np.mod(np.asarray(shifts) + 6, 12) - 6
        self.positions = np.concatenate((self.positions, positions))
        self.shifts = np.concatenate((self.shifts, np.broadcast_to(shifts, np.shape(positions)).astype(int)))
        self.scores = np.concatenate((self.scores, np.broadcast_to(scores, np.shape(positions))))
        place = np.round(self.positions / WIDTH).astype(np.int64) * 24 + self.shifts + 12
        _, first, group = np.unique(place, return_index=True, return_inverse=True)
        scores = np.bincount(group, self.scores)
        self.positions = np.bincount(group, self.scores * self.positions) / scores
        self.shifts, self.scores = self.shifts[first], scores
        if len(self.scores) > MOST:
            self._keep(np.argpartition(self.scores, -MOST)[-MOST:])

    def _keep(self, which):
        self.positions, self.scores, self.shifts = self.positions[which], self.scores[which], self.shifts[which]

    def _fade(self, now):
        if self.coloured is not None:
            self.colour *= 0.5 ** (max(now - self.coloured, 0) / CHROMA_S)
        self.coloured = now


def _first_track(path):
    """[(onset, duration, pitch)] of the first track with notes of a MIDI file: seconds from its first note."""
    midi = mido.MidiFile(path)
    for track in midi.tracks:
        time_, tempo, held, notes = 0.0, 500000, {}, []
        for message in track:
            time_ += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
            if message.type == 'set_tempo':
                tempo = message.tempo
            elif message.type == 'note_on' and message.velocity:
                held[message.note] = time_
            elif message.type in ('note_on', 'note_off') and message.note in held:
                onset = held.pop(message.note)
                notes.append((onset, time_ - onset, message.note))
        if notes:
            notes.sort()
            return [(o - notes[0][0], d, p) for o, d, p in notes]
    return []


def midi_files(path):
    """The MIDI files in a folder."""
    return [os.path.join(path, name) for name in os.listdir(path) if name.lower().endswith(('.mid', '.midi'))]
