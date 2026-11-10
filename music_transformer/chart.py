"""
A chart, as in a Real Book: the chords of a tune bar by bar, in 4/4, for the duo to play over with a metronome.
A file in charts/ is a few `name: value` lines (title, key as "G minor", bpm) and then the bars, between `|`: one or
more chords each, sharing the bar equally; `%` is the bar before again. Chords are spelt as in a Real Book or
iReal Pro (Cm7, C-7, F7b9, Bbmaj7, B^7, Am7b5, Aø, Gm6, C7/E...) and heard as the nearest of listener.CHORDS.
"""
import glob
import os
import re

from music_transformer.listener import CHORDS

ROOTS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
BEATS = 4   # shortcut: every chart is in 4/4; give a chart its own `meter` when one in 3/4 is wanted


def chord(name):
    """(root, kind in listener.CHORDS) of a chord symbol; extensions and the bass after a slash are left out."""
    match = re.fullmatch(r'([A-G])([#b]?)([^/]*)(/.*)?', name)
    if not match:
        raise ValueError(f"not a chord: {name!r}")
    letter, accidental, kind, _ = match.groups()
    root = (ROOTS[letter] + {'#': 1, 'b': -1, '': 0}[accidental]) % 12
    kind = kind.replace('-', 'm').replace('^', 'maj').replace('M', 'maj').replace('ø', 'm7b5').replace('h', 'm7b5')
    kind = kind.replace('o', 'dim').replace('min', 'm').replace('aug', '+').replace('sus', 'sus4').replace('sus44', 'sus4')
    if kind in CHORDS:
        return root, kind
    for start, heard in (('m7b5', 'm7b5'), ('mmaj', 'mMaj7'), ('maj', 'maj7'), ('dim7', 'dim7'), ('dim', 'dim'),
                         ('m6', 'm6'), ('m', 'm7'), ('6', '6'), ('+', '+')):
        if kind.startswith(start):
            return root, heard
    if 'sus' in kind:
        return root, '7sus4'
    return root, '7' if re.match(r'(7|9|11|13)', kind) else ''


class Chart:
    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.title, self.bpm, self.key, self.bars = self.name.replace('_', ' '), 120, None, []
        with open(path) as f:
            for line in f:
                head, _, value = line.partition(':')
                if value and head.strip() in ('title', 'key', 'bpm'):
                    if head.strip() == 'title':
                        self.title = value.strip()
                    elif head.strip() == 'bpm':
                        self.bpm = int(value)
                    else:   # 0-11 major from C, 12-23 minor, as listener's keys
                        tonic, mode = value.split()
                        self.key = chord(tonic)[0] + 12 * (mode == 'minor')
                    continue
                for bar in filter(None, (bar.split() for bar in line.split('|'))):
                    self.bars.append(self.bars[-1] if bar == ['%'] else [chord(name) + (name,) for name in bar])
        if not self.bars:
            raise ValueError(f"no bars in {path}")

    def at(self, beats):
        """(bar from 0, (root, kind, name) there) at `beats` since the chart began; it goes round and round."""
        bar = int(beats // BEATS) % len(self.bars)
        chords = self.bars[bar]
        return bar, chords[int(beats % BEATS / BEATS * len(chords))]


CHARTS = [Chart(path) for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'charts', '*.txt')))]
