"""
A chart, as in a Real Book: the chords of a tune bar by bar, in 4/4, for the duo to play over with a metronome.
A file in charts/ is a few `name: value` lines (title, key as "G minor", bpm) and then the bars, between `|`: one or
more chords each, sharing the bar equally; `%` is the bar before again. Chords are spelt as in a Real Book or
iReal Pro (Cm7, C-7, F7b9, Bbmaj7, B^7, Am7b5, Aø, Gm6, C7/E...) and heard as the nearest of listener.CHORDS.
"""
import glob
import os
import random
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


CHANGES = 1.5   # over a chart it leans on the chord of the bar this much at least: it plays the changes
# the knob of a session that follows charts: its `chart`, `bpm`, `metronome`, `clock`, `started` and `listener`
KNOB = lambda: (0, 0, len(CHARTS), 'choice:' + '/'.join(['off', *(chart.name for chart in CHARTS)]),
                "a tune it plays over with you, bar by bar from the next bar on, at its tempo unless you fix one")


def bar_at(session, seconds):
    """(bar from 0, (root, kind, name)) of the session's chart at session `seconds`."""
    return CHARTS[session.chart - 1].at(max(seconds - session.began, 0) * session.pace / 60)


PHRASE = 2 * BEATS   # beats a soloist keeps to one way of playing the changes: two bars
WAYS = ('bebop',) * 6 + ('scale', 'arpeggio', 'pentatonic', 'blues')   # the ways, as often as each is here


def beats(session, seconds):
    """Beats of the session's chart played by session `seconds`."""
    return max(seconds - session.began, 0) * session.pace / 60


def way(session, seconds):
    """How it solos at `seconds`, one of listener.palette()'s: another, by chance, every two bars. In a major key
    the major pentatonic of the key is one more."""
    ways = WAYS + (('major_pentatonic',) if (CHARTS[session.chart - 1].key or 0) < 12 else ())
    return random.Random(session.varied + beats(session, seconds) // PHRASE).choice(ways)


FEELS = (2, 2, 2, 3, 3, 4, 4, 8, 0)   # how it divides the beat, as often as each is here; 0: out of time


def feel(session, seconds):
    """How a soloist divides the beat at `seconds`: eighths (2), triplets (3), sixteenths (4), thirty-seconds (8),
    or 0, free of the beat. Another, by chance, every two bars."""
    return random.Random(session.varied * 31 + 1 + beats(session, seconds) // PHRASE).choice(FEELS)


def follow(session, restart=True):
    """The session's chart or bpm knob was turned. With a chart it keeps to `pace`: the tempo you fix, or the
    chart's; the chart begins at `began`, the next bar from now, or, when only the tempo changed (`restart` False),
    it carries on from where it is. Without one: your chord and key again, at the tempo you fix or yours."""
    now, before = session.clock() - session.started, getattr(session, 'pace', 0)
    beats = max(now - getattr(session, 'began', 0.0), 0) * before / 60   # played so far
    session.listener.chart, session.began, session.pace = None, 0.0, session.bpm
    session.varied = getattr(session, 'varied', random.randrange(10 ** 6))   # its ways differ from one duo to the next
    if session.chart:
        tune = CHARTS[session.chart - 1]
        session.pace = session.bpm or tune.bpm
        bar = BEATS * 60 / session.pace
        session.began = now - beats * 60 / session.pace if before and not restart else -(-now // bar) * bar
        session.listener.chart = (lambda at: bar_at(session, at)[1][:2], tune.key)
    session.listener.fix(session.pace, session.began)


def reading(session, state):
    """`state` with the chart in the place of your key and chord: its title and bar, its chord and the next."""
    if session.chart:
        tune, now = CHARTS[session.chart - 1], session.clock() - session.started
        bar, here = bar_at(session, now)
        state['key'] = f"{tune.title} {bar + 1}/{len(tune.bars)}"
        state['chord'] = f"{here[2]} > {bar_at(session, now + BEATS * 60 / session.pace)[1][2]}"
        state['bpm'] = session.pace
        state['chart'] = {'title': tune.title, 'bars': [[name for *_, name in chords] for chords in tune.bars],
                          'bar': bar, 'beat': int(max(now - session.began, 0) * session.pace / 60 % BEATS)}
    return state


CHARTS = [Chart(path) for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'charts', '*.txt')))]
