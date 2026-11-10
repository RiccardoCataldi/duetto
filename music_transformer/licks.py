"""
The phrases the duo quotes when it solos over a tune: one or two bars of the great soloists as they played them,
filed under the chords they were played over and kept free of any key (licks.json, made by make_licks.py from the
Weimar Jazz Database, under the ODbL).
"""
import json
import os

from music_transformer.bebop import kind_of
from music_transformer.make_licks import key_of

with open(os.path.join(os.path.dirname(__file__), 'licks.json')) as f:
    DATA = json.load(f)


def quote(rng, chords, low, high, near):
    """A phrase to play over `chords`, the (root, kind in listener.CHORDS) of each beat of two bars: one played over
    the same chords, a fourth up or a tone down as they may be, or failing that one for each bar. Drawn with `rng`,
    in the octave that keeps it from `low` to `high` and nearest the pitch `near`. Returns ("who · the tune", [(beats
    from the first bar line, beats it lasts, pitch), ...]), or None when nothing was played over these chords."""
    under = [(root, kind_of(kind)) for root, kind in chords]
    for parts in ([under], [under[:4], under[4:]]):
        found = [DATA['licks'].get(key_of(part)) for part in parts]
        if all(found):
            break
    else:
        return None
    notes, names = [], []
    for part, licks in zip(parts, found):
        player, lick = rng.choice(licks)
        pitches = sorted(pitch for *_, pitch in lick)
        octaves = [12 * k + part[0][0] for k in range(11)
                   if 12 * k + part[0][0] + pitches[0] >= low and 12 * k + part[0][0] + pitches[-1] <= high]
        if not octaves:
            return None
        shift = min(octaves, key=lambda shift: abs(shift + pitches[len(pitches) // 2] - near))
        notes += [(at + len(notes and under[:4]), lasts, pitch + shift) for at, lasts, pitch in lick]
        names.append(' · '.join(DATA['players'][player]))
        near = notes[-1][2]
    return ', '.join(dict.fromkeys(names)), notes
