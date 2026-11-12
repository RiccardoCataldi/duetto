"""
The phrases the duo quotes over a tune: one to four bars of the great soloists as they played them, filed under the
chords they were played over and kept free of any key (licks.json.gz, made by make_licks.py from the Weimar Jazz
Database, under the ODbL). The duo keeps yours in a library of the same form, and quotes them the same way.
"""
import functools
import gzip
import json
import os

from music_transformer.bebop import kind_of
from music_transformer.make_licks import LONGEST, key_of

with gzip.open(os.path.join(os.path.dirname(__file__), 'licks.json.gz'), 'rt') as f:
    DATA = json.load(f)


LONGER = 0.8   # how often it takes the phrase that fits the chords to come best, before it tries another
SAME = 0.8     # how often, when the soloist it last quoted played over these chords too, it stays with them
FEW = 6        # chords played by fewer soloists than this are quoted less often: it would say the same thing again


def quote(library, rng, chords, low, high, near, like=None, most=LONGEST):
    """A phrase of `library` (DATA's form) to play over `chords`, the (root, kind in listener.CHORDS) of each beat
    of the bars to come: one played whole over the same chords from the first bar on, a fourth up or a tone down as
    they may be, `most` bars long at most. Of the lengths someone played over these chords (four bars of a
    ii-V-I-IV, three of a ii-V-I, a turnaround, one bar...), it takes the one that leaves the bars after it to the
    fewest phrases: it ends where the next progression begins, and the longest of those. By the soloist named
    `like` when there is one of theirs. Drawn with `rng`, in the octave that keeps it from `low` to `high` and
    nearest the pitch `near`.
    Returns ("who · the tune", [(beats from the bar line, beats it lasts, pitch), ...], the bars it lasts), or None
    when nothing was played over these chords."""
    under = [(root, kind_of(kind)) for root, kind in chords]
    end = len(under) // 4 if most >= LONGEST else min(most, len(under) // 4)   # bars it plans over

    @functools.cache
    def played(first, bars):
        return library['licks'].get(key_of(under[4 * first:4 * (first + bars)]), ()) if first + bars <= end else ()

    @functools.cache
    def phrases(first):
        """The fewest phrases that cover the bars from `first` on; a bar nobody played counts as one."""
        if first >= end:
            return 0
        return 1 + min([phrases(first + bars) for bars in range(1, LONGEST + 1) if played(first, bars)] or [phrases(first + 1)])

    found = sorted((phrases(bars), -bars) for bars in range(1, LONGEST + 1) if played(0, bars))
    first = next((k for k, (_, bars) in enumerate(found) if rng.random() < LONGER * min(len(played(0, -bars)) / FEW, 1)), 0)
    for _, bars in found[first:] + found[:first]:
        licks = played(0, -bars)
        theirs = [lick for lick in licks if library['players'][lick[0]][0] == like]
        for player, lick in (rng.sample(theirs, min(len(theirs), 8)) if rng.random() < SAME else []) + \
                rng.sample(licks, min(len(licks), 8)):
            pitches = sorted(pitch for *_, pitch in lick)
            octaves = [12 * k + under[0][0] for k in range(11)
                       if 12 * k + under[0][0] + pitches[0] >= low and 12 * k + under[0][0] + pitches[-1] <= high]
            if octaves:
                shift = min(octaves, key=lambda shift: abs(shift + pitches[len(pitches) // 2] - near))
                return ' · '.join(library['players'][player]), [(at, lasts, pitch + shift) for at, lasts, pitch in lick], -bars
    return None
