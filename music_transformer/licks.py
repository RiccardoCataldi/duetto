"""
The phrases the duo quotes over a tune: phrases of the great soloists as they played them, one to four bars from
where they began one to where they ended one, filed under the chords they were played over and kept free of any key (licks.json.gz, made by make_licks.py from the Weimar Jazz
Database, under the ODbL). The duo keeps yours in a library of the same form, and quotes them the same way.
"""
import functools
import gzip
import json
import os

import numpy as np

from music_transformer.bebop import MOVES, kind_of
from music_transformer.make_licks import LONGEST, UPBEAT, key_of

with gzip.open(os.path.join(os.path.dirname(__file__), 'licks.json.gz'), 'rt') as f:
    DATA = json.load(f)


LONGER = 0.8   # how often it takes the phrase that fits the chords to come best, before it tries another
SAME = 0.8     # how often, when the soloist it last quoted played over these chords too, it stays with them
FEW = 6        # chords played by fewer soloists than this are quoted less often: it would say the same thing again
DRAWN = 8      # phrases it draws over the chords, to take the one that goes on best from where its line is


def pickup(last, target, room, scale, chromatic=True):
    """The pitches that lead from `last`, the last note of a phrase, into `target`, the first of the next, as bebop
    players join them: `room` of them at most, and never more than three. The one before the target approaches it
    by a semitone, from the side the line comes from (by a step of `scale`, 12 pitch classes, when `chromatic` is
    False: it falls on the beat); those before it walk up to it by steps of the scale, no further back than `last`.
    With no room to walk and room for two, they enclose the target: the step of the scale beyond it, then the
    approach. None when the approach would be `last` again."""
    way = 1 if target >= last else -1

    def step(pitch, way):
        return next(pitch + way * k for k in range(1, 13) if scale[(pitch + way * k) % 12])

    notes = [target - way if chromatic else step(target, -way)]
    if room < 1 or notes[0] == last:
        return []
    while len(notes) < min(room, 3) and (step(notes[0], -way) - last) * way > 0:
        notes.insert(0, step(notes[0], -way))
    if len(notes) == 1 and room > 1:
        notes.insert(0, step(target, way))
    return notes


def place(lick, root, low, high, near):
    """(how readily a soloist's line moves from the pitch `near` to the first note of `lick` (bebop.MOVES; of two
    leaps, the shorter), its notes) in the octave over the pitch class `root` that keeps it from `low` to `high`
    and goes on best from `near`; None when none keeps it there."""
    pitches = [pitch for *_, pitch in lick]
    leap = lambda shift: shift + pitches[0] - near
    return max((((MOVES[int(np.clip(leap(shift), -12, 12)) + 12], -abs(leap(shift))),
                 [(at, lasts, pitch + shift) for at, lasts, pitch in lick])
                for shift in range(root - 48, 132, 12) if shift + min(pitches) >= low and shift + max(pitches) <= high),
               key=lambda placed: placed[0], default=None)


def fit(notes, scale):
    """`notes`, (beats from a bar line, beats it lasts, pitch), of a phrase played over chords of the same kinds
    as those it sounds over now, but not the same (an altered dominant's over a plain one): `scale(beats)` is the
    scale of the chord there, 12 pitch classes. A note the ear rests on, on the beat or a beat long, that is outside
    the scale of the chord it sounds over moves a semitone into it: on the way the line was going, and not to the
    note before or after it when it can. The others stay as they were played: passing and approach notes."""
    fitted = []
    for k, (at, lasts, pitch) in enumerate(notes):
        inside = scale(at + min(lasts, 1) / 2)
        if (min(at % 1, 1 - at % 1) < 0.1 or lasts >= 1) and not inside[pitch % 12]:
            around = (fitted[-1][2] if fitted else pitch, notes[k + 1][2] if k + 1 < len(notes) else pitch)
            pitch += min((way for way in (-1, 1) if inside[(pitch + way) % 12]),
                         key=lambda way: (pitch + way in around, (pitch - around[0]) * way < 0), default=0)
        fitted.append((at, lasts, pitch))
    return fitted


def vary(lick, bars, rng):
    """`lick`, of `bars` bars, as a soloist plays a phrase again to develop it: as it was (over the chords that
    follow, a sequence), a beat later, its first half only, or upside down from its first note."""
    how = rng.randrange(4)
    if how == 1 and lick[2][0] + 1 < 4 * bars:   # (three notes of it, at least, still in its bars)
        return [[at + 1, lasts, pitch] for at, lasts, pitch in lick if at + 1 < 4 * bars]
    if how == 2:
        return lick[:max((len(lick) + 1) // 2, 3)]
    return [[at, lasts, 2 * lick[0][2] - pitch if how == 3 else pitch] for at, lasts, pitch in lick]


def quote(library, rng, chords, low, high, near, like=None, most=LONGEST):
    """A phrase of `library` (DATA's form) to play over `chords`, the (root, kind in listener.CHORDS) of each beat
    of the bars to come: one played whole over the same chords from the first bar on, a fourth up or a tone down as
    they may be, `most` bars long at most. Of the lengths someone played over these chords (four bars of a
    ii-V-I-IV, three of a ii-V-I, a turnaround, one bar...), it takes the one that leaves the bars after it to the
    fewest phrases: it ends where the next progression begins, and the longest of those. By the soloist named
    `like` when there is one of theirs. Of DRAWN of them, drawn with `rng`, the one whose first note goes on best
    from the pitch `near`, the last of its line, in the octave that keeps it from `low` to `high` (place).
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
        for pool in ([theirs] if theirs and rng.random() < SAME else []) + [licks]:
            placed = [(place(lick, under[0][0], low, high, near), player)
                      for player, lick in rng.sample(pool, min(len(pool), DRAWN))]
            best = max((drawn for drawn in placed if drawn[0]), key=lambda drawn: drawn[0][0], default=None)
            if best:
                return ' · '.join(library['players'][best[1]]), best[0][1], -bars
    return None
