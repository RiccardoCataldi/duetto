"""
The phrases the duo quotes over a tune: phrases of the great soloists as they played them, one to four bars from
where they began one to where they ended one, filed under the chords they were played over and kept free of any key (licks.json.gz, made by make_licks.py from the Weimar Jazz
Database, under the ODbL). The duo keeps yours in a library of the same form, and quotes them the same way.
"""
import functools
import gzip
import json
import math
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
SLOW = 100     # bpm: a solo no faster than this is a ballad's; over a ballad its phrases are quoted first, and their
SLOWER = 3     # rhythms borrowed this many times as often


def tempo(library, lick):
    """The tempo the solo of `lick` was played at, in bpm; 0 when the library does not say."""
    player = library['players'][lick[0]]
    return player[2] if len(player) > 2 else 0


@functools.cache
def rhythms(fastest):
    """The rhythms of DATA's phrases with no two notes less than `fastest` beats apart, each once: (the bars it
    lasts, ((beats from its first bar line, beats it lasts), ...), whether its solo was SLOW)."""
    found = {}
    for key, cells in DATA['licks'].items():
        bars = sum(int(chord.split('.')[2]) for chord in key.split()) // 4
        for lick in cells:
            notes = tuple((at, lasts) for at, lasts, _ in lick[1])
            if all(b[0] - a[0] > fastest - 1e-6 for a, b in zip(notes, notes[1:])):
                found[notes] = (bars, notes, found.get(notes, (0, 0, False))[2] or 0 < tempo(DATA, lick) <= SLOW)
    return sorted(found.values())


def rhythm(rng, density, fastest, keep=1.0, ballad=True):
    """A rhythm for a phrase of its own, borrowed from a soloist's: (the bars it lasts, [(beats from its first bar
    line, beats it lasts), ...]). One of about `density` notes a bar, over a `ballad` a ballad's rather than
    another's (SLOWER); of its notes between the first and the last, `keep` stay, by chance: the fewer, the more room
    in it."""
    pool = rhythms(fastest)
    bars, notes, _ = rng.choices(pool, [math.exp(-abs(len(notes) / bars - density)) * (SLOWER if slow and ballad else 1)
                                        for bars, notes, slow in pool])[0]
    return bars, [note for k, note in enumerate(notes) if k in (0, len(notes) - 1) or rng.random() < keep]


def enclose(target, scale, how):
    """The two pitches that lead into the pitch `target` as bebop players do: `how` 0 or 1, they enclose it, the
    step of `scale` (12 pitch classes) over it then a semitone under it, or the step under it then a semitone over
    it; 2 or 3, a double chromatic approach, two semitones down to it from over it, or up from under it."""
    side = 1 if how % 2 == 0 else -1
    if how < 2:
        return [next(target + side * k for k in range(1, 13) if scale[(target + side * k) % 12]), target - side]
    return [target + 2 * side, target + side]


def settle(notes, low, high, near):
    """`notes`, (beats, beats it lasts, pitch), of a phrase of a tune's theme, in the octave that keeps the most of
    it from `low` to `high` and begins nearest the pitch `near`."""
    shift = min(range(-36, 48, 12), key=lambda shift: (sum(not low <= pitch + shift <= high for *_, pitch in notes),
                                                       abs(notes[0][2] + shift - near)))
    return [(at, lasts, pitch + shift) for at, lasts, pitch in notes]


def paraphrase(notes, rng, plain=False):
    """`notes` of a phrase of a tune's theme as a soloist states it: as it is written; the notes the ear rests on
    only (a beat long or more, its first and its last); or those, and the others its own: their pitch None, for its
    line to find on the way to the next of them. `plain`: as it is written, always."""
    if plain:
        return list(notes)
    how, rests = rng.randrange(3), lambda k: notes[k][1] >= 1 or k in (0, len(notes) - 1)
    if how == 1 and sum(map(rests, range(len(notes)))) >= 3:
        return [note for k, note in enumerate(notes) if rests(k)]
    if how == 2:
        return [(at, lasts, pitch if rests(k) else None) for k, (at, lasts, pitch) in enumerate(notes)]
    return list(notes)


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


def scaled(lick, factor):
    """`lick`'s notes (beats from the bar line, beats it lasts, pitch) played `factor` times as slowly: 2, in half
    time; 0.5, in double time. An upbeat that would begin more than UPBEAT beats before the bar line is left out."""
    return [[at * factor, lasts * factor, pitch] for at, lasts, pitch in lick if at * factor >= -UPBEAT]


def vary(lick, bars, rng, fastest=0):
    """`lick`, of `bars` bars, as a soloist plays a phrase again to develop it: (its notes, the bars they last). As it
    was (over the chords that follow, a sequence), a beat later, its first half only, upside down from its first
    note, backwards (its pitches in the other order, on its rhythm), a tone higher or lower (the duo fits it to the
    scale), twice as slow (when that keeps it within LONGEST bars) or twice as fast (when no two of its notes come
    closer than `fastest` beats); otherwise as it was."""
    how = rng.randrange(8)
    if how == 1 and lick[2][0] + 1 < 4 * bars:   # (three notes of it, at least, still in its bars)
        return [[at + 1, lasts, pitch] for at, lasts, pitch in lick if at + 1 < 4 * bars], bars
    if how == 2:
        return lick[:max((len(lick) + 1) // 2, 3)], bars
    if how == 4:
        return [[at, lasts, pitch] for (at, lasts, _), (*_, pitch) in zip(lick, lick[::-1])], bars
    if how == 5:
        tone = rng.choice((-2, 2))
        return [[at, lasts, pitch + tone] for at, lasts, pitch in lick], bars
    if how == 6 and 2 * bars <= LONGEST:
        return scaled(lick, 2), 2 * bars
    if how == 7 and all(b[0] - a[0] > 2 * fastest - 1e-6 for a, b in zip(lick, lick[1:])):
        return scaled(lick, 0.5), -(-bars // 2)
    return [[at, lasts, 2 * lick[0][2] - pitch if how == 3 else pitch] for at, lasts, pitch in lick], bars


PACES = ((1, ''), (0.5, 'double time'), (2, 'half time'))   # how a phrase may be played over chords nobody played
                                                            # it over: as it was, over the same chords twice as long,
                                                            # at twice its pace; or over them half as long, at half
PACED = 0.15  # how often it quotes a phrase played over its chords at another pace than theirs, when there is one
OVER = 0.15   # how often it hears the chords as a bebop soloist may, to play over them (superimposed), when it can


def superimposed(under):
    """`under`, the (root, kind) of each beat, as a bebop soloist may hear it to play over it: a dominant that
    resolves, a fourth up, as the dominant a tritone away (its substitution: the altered scale); another, of a bar or
    more, as its ii and itself, half each (a ii-V over its V). The same list when it has no dominant."""
    runs = []
    for chord in under:
        if runs and runs[-1][0] == chord:
            runs[-1][1] += 1
        else:
            runs.append([chord, 1])
    over = []
    for k, ((root, kind), beats) in enumerate(runs):
        if kind == 'dom' and k + 1 < len(runs) and runs[k + 1][0][0] == (root + 5) % 12:
            over += [((root + 6) % 12, 'dom')] * beats
        elif kind == 'dom' and beats >= 4 and beats % 2 == 0:
            over += [((root + 7) % 12, 'min')] * (beats // 2) + [(root, 'dom')] * (beats // 2)
        else:
            over += [(root, kind)] * beats
    return over


def quote(library, rng, chords, low, high, near, like=None, most=LONGEST, fastest=0, slow=False):
    """A phrase of `library` (DATA's form) to play over `chords`, the (root, kind in listener.CHORDS) of each beat
    of the bars to come: one played whole over the same chords from the first bar on, a fourth up or a tone down as
    they may be, `most` bars long at most. Of the lengths someone played over these chords (four bars of a
    ii-V-I-IV, three of a ii-V-I, a turnaround, one bar...), it takes the one that leaves the bars after it to the
    fewest phrases: it ends where the next progression begins, and the longest of those. By the soloist named
    `like` when there is one of theirs. Of DRAWN of them, drawn with `rng`, the one whose first note goes on best
    from the pitch `near`, the last of its line, in the octave that keeps it from `low` to `high` (place). A phrase
    with two notes less than `fastest` beats apart is left out: a horn's run from a slower tune, a blur at this one's
    tempo. `slow`: over a ballad, a phrase from a solo played as slowly (SLOW), when there is one, as often as SAME.
    As often as PACED, and where nobody played over these chords, one played over them in double or half time
    (PACES); as often as OVER, and where nobody played that either, one played over the chords it hears over them
    (superimposed), as it was.
    Returns ("who · the tune", [(beats from the bar line, beats it lasts, pitch), ...], the bars it lasts, whether it
    is superimposed: outside these chords, as it was meant), or None when nothing was played over these chords."""
    under = [(root, kind_of(kind)) for root, kind in chords]
    asked = (library, rng, low, high, near, like, most, fastest, slow)
    found = _quote(under, PACES[:1], *asked)
    if found is None or rng.random() < PACED:
        found = _quote(under, PACES[1:], *asked) or found
    over = superimposed(under)
    if over != under and (found is None or rng.random() < OVER):
        found = _quote(over, PACES[:1], *asked, True) or found
    return found


def _quote(under, paces, library, rng, low, high, near, like, most, fastest, slow, outside=False):
    """quote(), over `under`, the (root, kind in bebop.DEGREES) of each beat, of the phrases played over them at
    `paces` (PACES'): the phrase it found, `outside` or not, or None."""
    end = len(under) // 4 if most >= LONGEST else min(most, len(under) // 4)   # bars it plans over

    @functools.cache
    def played(first, bars):
        """The phrases played over the bars from `first`, and how ('', or one of PACES)."""
        chords = under[4 * first:4 * (first + bars)]
        for factor, pace in paces if first + bars <= end else ():
            source = [chord for chord in chords for _ in (0, 1)] if factor < 1 else chords[::2] if factor > 1 else chords
            if bars / factor > LONGEST or bars / factor % 1 or factor > 1 and \
                    [chord for chord in source for _ in (0, 1)] != chords:   # half time: each chord an even number of beats
                continue
            licks = [[player, scaled(notes, factor) if factor != 1 else notes]
                     for player, notes in library['licks'].get(key_of(source), ())]
            licks = [lick for lick in licks if all(b[0] - a[0] > fastest - 1e-6 for a, b in zip(lick[1], lick[1][1:]))]
            if licks:
                return licks, pace
        return (), ''

    @functools.cache
    def phrases(first):
        """The fewest phrases that cover the bars from `first` on; a bar nobody played counts as one."""
        if first >= end:
            return 0
        return 1 + min([phrases(first + bars) for bars in range(1, LONGEST + 1) if played(first, bars)[0]] or
                       [phrases(first + 1)])

    found = sorted((phrases(bars), -bars) for bars in range(1, LONGEST + 1) if played(0, bars)[0])
    first = next((k for k, (_, bars) in enumerate(found) if rng.random() < LONGER * min(len(played(0, -bars)[0]) / FEW, 1)), 0)
    for _, bars in found[first:] + found[:first]:
        licks, pace = played(0, -bars)
        theirs = [lick for lick in licks if library['players'][lick[0]][0] == like]
        slower = [lick for lick in licks if 0 < tempo(library, lick) <= SLOW] if slow else []
        for pool in ([slower] if slower and rng.random() < SAME else []) + \
                ([theirs] if theirs and rng.random() < SAME else []) + [licks]:
            placed = [(place(lick, under[0][0], low, high, near), player)
                      for player, lick in rng.sample(pool, min(len(pool), DRAWN))]
            best = max((drawn for drawn in placed if drawn[0]), key=lambda drawn: drawn[0][0], default=None)
            if best:
                how = [pace] * bool(pace) + ['superimposed'] * outside
                return ' · '.join(library['players'][best[1]][:2] + how), best[0][1], -bars, outside
    return None
