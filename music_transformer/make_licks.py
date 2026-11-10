"""
Makes licks.json.gz, the phrases the duo quotes when it solos over a tune (licks.py), from the Weimar Jazz Database of
transcribed solos (wjazzd.db, from https://jazzomat.hfm-weimar.de, under the ODbL: so is licks.json.gz):

    python -m music_transformer.make_licks wjazzd.db

A lick is one to four bars of a solo in 4/4 as they were played, kept with the chords under them: each note is
(beats from the bar line, beats it lasts, semitones above the pitch class of the first chord's root), so that it is
played in any key. Licks are filed under their chords: each one's kind (bebop.DEGREES'), its root in semitones above
the first one's, and the beats it lasts.
"""
import collections
import gzip
import json
import os
import random
import re
import sqlite3
import sys

ROOTS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
STYLES = ('TRADITIONAL', 'SWING', 'BEBOP', 'COOL', 'HARDBOP', 'POSTBOP', 'FUSION')
MOST = 600       # licks kept under the same chords, drawn by lot: nearly all of them
LEAST = 2        # chords with fewer licks than this are left out
SPAN = 19        # a lick of one or two bars wider than this many semitones is left out
WIDE = 24        # and one of three or four bars wider than this
LONGEST = 4      # bars of the longest licks


def chord(name):
    """(root, kind) of one of the database's chord symbols, or None."""
    match = re.match(r'([A-G])([#b]?)(.*)', name or '')
    if not match:
        return None
    root = (ROOTS[match.group(1)] + {'#': 1, 'b': -1, '': 0}[match.group(2)]) % 12
    kind = match.group(3).split('/')[0]
    kind = 'hdim' if kind.startswith('m7b5') else 'dim' if kind.startswith('o') else 'min' if kind.startswith('-') \
        else 'maj' if kind.startswith(('j', '6')) or kind == '' else 'dom'
    return root, kind


def key_of(chords):
    """The name licks are filed under for `chords`, the (root, kind) of each beat: 'min.0.4 dom.5.4'."""
    runs = []
    for root, kind in chords:
        if runs and runs[-1][:2] == [root, kind]:
            runs[-1][2] += 1
        else:
            runs.append([root, kind, 1])
    return ' '.join(f"{kind}.{(root - runs[0][0]) % 12}.{beats}" for root, kind, beats in runs)


def licks(db):
    solos = db.execute(f"select melid, performer, title from solo_info where signature = '4/4' and style in {STYLES}")
    for melid, performer, title in solos.fetchall():
        chords, last = {}, None
        for bar, beat, name in db.execute("select bar, beat, chord from beats where melid = ? order by onset", (melid,)):
            last = chord(name) or last
            if last:
                chords[(bar, beat)] = last
        bars = collections.defaultdict(list)
        for bar, beat, tatum, division, pitch, duration, beatdur in db.execute(
                "select bar, beat, tatum, division, pitch, duration, beatdur from melody where melid = ? order by onset",
                (melid,)):
            if beatdur:
                bars[bar].append((beat - 1 + (tatum - 1) / max(division, 1), min(max(duration / beatdur, 0.1), 2), round(pitch)))
        for first in bars:
            for length in range(1, LONGEST + 1):
                under = [chords.get((first + b, beat)) for b in range(length) for beat in range(1, 5)]
                notes = [(round((b * 4 + at) * 12) / 12, lasts, pitch) for b in range(length) for at, lasts, pitch in bars.get(first + b, [])]
                pitches = [pitch for *_, pitch in notes]
                later = [len(bars.get(first + b, [])) for b in range(1, length)]   # no bar of it is silent
                if None in under or len(notes) < 3 * length + 1 or max(pitches) - min(pitches) > (SPAN, WIDE)[length > 2] \
                        or (length > 1 and (min(later[:-1], default=2) < 2 or later[-1] < 4 - length or later[-1] < 1
                                            or len(set(under)) == 1 and first % length)):
                    continue
                low = min(pitches) - (min(pitches) - under[0][0]) % 12   # the root's pitch at or under its lowest note
                yield key_of(under), (performer, title), [[round(at, 3), round(lasts, 2), pitch - low] for at, lasts, pitch in notes]


if __name__ == '__main__':
    found, players = collections.defaultdict(list), {}
    for key, player, notes in licks(sqlite3.connect(sys.argv[1])):
        if notes not in [lick[1] for lick in found[key]]:
            found[key].append([players.setdefault(player, len(players)), notes])
    rng = random.Random(0)
    kept = {key: sorted(rng.sample(cells, min(len(cells), MOST))) for key, cells in sorted(found.items()) if len(cells) >= LEAST}
    out = os.path.join(os.path.dirname(__file__), 'licks.json.gz')
    with gzip.open(out, 'wt') as f:
        json.dump({'source': 'Weimar Jazz Database, https://jazzomat.hfm-weimar.de, ODbL', 'players': list(players),
                   'licks': kept}, f, separators=(',', ':'))
    print(f"{sum(map(len, kept.values()))} licks under {len(kept)} chord sequences, {os.path.getsize(out) // 1024} kB")
