"""
Makes licks.json.gz, the phrases the duo quotes when it solos over a tune (licks.py), from the Weimar Jazz Database of
transcribed solos (wjazzd.db, from https://jazzomat.hfm-weimar.de, under the ODbL: so is licks.json.gz):

    python -m music_transformer.make_licks wjazzd.db

A lick is a phrase of a solo in 4/4 as it was played, or several in a row, over one to four bars: it begins where
its soloist began a phrase and ends where they ended one (the database's phrases), kept with the chords of its bars.
Each note is (beats from the first bar line, beats it lasts, semitones above the pitch class of the first chord's
root), so that it is played in any key; a phrase that begins in the last UPBEAT beats of a bar and goes on is the
next bar's, and those notes are its upbeat, before the bar line. Licks are filed under their chords: each one's kind
(bebop.DEGREES'), its root in semitones above the first one's, and the beats it lasts.
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
BEATS = 4
STYLES = ('TRADITIONAL', 'SWING', 'BEBOP', 'COOL', 'HARDBOP', 'POSTBOP', 'FUSION')
MOST = 600       # licks kept under the same chords, drawn by lot: nearly all of them
LEAST = 2        # chords with fewer licks than this are left out
SPAN = 19        # a lick of one or two bars wider than this many semitones is left out
WIDE = 24        # and one of three or four bars wider than this
LONGEST = 4      # bars of the longest licks
UPBEAT = 2       # beats before a bar line from which a phrase that goes on past it is an upbeat to the next bar


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
        notes = [(bar, beat - 1 + (tatum - 1) / max(division, 1), min(max(duration / beatdur, 0.1), 2), round(pitch))
                 for bar, beat, tatum, division, pitch, duration, beatdur in db.execute(
                     "select bar, beat, tatum, division, pitch, duration, beatdur from melody where melid = ? "
                     "order by onset", (melid,))]
        phrases = db.execute("select start, end from sections where melid = ? and type = 'PHRASE' order by start",
                             (melid,)).fetchall()   # each one's first note and its last, counted through the solo
        for k, (begins, end) in enumerate(phrases):
            first = notes[begins][0] + (notes[begins][1] >= BEATS - UPBEAT and notes[end][0] > notes[begins][0])
            for _, end in phrases[k:]:   # as far as the end of this phrase, or of one after it
                length = notes[end][0] - first + 1
                if length > LONGEST:
                    break
                under = [chords.get((first + b, beat)) for b in range(length) for beat in range(1, BEATS + 1)]
                lick = [(round(((bar - first) * BEATS + at) * 12) / 12, lasts, pitch)
                        for bar, at, lasts, pitch in notes[begins:end + 1]]
                pitches = [pitch for *_, pitch in lick]
                inner = [sum(first + b == bar for bar, *_ in notes[begins:end + 1]) for b in range(1, length - 1)]
                if None in under or len(lick) < 3 * length + 1 or max(pitches) - min(pitches) > (SPAN, WIDE)[length > 2] \
                        or min(inner, default=2) < 2:   # no bar of it is silent
                    continue
                low = min(pitches) - (min(pitches) - under[0][0]) % 12   # the root's pitch at or under its lowest note
                yield key_of(under), (performer, title), [[round(at, 3), round(lasts, 2), pitch - low] for at, lasts, pitch in lick]


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
