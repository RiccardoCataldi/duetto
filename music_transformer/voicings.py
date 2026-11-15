"""
The voicings a jazz pianist comps with over a bass that plays the roots: rootless, of four notes, in two inversions
(Bill Evans's A, from the third, and B, from the seventh), a dominant with its 13th in place of its 5th; or their
shell, the third and the seventh alone, to leave you room while you play. From one chord to the next the hand moves
as little as it can: the inversion nearest the last voicing is played, as in a II V I.
"""

# for each kind of chord (listener.CHORDS), its two rootless voicings, in semitones above its root from low to high;
# the first and third notes of each are its shell: the third and the seventh
_MAJOR = ((4, 7, 9, 14), (9, 14, 16, 19))          # 3 5 6 9 and 6 9 3 5: a chart's plain C is played as C6/9
_MINOR = ((3, 7, 10, 14), (10, 14, 15, 19))        # b3 5 b7 9 and b7 9 b3 5
_SUS = ((5, 9, 10, 14), (10, 14, 17, 21))          # 4 13 b7 9 and b7 9 4 13
_DIM = ((0, 3, 6, 9), (6, 9, 12, 15))
ROOTLESS = {
    'maj7': ((4, 7, 11, 14), (11, 14, 16, 19)),    # 3 5 7 9 and 7 9 3 5
    '6': _MAJOR, '': _MAJOR,
    'm7': _MINOR, 'm': _MINOR,
    'm6': ((3, 7, 9, 14), (9, 14, 15, 19)),
    'mMaj7': ((3, 7, 11, 14), (11, 14, 15, 19)),
    '7': ((4, 9, 10, 14), (10, 14, 16, 21)),       # 3 13 b7 9 and b7 9 3 13
    '7sus4': _SUS, 'sus4': _SUS,
    'm7b5': ((3, 6, 10, 12), (10, 12, 15, 18)),    # b3 b5 b7 1: its ninth is no note of its scale
    'dim7': _DIM, 'dim': _DIM,
    '7#5': ((4, 8, 10, 14), (10, 14, 16, 20)), '+': ((4, 8, 10, 14), (10, 14, 16, 20)),
}


def fitted(shape, root, kind, scale):
    """`shape` with its tensions as the chord's `scale` has them (for each pitch class, 1 if it is in it): a ninth or
    a thirteenth outside it is flattened over a dominant (D7 to G minor: b9 and b13), a ninth gives way to the root
    over the others (Em7 in C: no F#)."""
    def fit(step):
        if scale[(root + step) % 12]:
            return step
        if step % 12 == 2:   # the ninth
            return step - 1 if kind.startswith('7') and scale[(root + step - 1) % 12] else step - 2
        return step - 1 if scale[(root + step - 1) % 12] else step
    return tuple(sorted(fit(step) for step in shape))


def distance(voicing, before):
    """How far the hand moves from `before` to `voicing`: each note of either to the nearest of the other."""
    return sum(min(abs(p - q) for q in before) for p in voicing) + sum(min(abs(p - q) for q in voicing) for p in before)


def voicing(root, kind, scale, low, high, before=(), shell=False, other=False):
    """The pitches of the voicing of the chord `root` `kind` within `low` and `high` nearest `before`, the last one
    played (with none, nearest the middle of the register); `shell`: its third and seventh only; `other`: not
    `before` again, when the chord goes on. () when the register has room for none."""
    found = []
    for shape in ROOTLESS[kind]:
        shape = fitted(shape, root, kind, scale)
        if shell:
            shape = (shape[0], shape[2])
        for octave in range(11):
            pitches = tuple(12 * octave + root + step for step in shape)
            if low <= pitches[0] and pitches[-1] <= high and not (other and pitches == tuple(before)):
                found.append(pitches)
    if not found:
        return ()
    if before:
        return min(found, key=lambda pitches: (distance(pitches, before), pitches))
    return min(found, key=lambda pitches: (abs(sum(pitches) / len(pitches) - (low + high) / 2), pitches))
