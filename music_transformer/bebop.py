"""
How bebop and hard bop soloists play the changes: counted over the 132 solos of those styles (25 players, Charlie
Parker's 17 among them; 55486 notes) transcribed in the Weimar Jazz Database (https://jazzomat.hfm-weimar.de, ODbL).
Only these counts are kept here, not the solos.
"""

# for each kind of chord, how often each of the 12 steps above its root is played: (off the beat, on the beat)
DEGREES = {
    'maj': ((0.141, 0.025, 0.127, 0.041, 0.13, 0.077, 0.031, 0.166, 0.03, 0.09, 0.029, 0.114),
               (0.16, 0.021, 0.119, 0.034, 0.175, 0.054, 0.024, 0.178, 0.019, 0.087, 0.025, 0.104)),
    'dom': ((0.132, 0.053, 0.103, 0.046, 0.097, 0.091, 0.046, 0.14, 0.048, 0.088, 0.113, 0.044),
               (0.157, 0.045, 0.092, 0.046, 0.12, 0.079, 0.035, 0.136, 0.041, 0.095, 0.122, 0.031)),
    'min': ((0.133, 0.037, 0.121, 0.139, 0.035, 0.109, 0.038, 0.142, 0.041, 0.057, 0.094, 0.055),
               (0.152, 0.031, 0.114, 0.156, 0.021, 0.109, 0.033, 0.148, 0.035, 0.068, 0.082, 0.052)),
    'hdim': ((0.107, 0.087, 0.042, 0.13, 0.036, 0.152, 0.11, 0.044, 0.076, 0.051, 0.116, 0.048),
                (0.123, 0.081, 0.039, 0.154, 0.061, 0.11, 0.116, 0.037, 0.066, 0.079, 0.092, 0.044)),
    'dim': ((0.103, 0.09, 0.064, 0.096, 0.083, 0.056, 0.096, 0.056, 0.085, 0.1, 0.062, 0.109),
               (0.075, 0.097, 0.086, 0.081, 0.065, 0.016, 0.118, 0.059, 0.097, 0.129, 0.081, 0.097)),
}
# how often the line moves by each interval, in semitones from 12 down to 12 up (wider ones count as 12)
INTERVALS = (0.0014, 0.0004, 0.0014, 0.004, 0.0041, 0.0065, 0.006, 0.0274, 0.0516, 0.0859, 0.1656, 0.1528, 0.0383, 0.1425, 0.1303, 0.0863, 0.0385, 0.0259, 0.006, 0.0105, 0.0052, 0.0041, 0.0022, 0.0014, 0.0015)


import numpy as np

FLOOR = 0.01   # no step or interval counts as rarer than this: it is discouraged, never forbidden


def kind_of(kind):
    """The kind of chord here (DEGREES) of one of listener.CHORDS."""
    if kind == 'm7b5':
        return 'hdim'
    if kind.startswith('dim'):
        return 'dim'
    if kind.startswith('m') and kind != 'maj7':
        return 'min'
    return 'maj' if kind in ('', 'maj7', '6') else 'dom'   # sus and + are played as dominants


def degrees(root, kind, on_beat):
    """For each pitch class, the logit a bebop soloist's habits add to it over the chord `root` `kind` (one of
    listener.CHORDS), on the beat or off it: 0 for the step played most, less for the others."""
    often = np.maximum(DEGREES[kind_of(kind)][bool(on_beat)], FLOOR)
    return np.roll(np.log(often / often.max()), root)


MOVES = np.log(np.maximum(INTERVALS, FLOOR) / max(INTERVALS))


def moves(last):
    """For each of the 128 pitches, the logit the way their lines move adds to it after the pitch `last`: mostly
    by step, a little more often down, seldom the same note again."""
    return MOVES[np.clip(np.arange(128) - last, -12, 12) + 12]
