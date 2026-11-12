import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_transformer.listener import NAMES, Listener

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 62, 65, 69, 71, 67]
EB_MAJOR = [p + 3 for p in C_MAJOR]


def play(listener, pitches, start=0.0, step=0.25):
    for i, pitch in enumerate(pitches):
        listener.hear(start + i * step, 0.2, pitch)
    return start + len(pitches) * step


def label(listener, now):
    """The name of the chord it hears you play at `now`, e.g. G7 (C6 rather than Am7 when C is in the bass)."""
    harmony = listener.harmony(now)
    return NAMES[harmony[0]] + harmony[1] if harmony else None


class ListenerTest(unittest.TestCase):
    def test_it_finds_your_key_and_follows_a_modulation_within_a_bar_or_two(self):
        listener = Listener()
        end = play(listener, C_MAJOR)
        self.assertEqual(listener.name(listener.key()[0]), 'C major')
        keys = []
        for i, pitch in enumerate(EB_MAJOR):
            listener.hear(end + i * 0.25, 0.2, pitch)
            keys.append(listener.name(listener.key()[0]))
        self.assertEqual(keys[-1], 'Eb major')
        self.assertLessEqual(keys.index('Eb major'), 10)
        self.assertEqual(keys[keys.index('Eb major'):], ['Eb major'] * (len(keys) - keys.index('Eb major')))

    def test_a_lower_stay_follows_a_modulation_sooner(self):
        heard = []
        for stay in (0.97, 0.85):
            listener = Listener()
            listener.stay(stay)
            end = play(listener, C_MAJOR)
            for i, pitch in enumerate(EB_MAJOR):
                listener.hear(end + i * 0.25, 0.2, pitch)
                if listener.name(listener.key()[0]) == 'Eb major':
                    heard.append(i)
                    break
        self.assertLess(heard[1], heard[0])

    def test_a_passing_note_does_not_change_the_key(self):
        listener = Listener()
        end = play(listener, C_MAJOR)
        listener.hear(end, 0.2, 66)
        self.assertEqual(listener.name(listener.key()[0]), 'C major')

    def test_it_names_the_chord_you_hold(self):
        listener = Listener()
        play(listener, C_MAJOR)
        for pitch in (55, 59, 62, 65):   # G7
            listener.hear(10.0, 1.0, pitch)
        self.assertEqual(listener.harmony(10.3)[2], {7, 11, 2, 5})

    def test_a_chord_you_hold_sounds_past_the_guess_until_you_let_it_go(self):
        for releases in (False, True):
            listener = Listener()
            play(listener, C_MAJOR)
            if releases:   # you say when your notes stop sounding
                listener.release(0.0, C_MAJOR[0], 0.2)
            for pitch in (43, 59, 62, 65):   # G7, G in the bass, its duration guessed at 0.3 s
                listener.hear(10.0, 0.3, pitch)
            self.assertEqual(label(listener, 14.0), 'G7' if releases else None)
        for pitch in (43, 59, 62, 65):
            listener.release(10.0, pitch, 1.0)
        self.assertIsNone(listener.harmony(14.0))

    def test_it_hears_jazz_chords_and_tells_them_apart_by_the_bass(self):
        for pitches, name in (((48, 52, 55, 59), 'Cmaj7'), ((48, 52, 55, 57), 'C6'), ((45, 52, 55, 60), 'Am7'),
                              ((47, 53, 57, 62), 'Bm7b5'), ((43, 48, 50, 53), 'G7sus4')):
            listener = Listener()
            play(listener, C_MAJOR)
            for pitch in pitches:
                listener.hear(10.0, 1.0, pitch)
            self.assertEqual(label(listener, 10.3), name)

    def test_it_hears_where_your_phrase_ends(self):
        listener = Listener()
        listener.fix(100)                            # a beat is 0.6 s: a rest of 0.9 s ends a phrase
        end = play(listener, C_MAJOR, start=0.1, step=0.3)
        self.assertEqual(listener.phrase(end - 0.2), [])
        self.assertEqual(listener.phrase(end + 1.0), C_MAJOR)


class ChartTest(unittest.TestCase):
    def over(self, root, kind, key):
        listener = Listener()
        listener.chart = (lambda now: (root, kind), key)
        return listener, listener.bias(0.0, 1.0, 1.0, 2.0)

    def test_a_chart_spells_a_dominant_with_a_raised_fifth_as_it_sounds(self):
        from music_transformer import chart
        for name in ('C7#5', 'C7+', 'C+7', 'Caug7', 'C7(#5)'):
            self.assertEqual(chart.chord(name), (0, '7#5'), name)
        self.assertEqual(chart.chord('Fm9'), (5, 'm7'))
        listener, bias = self.over(0, '7#5', 5)       # C7#5 to F minor: G# and Bb, never G
        self.assertEqual(sorted(listener.harmony(0.0)[2]), [0, 4, 8, 10])
        self.assertLess(bias[67], -9)
        self.assertGreater(min(bias[p] for p in (60, 64, 68, 70)), 0)

    def test_over_a_chart_it_keeps_to_the_scale_of_the_chord(self):
        listener, bias = self.over(0, 'm7', 10)       # Cm7 in Bb major: C dorian
        self.assertEqual(sorted(np.flatnonzero(listener.chord_scale(0.0))), [0, 2, 3, 5, 7, 9, 10])
        for out in (61, 64, 66, 68, 71):              # Db E F# Ab B: none of B major's own notes gets in
            self.assertLess(bias[out], -9)
        self.assertGreater(min(bias[p] for p in (60, 62, 63, 65, 67, 69, 70)), -1)
        listener, _ = self.over(2, '7', 19)           # D7 in G minor: with its b9 and b13
        self.assertEqual(sorted(np.flatnonzero(listener.chord_scale(0.0))), [0, 2, 3, 6, 7, 9, 10])

    def test_over_a_chart_its_bass_plays_the_chord_and_its_root_most_of_all(self):
        _, bias = self.over(0, 'm7', 10)
        self.assertEqual(int(np.argmax(bias[36:48])) + 36, 36)        # C2
        self.assertLess(bias[38], -8)                 # D2 is in the scale, but no bass note of Cm7
        self.assertGreater(bias[43], bias[39])        # its fifth over its third
        self.assertGreater(bias[62], -1)              # up there D is a note of the line


class PaletteTest(unittest.TestCase):
    def test_a_soloist_plays_the_changes_in_more_ways_than_the_chords_scale(self):
        listener = Listener()
        listener.chart = (lambda now: (10, '7'), 5)   # Bb7 in a blues in F
        ways = {name: sorted(np.flatnonzero(listener.palette(0.0, name))) for name in
                ('scale', 'arpeggio', 'pentatonic', 'blues', 'major_pentatonic')}
        self.assertEqual(ways['scale'], [0, 2, 3, 5, 7, 8, 10])           # Bb mixolydian
        self.assertEqual(ways['arpeggio'], [2, 5, 8, 10])
        self.assertEqual(ways['pentatonic'], [0, 2, 5, 7, 8, 10])         # Bb major pentatonic, and the chord's Ab
        self.assertEqual(ways['blues'], [0, 3, 5, 8, 10, 11])             # F blues, whatever the chord
        self.assertEqual(ways['major_pentatonic'], [0, 2, 5, 7, 9])       # F major pentatonic
        self.assertLess(listener.bias(0.0, 1.0, 1.0, 2.0, 'blues')[62], -8)   # D4 is not in F blues
        self.assertGreater(listener.bias(0.0, 1.0, 1.0, 2.0, 'blues')[68], -1)       # Ab4 is, and is Bb7's seventh


class AvoidTest(unittest.TestCase):
    def test_over_a_c_minor_the_major_third_clashes_whatever_the_key(self):
        listener = Listener()
        listener.chart = (lambda now: (0, 'm'), 0)   # a C minor chord in a tune in C major
        bias = listener.bias(8.0, 1.0, 1.0, 0.5)
        self.assertLess(bias[64], bias[63] - 3)      # E against Eb
        self.assertGreater(bias[62], bias[64])       # its ninth is no avoid note


if __name__ == '__main__':
    unittest.main()
