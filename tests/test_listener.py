import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_transformer.listener import Listener

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 62, 65, 69, 71, 67]
EB_MAJOR = [p + 3 for p in C_MAJOR]


def play(listener, pitches, start=0.0, step=0.25):
    for i, pitch in enumerate(pitches):
        listener.hear(start + i * step, 0.2, pitch)
    return start + len(pitches) * step


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

    def test_the_key_it_says_follows_a_modulation_without_flickering(self):
        listener = Listener()
        self.assertIsNone(listener.said_key())
        end = play(listener, C_MAJOR)
        said = []
        for i, pitch in enumerate(EB_MAJOR * 2):
            listener.hear(end + i * 0.25, 0.2, pitch)
            said.append(listener.name(listener.said_key()))
        self.assertLessEqual(said.index('Eb major'), 20)
        changes = [b for a, b in zip(said, said[1:]) if a != b]
        self.assertEqual(changes, ['Eb major'])

    def test_it_keeps_saying_a_key_while_no_other_is_sure(self):
        listener = Listener()
        play(listener, C_MAJOR)
        listener.belief[:] = 0
        listener.belief[[0, 7, 9 + 12]] = [0.3, 0.4, 0.3]   # C major, G major, A minor: none sure
        listener.hear(10.0, 0.2, 67)
        self.assertEqual(listener.name(listener.said_key()), 'C major')

    def test_it_names_the_chord_you_hold(self):
        listener = Listener()
        play(listener, C_MAJOR)
        for pitch in (55, 59, 62, 65):   # G7
            listener.hear(10.0, 1.0, pitch)
        self.assertEqual(listener.chord(10.3), {7, 11, 2, 5})

    def test_a_chord_you_hold_sounds_past_the_guess_until_you_let_it_go(self):
        for releases in (False, True):
            listener = Listener()
            play(listener, C_MAJOR)
            if releases:   # you say when your notes stop sounding
                listener.release(0.0, C_MAJOR[0], 0.2)
            for pitch in (43, 59, 62, 65):   # G7, G in the bass, its duration guessed at 0.3 s
                listener.hear(10.0, 0.3, pitch)
            self.assertEqual(listener.chord_label(14.0), 'G7' if releases else None)
        for pitch in (43, 59, 62, 65):
            listener.release(10.0, pitch, 1.0)
        self.assertIsNone(listener.chord(14.0))

    def test_it_hears_jazz_chords_and_tells_them_apart_by_the_bass(self):
        for pitches, name in (((48, 52, 55, 59), 'Cmaj7'), ((48, 52, 55, 57), 'C6'), ((45, 52, 55, 60), 'Am7'),
                              ((47, 53, 57, 62), 'Bm7b5'), ((43, 48, 50, 53), 'G7sus4')):
            listener = Listener()
            play(listener, C_MAJOR)
            for pitch in pitches:
                listener.hear(10.0, 1.0, pitch)
            self.assertEqual(listener.chord_label(10.3), name)

    def test_the_tensions_of_a_major_seventh_are_its_9th_sharp_11th_and_13th(self):
        listener = Listener()
        for pitch in (48, 52, 55, 59):
            listener.hear(0.0, 1.0, pitch)
        self.assertEqual(listener.tensions(0.3), {2, 6, 9})
        self.assertEqual(Listener().tensions(0.3), set())

    def test_it_finds_your_pulse_and_hears_where_your_phrase_ends(self):
        listener = Listener()
        end = play(listener, C_MAJOR, start=0.1, step=0.3)
        tatum, phase = listener.pulse(end)
        self.assertAlmostEqual(tatum, 0.3)
        self.assertAlmostEqual(phase, 0.1, delta=0.02)
        self.assertEqual(listener.phrase(end - 0.2), [])
        self.assertEqual(listener.phrase(end + 1.0), C_MAJOR)

    def test_a_forced_scale_takes_the_place_of_your_key_and_its_tensions_too(self):
        listener = Listener()
        end = play(listener, C_MAJOR)
        listener.force(5, 'minor_pentatonic')
        self.assertEqual(listener.forced_name(), 'F minor pentatonic')
        self.assertEqual(list(listener.in_key().nonzero()[0]), [0, 3, 5, 8, 10])   # F Ab Bb C Eb
        bias = listener.bias(end, 1.0, 1.0, 0.0)
        self.assertLessEqual(bias[64], -4.0)   # E: in your key, outside the forced scale
        self.assertGreater(bias[60], bias[62])
        listener.force(0, None)
        self.assertIsNone(listener.forced_name())
        self.assertEqual(listener.name(listener.key()[0]), 'C major')

    def test_a_tempo_you_fix_takes_the_place_of_your_pulse(self):
        listener = Listener()
        end = play(listener, C_MAJOR, start=0.1, step=0.3)
        listener.fix(100)
        self.assertEqual(listener.pulse(end), (0.3, 0.0))
        self.assertEqual(listener.beat(0.3), 0.6)
        listener.fix(0)
        self.assertAlmostEqual(listener.pulse(end)[0], 0.3)
        self.assertAlmostEqual(listener.beat(0.3), 0.6)


class ChartTest(unittest.TestCase):
    def over(self, root, kind, key):
        listener = Listener()
        listener.chart = (lambda now: (root, kind), key)
        return listener, listener.bias(0.0, 1.0, 1.0, 2.0)

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


class AvoidTest(unittest.TestCase):
    def test_over_your_c_minor_the_major_third_clashes_whatever_the_key(self):
        listener = Listener()
        for i in range(30):                          # a long while in C major
            listener.hear(i * 0.2, 0.2, (60, 62, 64, 65, 67, 69, 71)[i % 7])
        for pitch in (48, 51, 55):                   # then your C minor chord, held
            listener.hear(7.0, 4.0, pitch)
        bias = listener.bias(8.0, 1.0, 1.0, 0.5)
        self.assertLess(bias[64], bias[63] - 3)      # E against Eb
        self.assertGreater(bias[62], bias[64])       # its ninth is no avoid note


if __name__ == '__main__':
    unittest.main()
