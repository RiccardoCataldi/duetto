import os
import sys
import unittest

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


if __name__ == '__main__':
    unittest.main()
