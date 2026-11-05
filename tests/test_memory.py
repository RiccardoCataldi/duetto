import os
import sys
import tempfile
import unittest

import mido

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_transformer.memory import GAP, Memory

LINE = [60, 64, 67, 72, 71, 69, 65, 62]


def play(memory, pitches, start=0.0, step=0.25):
    for i, pitch in enumerate(pitches):
        memory.hear(start + i * step, pitch, 0.2)
    memory.update(start + len(pitches) * step)
    return start + len(pitches) * step


class MemoryTest(unittest.TestCase):
    def test_a_line_you_play_again_wakes_what_came_after_it_and_it_says_what_comes_next(self):
        memory = Memory(harmonic=0, least=0.5)
        end = play(memory, LINE + [55])
        end = play(memory, LINE[:6], end + 1.0)
        slice_, shift, matched = memory.step(end)
        self.assertTrue(matched)
        self.assertEqual((memory.slices[slice_][0][0], shift), (LINE[6], 0))

    def test_it_knows_you_in_another_key(self):
        memory = Memory(harmonic=0, least=0.5)
        end = play(memory, LINE + [55])
        end = play(memory, [p + 2 for p in LINE[:6]], end + 1.0)
        slice_, shift, _ = memory.step(end)
        self.assertEqual(shift, 2)
        self.assertEqual(memory.notes(slice_, shift)[0][0], LINE[6] + 2)

    def test_several_notes_that_agree_grow_one_peak(self):
        memory = Memory(harmonic=0)
        end = play(memory, LINE + [55])
        play(memory, LINE[:5], end + 1.0)
        one = memory.scores.max()
        memory.hear(end + 1.0 + 5 * 0.25, LINE[5], 0.2)
        memory.update(end + 1.0 + 6 * 0.25)
        self.assertGreater(memory.scores.max(), one)

    def test_a_peak_fades_and_dies(self):
        memory = Memory(harmonic=0, least=0.5, decay=4.0)
        end = play(memory, LINE * 3 + [55])
        play(memory, LINE[:6], end + 1.0)
        self.assertGreater(len(memory.scores), 0)
        memory.update(end + 1.0 + 40.0)
        self.assertEqual(len(memory.scores), 0)

    def test_the_rests_are_in_the_corpus(self):
        memory = Memory()
        memory.hear(0.0, 60, 0.2)
        memory.hear(1.5, 62, 0.2)
        memory.hear(1.75, 64, 0.2)
        memory.update(2.0)
        self.assertAlmostEqual(memory.duration(0), 1.5)
        self.assertAlmostEqual(memory.duration(1), 0.25)

    def test_notes_together_are_one_slice(self):
        memory = Memory()
        for pitch in (60, 64, 67):
            memory.hear(1.0 + pitch / 10000, pitch, 0.2)
        memory.update(1.1)
        self.assertEqual(memory.size, 1)
        self.assertEqual([p for p, _ in memory.notes(0, 2)], [62, 66, 69])

    def test_it_must_jump_after_going_long_enough_along_one_line(self):
        memory = Memory(harmonic=0, least=0.5, jump=2)
        end = play(memory, LINE * 2 + [55])
        end = play(memory, LINE[:6], end + 1.0)
        at, played = end, []
        for _ in range(5):
            event = memory.step(at)
            played.append(event[0])
            at += 0.25
        self.assertTrue(any(b != a + 1 for a, b in zip(played, played[1:])))

    def test_with_no_peak_it_carries_on_along_what_it_was_remembering(self):
        memory = Memory(harmonic=0, least=0.5)
        end = play(memory, LINE + [55])
        end = play(memory, LINE[:6], end + 1.0)
        slice_ = memory.step(end)[0]
        memory._keep([])
        self.assertEqual(memory.step(end + 0.25)[:3], (slice_ + 1, 0, False))

    def test_its_own_line_wakes_its_memory_too(self):
        memory = Memory(harmonic=0, itself=1.0, melodic=0)
        end = play(memory, LINE + [55])
        memory.update(end + 1.0)
        self.assertEqual(len(memory.scores), 0)
        for i, pitch in enumerate(LINE[:6]):
            memory.own(end + 1.0 + i * 0.25, pitch)
        self.assertGreater(len(memory.scores), 0)

    def test_your_harmony_tells_the_chord_that_comes_next(self):
        c, d, g = (60, 64, 67), (62, 65, 69), (55, 59, 62, 65)
        memory = Memory(melodic=0, least=0.5)
        time_ = 0.0
        for chord in (c, d, g, c, d, g, c, d, g, c, d, g, c, d):
            for pitch in chord:
                memory.hear(time_ + pitch / 10000, pitch, 0.5)
            time_ += 1.0
        memory.update(time_ - 0.2)
        expected, strength = memory.expect(time_ + 0.2)
        self.assertGreater(strength, 0)
        self.assertEqual(set(expected.argsort()[-4:].tolist()), {7, 11, 2, 5})

    def test_it_remembers_the_first_track_of_saved_duos_before_you_start(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'duo.mid')
            mid = mido.MidiFile()
            for pitches in (LINE, [30]):
                track = mido.MidiTrack()
                for pitch in pitches:
                    track.append(mido.Message('note_on', note=pitch, velocity=80, time=0))
                    track.append(mido.Message('note_off', note=pitch, velocity=0, time=240))
                mid.tracks.append(track)
            mid.save(path)
            memory = Memory(harmonic=0, least=0.5)
            memory.load([path])
        self.assertEqual(memory.tops, LINE)
        self.assertLess(memory.times[memory.size - 1], -GAP)
        end = play(memory, LINE[:6])
        self.assertEqual(memory.slices[memory.step(end)[0]][0][0], LINE[6])
