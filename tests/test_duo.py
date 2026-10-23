import os
import random
import sys
import tempfile
import unittest

import mido

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


class Out:
    def __init__(self):
        self.messages = []

    def play_midi_message(self, message, is_generated=False):
        self.messages.append((list(message), is_generated))

    def generated_on(self):
        return [m[1] for m, generated in self.messages if generated and m[0] == 0x90]


class DuoTest(unittest.TestCase):
    def setUp(self):
        self.sent, self.out = [], Out()
        self.duo = main.Duo(self.sent.append, self.out, started=100.0)

    def test_your_key_goes_to_the_model_at_once_with_a_guessed_duration(self):
        self.duo.note_on(60, 90, 101.5)
        self.assertEqual(self.sent, [f"on 1.500 {main.GUESS_S:.3f} 60 90\n"])

    def test_the_guess_follows_how_long_you_hold_your_notes(self):
        for i, pitch in enumerate((60, 62, 64)):
            self.duo.note_on(pitch, 90, 101 + i)
            self.duo.note_off(pitch, 101 + i + 0.1)
        self.duo.note_on(65, 90, 105)
        self.assertEqual(self.sent[-1], "on 5.000 0.100 65 90\n")

    def test_when_your_note_stops_sounding_the_model_hears_how_long_it_lasted_pedal_and_all(self):
        self.duo.note_on(60, 90, 101.0)
        self.duo.note_off(60, 101.4)
        self.assertEqual(self.sent[-1], "off 1.000 60 0.400\n")
        self.duo.sustain(127, 102.0)
        self.duo.note_on(62, 90, 102.5)
        self.duo.note_off(62, 102.6)
        self.assertTrue(self.sent[-1].startswith("on 2.500"))   # the pedal holds it
        self.duo.sustain(0, 104.0)
        self.assertEqual(self.sent[-1], "off 2.500 62 1.500\n")

    def test_the_model_plays_its_notes_when_they_are_due_and_lets_them_ring(self):
        self.duo.receive("note 2.000 0.500 67\n")
        self.duo.tick(101.9)
        self.assertEqual(self.out.generated_on(), [])
        self.duo.tick(102.0)
        self.assertEqual(self.out.generated_on(), [67])
        self.duo.tick(102.6)
        self.assertNotIn(([0x80, 67, 0], True), self.out.messages)
        self.duo.tick(102.0 + 0.5 + main.RING_S)
        self.assertIn(([0x80, 67, 0], True), self.out.messages)

    def test_a_cut_drops_the_notes_that_have_not_sounded(self):
        for line in ("note 2.000 0.200 60\n", "note 2.500 0.200 62\n", "note 3.000 0.200 64\n"):
            self.duo.receive(line)
        self.duo.receive("cut 2.500\n")
        for now in (102.0, 102.5, 103.0):
            self.duo.tick(now)
        self.assertEqual(self.out.generated_on(), [60])

    def generated_velocities(self):
        return {m[1]: m[2] for m, generated in self.out.messages if generated and m[0] == 0x90}

    def test_the_pedal_holds_your_notes_until_it_comes_up_and_is_saved(self):
        self.duo.sustain(127, 100.5)
        self.duo.note_on(60, 90, 101.0)
        self.duo.note_off(60, 101.2)
        self.assertEqual(self.duo.notes, [])
        self.duo.sustain(0, 102.0)
        self.assertEqual(self.duo.notes, [(101.0, 102.0, 60, 90, False)])
        self.duo.note_on(62, 90, 103.0)
        self.duo.note_off(62, 103.1)
        self.assertEqual(self.duo.notes[-1][:2], (103.0, 103.1))
        path = os.path.join(tempfile.mkdtemp(), 'duo.mid')
        self.duo.save(path)
        pedal = [m.value for m in mido.MidiFile(path) if m.type == 'control_change' and m.control == main.PEDAL]
        self.assertEqual(pedal, [0])   # the pedal down came before your first note, where the duo starts

    def test_its_top_voice_sings_over_its_inner_voices_and_its_left_hand(self):
        self.duo = main.Duo(self.sent.append, self.out, started=100.0, rng=random.Random(0))
        main_human, main.HUMAN = main.HUMAN, 0
        try:
            for line in ("note 2.000 0.500 60 right\n", "note 2.000 0.500 67 right\n", "note 2.000 0.500 43 left\n"):
                self.duo.receive(line)
            self.duo.tick(102.0)
        finally:
            main.HUMAN = main_human
        velocity = self.generated_velocities()
        self.assertGreater(velocity[67], velocity[60])
        self.assertGreater(velocity[67], velocity[43])

    def test_it_plays_louder_when_you_do(self):
        levels = []
        for loud in (40, 120):
            out = Out()
            duo = main.Duo(self.sent.append, out, started=100.0, rng=random.Random(0))
            for i in range(6):
                duo.note_on(60, loud, 100.0 + i * 0.1)
            duo.receive("note 5.000 0.500 64\n")
            duo.tick(105.0)
            levels.append([m[2] for m, generated in out.messages if generated][0])
        self.assertGreater(levels[1], levels[0] + 30)

    def test_it_strikes_its_keys_as_hard_as_the_model_says_when_it_says(self):
        self.duo.receive("note 2.000 0.500 60 right 110\n")
        self.duo.tick(102.0)
        self.assertEqual(self.generated_velocities(), {60: 110})

    def test_its_line_swells_as_it_rises(self):
        line = [60, 62, 64, 65, 67, 69, 71, 72]
        low = main.dynamics(60, 'right', [60], line, 80, False, random.Random(0))
        high = main.dynamics(72, 'right', [72], line, 80, False, random.Random(0))
        self.assertGreater(high, low)

    def test_a_note_that_arrives_too_late_is_skipped(self):
        self.duo.receive("note 2.000 0.200 60\n")
        self.duo.tick(102.0 + main.LATE_S + 0.05)
        self.assertEqual(self.out.generated_on(), [])
        self.assertEqual(self.duo.late, 1)


if __name__ == '__main__':
    unittest.main()
