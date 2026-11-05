"""Needs torch and anticipation: run it where the model runs (python -m unittest tests.test_jam)."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anticipation.config import MAX_PITCH
from anticipation.vocab import ANOTE_OFFSET, ANTICIPATE, ATIME_OFFSET, AUTOREGRESS, NOTE_OFFSET
from music_transformer import jam
from music_transformer.duo import LOOKAHEAD
from music_transformer.jam import JamSession
from tests.test_duo_session import Clock, UniformModel


class JamSessionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.clock = Clock()

    def session(self, strategy, **knobs):
        return JamSession(UniformModel(), 'cpu', clock=self.clock, strategy=jam.STRATEGIES.index(strategy), **knobs)

    def play_until(self, session, seconds):
        """Samples as the server does, its clock moving on, up to `seconds`: its notes."""
        notes = []
        while self.clock.t < seconds:
            if session.ahead() < LOOKAHEAD:
                note = session.play()
                if note:
                    notes.append(note)
                    continue
            self.clock.t += 0.05
        return notes

    def test_trading_it_keeps_silent_in_your_turn_and_plays_only_in_its_own(self):
        session = self.session('trade', bars=1, bpm=120)   # bars of 2 s on the metronome's
        for i in range(4):
            self.clock.t = i * 0.5
            session.hear(i * 0.5, 0.4, 60 + i)
        notes = self.play_until(session, 4.0)
        self.assertTrue(notes)
        self.assertTrue(all(2.0 <= onset < 4.0 for onset, *_ in notes))
        self.assertEqual(session.state()['mode'], 'your turn')

    def test_trading_it_stops_when_you_did_not_play_in_your_turn(self):
        session = self.session('trade', bars=1, bpm=120)
        session.hear(0.0, 0.4, 60)        # in your first turn, not in your second (4 to 6 s)
        self.play_until(session, 8.5)
        self.assertIsNone(session.turns)
        self.assertEqual(session.state()['mode'], 'listening')

    def test_trading_its_prompt_is_you_both_as_one_piano(self):
        session = self.session('trade', bars=1, bpm=120)
        session.hear(0.0, 0.4, 60)
        self.play_until(session, 2.5)
        notes = session.seq[3::3]
        self.assertTrue(all((n - NOTE_OFFSET) // MAX_PITCH == session.you for n in notes))

    def test_after_your_gesture_it_drops_what_it_has_not_played_and_starts_again_from_it(self):
        session = self.session('gesture', wait=20)
        session.hear(0.0, 0.4, 60)
        self.play_until(session, 1.0)
        self.clock.t = 1.0
        session.hear(1.0, 0.4, 67)
        self.assertGreater(session.ahead(), LOOKAHEAD)   # it waits for the rest of your gesture
        self.clock.t = 1.2
        self.assertEqual(session.ahead(), 0)
        session.play()
        self.assertAlmostEqual(session.recut(), 1.35)
        self.assertIsNone(session.recut())
        self.assertFalse([n for n in session.notes if n[3] and n[0] >= 1.35 and n[0] != session.last])
        self.assertIn(NOTE_OFFSET + session.you * MAX_PITCH + 67, session.seq)

    def test_its_prompt_is_only_the_last_notes(self):
        session = self.session('gesture', context=10)
        for i in range(30):
            self.clock.t = i * 0.2
            session.hear(i * 0.2, 0.1, 60 + i % 12)
        self.clock.t = 6.2
        session.play()
        self.assertLessEqual(len(session.seq), 1 + 3 * 11)

    def test_it_plays_on_from_its_own_last_notes_keeping_its_context_short(self):
        session = self.session('gesture', context=10, timeout=6000)
        session.hear(0.0, 0.4, 60)
        self.play_until(session, 20.0)
        self.assertLessEqual(len(session.seq), 1 + 3 * 21)

    def test_locked_your_gesture_is_held_ahead_of_it_as_anticipated_controls(self):
        session = self.session('gesture', lock=1, every=50, wait=0)
        for pitch in (60, 64, 67):
            session.hear(0.0, 1.0, pitch)
        session.play()
        self.assertEqual(session.seq[0], ANTICIPATE)
        controls = [t for t in session.seq[1:] if t >= ATIME_OFFSET]
        held = {(n - ANOTE_OFFSET) % MAX_PITCH for n in controls[2::3]}
        self.assertEqual(held, {60, 64, 67})
        times = sorted({t - ATIME_OFFSET for t in controls[0::3]})
        self.assertEqual(times[1] - times[0], 50)

    def test_unlocked_it_only_continues(self):
        session = self.session('gesture', wait=0)
        session.hear(0.0, 1.0, 60)
        session.play()
        self.assertEqual(session.seq[0], AUTOREGRESS)
        self.assertFalse([t for t in session.seq[1:] if t >= ATIME_OFFSET])

    def test_on_request_only_a_low_key_cues_it_and_it_starts_from_your_notes_only(self):
        session = self.session('request', trigger=36)
        session.hear(0.0, 0.4, 60)
        self.assertIsNone(session.play())
        self.assertGreater(session.ahead(), LOOKAHEAD)
        self.play_until(session, 1.0)
        session.hear(1.0, 0.4, 31)
        self.clock.t = 1.0
        self.assertIsNotNone(session.play())
        voices = {(n - NOTE_OFFSET) // MAX_PITCH for n in session.seq[3:-3:3]}
        self.assertEqual(voices, {session.you})

    def test_it_stays_silent_once_it_has_played_on_long_enough_after_you(self):
        session = self.session('gesture', timeout=200)
        session.hear(0.0, 0.4, 60)
        self.play_until(session, 3.0)
        self.assertIsNone(session.play())
        self.assertGreater(session.ahead(), LOOKAHEAD)
        self.assertEqual(session.state()['mode'], 'listening')

    def test_it_says_which_knobs_shape_its_notes(self):
        state = self.session('trade').state()
        self.assertIn('bars', state['active'])
        self.assertNotIn('wait', state['active'])
        self.assertNotIn('space', state['active'])


if __name__ == '__main__':
    unittest.main()
