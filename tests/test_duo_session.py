"""Needs torch and anticipation: run it where the model runs (python -m unittest tests.test_duo_session)."""
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anticipation.config import CONTEXT_SIZE, MAX_PITCH
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET, VOCAB_SIZE
from music_transformer import duo
from music_transformer.duo import DuoSession


class Cache:
    def __init__(self):
        self.length = 0

    def crop(self, length):
        self.length = length


class UniformModel:
    """Every token equally likely; checks the cache stays in step with the tokens fed."""

    def __call__(self, inp, past_key_values=None, use_cache=True):
        past = past_key_values or Cache()
        past.length += inp.shape[1]
        self.fed = past.length
        return SimpleNamespace(logits=torch.zeros(1, inp.shape[1], VOCAB_SIZE), past_key_values=past)


class StuckModel(UniformModel):
    """Always wants the earliest time and pitch 61: the loop AMT falls into once a doubled note is in its context."""

    def __call__(self, inp, past_key_values=None, use_cache=True):
        out = super().__call__(inp, past_key_values, use_cache)
        out.logits[0, -1, TIME_OFFSET:TIME_OFFSET + 10000] = -torch.arange(10000).float()
        out.logits[0, -1, NOTE_OFFSET + 4 * MAX_PITCH + 61] = 50
        return out


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class DuoSessionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.clock, self.model = Clock(), UniformModel()
        self.session = DuoSession(self.model, 'cpu', clock=self.clock)

    def fill(self):
        while self.session.ahead() < duo.LOOKAHEAD:
            yield self.session.play()

    def test_the_model_plays_its_own_instrument_after_now_without_long_rests(self):
        self.clock.t = 3.0
        self.session.hear(3.0, 0.3, 60)
        notes = list(self.fill())
        self.assertGreater(len(notes), 0)
        onsets = [onset for onset, *_ in notes]
        self.assertEqual(onsets, sorted(onsets))
        self.assertGreaterEqual(onsets[0], 3.0 + duo.MARGIN / 100)
        self.assertLessEqual(max(b - a for a, b in zip([3.0] + onsets, onsets)), duo.MAX_REST / 100 + 1e-9)
        instruments = {(n - NOTE_OFFSET) // MAX_PITCH for n in self.session.seq[3::3][1:]}
        self.assertLessEqual(instruments, {self.session.me, self.session.left})
        self.assertLessEqual({hand for *_, hand in notes}, {'left', 'right'})

    def test_your_note_drops_only_what_the_model_has_not_played_yet(self):
        self.session.hear(0.0, 0.3, 60)
        notes = list(self.fill())
        self.clock.t = 0.5
        cut = self.session.hear(0.5, 0.3, 62)
        kept = [n for n in notes if n[0] <= 0.5 + duo.MARGIN / 100]
        dropped = [n for n in notes if n[0] > 0.5 + duo.MARGIN / 100]
        self.assertEqual(cut, dropped[0][0] if dropped else None)
        self.assertEqual(len(self.session.seq), 1 + 3 * (1 + len(kept) + 1))
        events = self.session._events()
        self.assertEqual([e[0] for e in events], sorted(e[0] for e in events))
        self.assertIn([TIME_OFFSET + 50, DUR_OFFSET + 30, NOTE_OFFSET + 62], events)   # when you played it
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_you_are_heard_when_you_played_even_before_its_notes_already_on_their_way(self):
        self.session.hear(0.0, 0.3, 60)
        notes = list(self.fill())
        coming = next(n for n in notes if n[0] > 0.5)
        self.clock.t = coming[0] - 0.05   # it sounds within MARGIN: it stays
        self.session.hear(coming[0] - 0.08, 0.3, 62)   # your key reached it a little late
        times = [e[0] - TIME_OFFSET for e in self.session._events()]
        yours = [e[2] - NOTE_OFFSET for e in self.session._events()].index(62)
        self.assertEqual(times[yours], round((coming[0] - 0.08) * 100))
        self.assertIn(round(coming[0] * 100), times[yours + 1:])
        self.assertEqual(times, sorted(times))
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_when_you_let_go_it_hears_how_long_your_note_really_was(self):
        self.session.hear(0.0, 0.3, 60)
        list(self.fill())
        self.session.release(0.0, 60, 2.0)
        self.assertEqual(self.session.seq[2], DUR_OFFSET + 200)
        self.assertEqual(self.session.past.length, 2)   # fed again from there
        self.assertEqual(self.session.listener.held, set())
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_a_full_context_is_trimmed_and_restarts_at_time_zero(self):
        self.session.hear(0.0, 0.3, 60)
        longest = 0
        for _ in range(2000):
            self.clock.t += 0.05
            list(self.fill())
            longest = max(longest, len(self.session.seq))
            if self.session.base > 0:
                break
        self.assertGreater(self.session.base, 0)
        self.assertLessEqual(longest, CONTEXT_SIZE)
        self.assertLessEqual(len(self.session.seq), duo.KEEP + 30)
        self.assertEqual(self.session.seq[1], TIME_OFFSET)


    def test_the_model_stays_a_soloist_even_when_it_wants_to_loop(self):
        for hands in (1, 2):
            session = DuoSession(StuckModel(), 'cpu', clock=self.clock, hands=hands)
            session.hear(0.0, 0.3, 60)
            notes = [session.play() for _ in range(40)]
            for hand, most, step in (('right', duo.MAX_CHORD, duo.MIN_STEP), ('left', duo.LEFT_CHORD, duo.LEFT_STEP)):
                onsets = {}
                for onset, _, pitch, played in notes:
                    if played == hand:
                        onsets.setdefault(onset, []).append(pitch)
                for pitches in onsets.values():
                    self.assertEqual(len(pitches), len(set(pitches)))
                    self.assertLessEqual(len(pitches), most)
                times = sorted(onsets)
                if len(times) > 1:
                    self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), step / 100 - 1e-9)
                pitches = [pitch for _, _, pitch, played in notes if played == hand]
                runs = [tuple(pitches[i:i + duo.NO_REPEAT]) for i in range(len(pitches) - duo.NO_REPEAT + 1)]
                self.assertEqual(len(runs), len(set(runs)))

    def test_a_knob_turned_while_it_plays_changes_its_next_notes(self):
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock)
        session.hear(0.0, 0.3, 60)
        for _ in range(10):
            session.play()
        session.turn(min_step=50, top_p=2.0)
        self.assertEqual(session.knobs()['top_p'], 1.0)   # kept within its range
        times = sorted({onset for onset, *_, hand in (session.play() for _ in range(20)) if hand == 'right'})
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.5 - 1e-9)

    def test_it_plays_only_in_the_octaves_it_is_given_even_turned_the_wrong_way_round(self):
        self.session.turn(lowest=5, highest=4)
        self.session.hear(0.0, 0.3, 30)
        pitches = [self.session.play()[2] for _ in range(60)]
        self.assertTrue(all(60 <= p <= 83 for p in pitches), pitches)
        self.assertEqual(self.session.octaves(), (60, 83))

    def test_it_says_what_it_does_and_which_knobs_are_in_effect_then(self):
        self.assertEqual(self.session.state()['mode'], 'listening')
        for i, pitch in enumerate((60, 64, 67, 72)):
            self.clock.t = i * 0.25
            self.session.hear(self.clock.t, 0.2, pitch)
        state = self.session.state()
        self.assertEqual(state['mode'], 'accompanying')
        self.assertIn('under', state['active'])
        self.assertNotIn('echo', state['active'])
        self.clock.t = 2.0
        state = self.session.state()
        self.assertEqual(state['mode'], 'answering')
        self.assertIn('echo', state['active'])
        self.assertNotIn('under', state['active'])
        self.clock.t = 10.0
        self.assertEqual(self.session.state()['mode'], 'alone')
        self.assertEqual(self.session.listener.chord_name({7, 11, 2, 5}), 'G7')

    def test_knobs_given_at_the_start_hold_from_the_first_note(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, memory=60.4, stay=0.9)
        self.assertEqual(session.memory, 60)
        self.assertEqual(session.knobs()['stay'], 0.9)

    def test_the_model_remembers_all_of_you_but_only_the_last_moments_of_itself(self):
        self.session.hear(0.0, 0.3, 60)
        while self.clock.t < 5:
            self.clock.t += 0.05
            list(self.fill())
        self.session.hear(self.clock.t, 0.3, 62)
        mine = [e[0] - TIME_OFFSET for e in self.session._events() if self.session._instrument(e[2]) == self.session.me]
        yours = [e[2] - NOTE_OFFSET for e in self.session._events() if self.session._instrument(e[2]) == self.session.you]
        self.assertEqual(yours, [60, 62])
        self.assertEqual(self.session.seq[1], TIME_OFFSET)
        self.assertGreaterEqual(min(mine), self.session.heard_at - duo.MEMORY)
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_the_model_prefers_the_pitch_classes_you_are_playing(self):
        for i, pitch in enumerate((60, 64, 67, 72)):   # C major triad
            self.session.hear(i * 0.2, 0.2, pitch)
        self.clock.t = 0.6
        notes = []
        for _ in range(60):
            notes.append(self.session.play())
            self.clock.t = notes[-1][0]
            self.session.hear(self.clock.t, 0.2, (60, 64, 67)[len(notes) % 3])
        in_key = sum(pitch % 12 in (0, 4, 7) for _, _, pitch, _ in notes) / len(notes)
        self.assertGreater(in_key, 0.35)   # 3 pitch classes of 12: 25% without ears

    def play_scale(self, pitches, start):
        for i, pitch in enumerate(pitches):
            self.clock.t = start + i * 0.3
            self.session.hear(self.clock.t, 0.2, pitch)
        self.clock.t += 1.5   # you rest: the model answers and leads

    def test_after_you_modulate_it_plays_in_the_new_key(self):
        c_major = [60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60, 62, 65, 69, 71, 67]
        self.play_scale(c_major, 0.1)
        self.play_scale([p + 3 for p in c_major], self.clock.t)
        pitches = [pitch % 12 for _ in range(4) for _, _, pitch, _ in self.fill()] + \
                  [self.session.play()[2] % 12 for _ in range(60)]
        self.assertGreater(sum(p in (3, 5, 7, 8, 10, 0, 2) for p in pitches) / len(pitches), 0.75)   # 58% by chance
        self.assertLess(sum(p in (4, 11) for p in pitches) / len(pitches), 0.1)   # only in C: A stays possible, as in Bb

    def test_its_onsets_fall_on_your_pulse(self):
        self.play_scale([60, 62, 64, 65, 67, 69, 71, 72], 0.1)
        onsets = [self.session.play()[0] for _ in range(30)]
        for onset in onsets:
            off = (onset - 0.1) % 0.3
            self.assertLessEqual(min(off, 0.3 - off), duo.GRID / 100 + 1e-9)

    def follow(self, count):
        """The model's next `count` notes, the clock following them as it would while they sound."""
        notes = []
        for _ in range(count):
            notes.append(self.session.play())
            self.clock.t = notes[-1][0]
        return notes

    def test_with_two_hands_each_keeps_to_its_register_and_with_one_it_plays_a_single_line(self):
        self.session.hear(0.0, 0.3, 72)
        notes = self.follow(80)
        left = [pitch for _, _, pitch, hand in notes if hand == 'left']
        right = [pitch for _, _, pitch, hand in notes if hand == 'right']
        self.assertTrue(left and right)
        self.assertLessEqual(max(left), duo.LEFT_TOP)
        self.assertGreaterEqual(min(right), duo.RIGHT_BOTTOM)
        self.session.turn(hands=1)
        self.assertEqual({hand for *_, hand in self.follow(20)}, {'right'})

    def test_octaves_with_room_for_one_hand_leave_it_to_that_hand(self):
        self.session.turn(lowest=1, highest=2)   # C1 to B2: below the right hand's F3
        self.session.hear(0.0, 0.3, 40)
        notes = [self.session.play() for _ in range(40)]
        self.assertTrue(all(24 <= pitch <= 47 for _, _, pitch, _ in notes))
        self.assertEqual({hand for *_, hand in notes}, {'left'})

    def test_it_may_leave_longer_silences_when_you_let_it(self):
        self.session.turn(max_rest=600, hands=1)
        self.clock.t = 3.0
        self.session.hear(3.0, 0.3, 60)
        notes = list(self.fill())
        onsets = [onset for onset, *_ in notes]
        self.assertLessEqual(max(b - a for a, b in zip([3.0] + onsets, onsets)), 6.0 + 1e-9)

    def test_it_colours_your_chord_with_its_tensions_even_outside_your_key(self):
        for pitch in (48, 52, 55, 59):   # Cmaj7, C in the bass
            self.session.hear(0.0, 2.0, pitch)
        bias = self.session._partner(30, [])
        d, f_sharp, c_sharp = 62, 66, 61   # its 9th, its #11 (not in C major), and a wrong note
        self.assertGreater(bias[d], bias[c_sharp])
        self.assertGreater(bias[f_sharp], bias[c_sharp])
        self.session.turn(tension=0)
        self.assertAlmostEqual(float(self.session._partner(30, [])[d]), float(bias[d]) - duo.TENSION, places=5)

if __name__ == '__main__':
    unittest.main()
