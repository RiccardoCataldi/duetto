"""Needs torch and anticipation: run it where the model runs (python -m unittest tests.test_duo_session)."""
import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anticipation.config import CONTEXT_SIZE, MAX_PITCH
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET, VOCAB_SIZE
from music_transformer import duo
from music_transformer.duo import DuoSession
from music_transformer.listener import hands


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
            note = self.session.play()
            if note:
                yield note

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
        self.session.turn(commit=duo.MARGIN)
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

    def test_it_takes_a_breath_when_its_phrase_has_run_out_of_notes(self):
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock, timeout=duo.NEVER, phrase=4, breath=120, hands=1)
        session.hear(0.0, 0.3, 60)
        onsets = sorted({session.play()[0] for _ in range(30)})
        gaps = [b - a for a, b in zip(onsets, onsets[1:])]
        self.assertGreaterEqual(max(gaps), 1.2)
        self.assertGreater(sum(gap >= 1.2 for gap in gaps), 2)

    def test_it_samples_nothing_while_it_breathes(self):
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock, timeout=duo.NEVER, phrase=2, breath=400, hands=1)
        session.hear(0.0, 0.3, 60)
        while session._breath() == 0:
            session.play()
        self.assertGreaterEqual(session.ahead(), 400 - session.now())

    def test_it_plays_on_after_your_last_note_then_waits_for_you(self):
        self.session.turn(timeout=200, breath=0)
        self.session.hear(0.0, 0.3, 60)
        for _ in range(100):
            self.clock.t += 0.1
            notes = list(self.fill())
            self.assertTrue(all(n[0] <= 2.0 + duo.LOOKAHEAD / 100 + duo.MAX_REST / 100 for n in notes))
        self.assertEqual(self.session.state()['mode'], 'listening')
        self.assertEqual(list(self.fill()), [])
        self.session.hear(self.clock.t, 0.3, 62)
        self.assertGreater(len(list(self.fill())), 0)

    def test_you_can_come_back_after_a_silence_longer_than_the_context_lasts(self):
        self.session.hear(0.0, 0.3, 60)
        list(self.fill())
        self.clock.t = 500.0
        self.session.hear(500.0, 0.3, 62)
        self.assertLess(self.session.seq[-3] - TIME_OFFSET, 10000)
        self.assertGreater(len(list(self.fill())), 0)

    def planned(self, *pitches):
        """A session in C major, with its plan after 0.3 s: `pitches`, 0.1 s apart."""
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, commit=100)
        for i, pitch in enumerate((60, 64, 67, 60, 64, 67, 72, 62, 65, 69, 71) * 3):
            session.listener.hear(-10.0 + i / 10, 0.1, pitch)
        session.hear(0.0, 0.2, 60)
        session.seq += [token for i, pitch in enumerate(pitches) for token in
                        (TIME_OFFSET + 30 + 10 * i, DUR_OFFSET + 10, NOTE_OFFSET + 4 * MAX_PITCH + pitch)]
        session.spoken = [0.3 + i / 10 for i in range(len(pitches))]
        return session

    def test_its_plan_stands_when_your_note_does_not_contradict_it(self):
        session = self.planned(72, 74, 76)
        self.clock.t = 0.05
        self.assertIsNone(session.hear(0.05, 0.2, 64))
        self.assertEqual(len(session._events()), 5)

    def test_your_note_drops_its_plan_from_the_note_that_contradicts_it(self):
        session = self.planned(72, 73, 76)     # Db in C major
        self.clock.t = 0.05
        self.assertAlmostEqual(session.hear(0.05, 0.2, 64), 0.4)
        self.assertEqual(session.spoken, [0.3])
        self.assertEqual(len(session._events()), 3)

    def test_its_plan_does_not_stand_past_commit(self):
        session = self.planned(72, 74, 76)
        session.turn(commit=40)
        self.clock.t = 0.05
        self.assertAlmostEqual(session.hear(0.05, 0.2, 64), 0.5)

    def remembered(self, **knobs):
        """A session that has heard a line, a rest, and the line again up to where it should carry on."""
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, harmonic=0, least=0.5, hands=1,
                             recall=duo.EXACT, **{'silence': 0, **knobs})
        for i, pitch in enumerate((60, 64, 67, 72, 71, 69, 65, 62)):
            session.hear(i * 0.5, 0.2, pitch)
        session.hear(8.0, 0.2, 60)   # a rest of a second, and the rest of the line after it
        for i, pitch in enumerate((64, 67, 72)):
            session.hear(8.5 + i * 0.5, 0.2, pitch)
        for i, pitch in enumerate((60, 64, 67, 72)):
            session.hear(14.0 + i * 0.5, 0.2, pitch)
        self.clock.t = 16.0
        session.mind.update(16.0)
        return session

    def test_at_full_recall_it_plays_what_it_remembers_with_its_timing(self):
        session = self.remembered()
        notes = [session.play() for _ in range(2)]
        self.assertEqual([n[2] for n in notes], [71, 69])
        self.assertAlmostEqual(notes[1][0] - notes[0][0], 0.5, delta=0.02)

    def test_a_rest_it_remembers_it_keeps_silent(self):
        session = self.remembered()
        for pitch in (71, 69, 65, 62):
            self.assertEqual(session.play()[2], pitch)
        gap = session.play()
        self.assertGreaterEqual(gap[0] - round(session.spoken[-2], 2), 1.0)

    def test_it_samples_nothing_while_a_rest_it_remembers_passes(self):
        session = self.remembered(silence=0.9)
        with unittest.mock.patch('music_transformer.duo.random.random', return_value=0.0):
            self.assertIsNone(session.play())
        self.assertGreater(session.cursor, 16.0)

    def test_sparse_it_keeps_silent_until_it_remembers(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, sparse=1, hands=1)
        session.hear(0.0, 0.2, 60)
        self.assertIsNone(session.play())
        self.assertGreaterEqual(session.ahead(), duo.LOOKAHEAD)

    def test_with_little_recall_it_mostly_invents(self):
        session = self.remembered()
        session.turn(recall=1.0)
        with unittest.mock.patch('music_transformer.duo.random.random', return_value=0.5):
            session.play()
        self.assertTrue(session.free)
        with unittest.mock.patch('music_transformer.duo.random.random', return_value=0.05):
            session.plan = []
            session.play()
        self.assertFalse(session.free)

    def test_a_rest_it_remembers_lasts_no_longer_than_max_rest(self):
        session = self.remembered(max_rest=50)
        for pitch in (71, 69, 65, 62):
            self.assertEqual(session.play()[2], pitch)
        gap = session.play()
        self.assertLess(gap[0] - round(session.spoken[-2], 2), 0.6)

    def test_without_recall_it_only_invents(self):
        session = self.remembered()
        session.turn(recall=0)
        session.plan = []
        session.play()
        self.assertTrue(session.free)

    def test_with_no_timeout_it_answers_each_of_your_notes_and_waits(self):
        self.session.turn(timeout=0, breath=0)
        self.session.hear(0.0, 0.3, 60)
        self.assertLess(self.session.ahead(), duo.LOOKAHEAD)
        self.session.play()
        self.assertGreaterEqual(self.session.ahead(), duo.LOOKAHEAD)

    def test_over_a_chart_it_plays_the_chord_of_the_bar_at_the_charts_tempo(self):
        from music_transformer.chart import CHARTS
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'autumn_leaves')
        self.session.turn(chart=tune + 1)
        self.assertEqual((self.session.pace, self.session.state()['chord']), (120, 'Cm7 > F7'))
        self.clock.t = 10.2                          # bar 6 of 2 s each: D7, whose F# is not in G minor
        self.session.hear(10.2, 0.3, 62)
        self.assertEqual(self.session.state()['key'], 'Autumn Leaves 6/32')
        bias = self.session._partner(self.session.now() + 30, [], (0, 127), True)
        self.assertGreater(bias[66], bias[65] + 1)

    def test_soloing_over_a_tune_it_approaches_a_chord_tone_by_a_semitone_and_lands_on_it(self):
        from music_transformer.chart import CHARTS
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'blues_in_f')
        self.session.turn(chart=tune + 1, part=2, hands=1, apart=0, lead=0)   # F7 in the first bar
        self.session.hear(0.0, 0.3, 53)
        self.enterContext(unittest.mock.patch('music_transformer.chart.way', return_value='scale'))
        off = self.session._partner(self.session._tick(0.27), [], (0, 127), True)   # the "and" of the first beat
        on = self.session._partner(self.session._tick(0.55), [], (0, 127), True)    # the second beat
        self.assertGreater(off[68], on[68] + 5)      # Ab4, a semitone under A: only off the beat
        self.assertGreater(on[69], off[69])          # on the beat the chord's tones are its targets
        line = [(27, NOTE_OFFSET + 4 * MAX_PITCH + 68)]
        landed = self.session._partner(self.session._tick(0.55), line, (0, 127), True)
        self.assertEqual(int(landed.argmax()), 69)   # after Ab, A
        comps = self.session._partner(self.session._tick(0.27), [], (0, 127), False)   # its left hand does not
        self.assertLess(comps[68], -5)

    def test_soloing_over_a_tune_it_runs_in_triplets_or_sixteenths_for_a_phrase(self):
        from music_transformer.chart import CHARTS
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'autumn_leaves')   # 120 bpm: a beat is 0.5 s
        for division, step in ((3, 1 / 6), (4, 0.125)):
            session = DuoSession(UniformModel(), 'cpu', clock=Clock(), chart=tune + 1, part=2, hands=1, breath=0)
            session.hear(0.0, 0.3, 60)
            with unittest.mock.patch('music_transformer.chart.feel', return_value=division):
                onsets = sorted({session.play()[0] for _ in range(30)})
            self.assertTrue(all(min(o % step, step - o % step) <= 0.035 for o in onsets), onsets)   # on its grid
            gaps = [b - a for a, b in zip(onsets, onsets[1:])]
            self.assertLessEqual(max(gaps), 2 * step + 0.05)       # a run: no more than two of them of silence
            self.assertLess(min(gaps), step + 0.03)                # as fast as the division

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
        self.session.turn(timeout=duo.NEVER)
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
            for hand, most, step in (('right', duo.CHORD, duo.MIN_STEP), ('left', duo.LEFT_CHORD, duo.LEFT_STEP)):
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

    def test_it_comps_in_chords_while_you_play_and_plays_lines_when_you_rest(self):
        for you_play, most in ((True, 4), (False, 1)):
            session = DuoSession(StuckModel(), 'cpu', clock=self.clock, hands=1, chord=4, solo_chord=1, no_repeat=8)
            session.hear(0.0, 0.3, 60)
            self.clock.t = 0.0 if you_play else 3.0
            onsets = {}
            for onset, _, pitch, _ in (session.play() for _ in range(12)):
                onsets.setdefault(onset, []).append(pitch)
            self.assertEqual(max(map(len, onsets.values())), most)
        self.clock.t = 0.0

    def test_your_hands_are_where_you_play_an_octave_apart(self):
        self.assertEqual(hands([52, 55, 59, 62, 76, 79, 83]), [(52, 62), (76, 83)])
        self.assertEqual(hands([60, 71]), [(60, 71)])   # a seventh apart: one hand, the keys between too
        self.assertEqual(hands([60, 72]), [(60, 60), (72, 72)])   # an octave: room between
        self.assertEqual(hands([]), [])

    def while_you_play(self, held, n=40):
        """The first note of `n` duos in which you hold `held`: each sounds while you are still playing."""
        pitches = []
        for _ in range(n):
            session = DuoSession(UniformModel(), 'cpu', clock=Clock(), apart=6.0, hands=1, lowest=2, highest=7,
                                 max_rest=50)
            for pitch in held:
                session.hear(0.0, 30.0, pitch)
            pitches.append(session.play()[2])
        return pitches

    def test_while_you_play_it_keeps_apart_from_your_hands_and_under_your_melody(self):
        pitches = self.while_you_play((40, 44, 47, 76, 79, 83))   # a left hand and a right hand, held
        inside = [p for p in pitches if 38 <= p <= 49 or p >= 74]
        self.assertLess(len(inside), len(pitches) / 4, pitches)   # 46 of the 96 keys without it
        self.assertTrue([p for p in pitches if 50 <= p <= 73], pitches)

    def test_while_you_only_comp_it_may_take_the_melody_over_you(self):
        pitches = self.while_you_play((48, 52, 55, 59))   # your left hand alone, held
        self.assertLess(len([p for p in pitches if 46 <= p <= 61]), 4, pitches)
        self.assertTrue([p for p in pitches if p >= 72], pitches)

    def test_when_your_hands_leave_it_no_room_it_plays_as_it_would(self):
        runs = []
        for apart in (0.0, 6.0):
            torch.manual_seed(0)
            session = DuoSession(UniformModel(), 'cpu', clock=Clock(), hands=1, lowest=4, highest=5, apart=apart)
            for pitch in (60, 64, 67, 72, 76, 79):   # one hand over all of C4-B5, and your melody on top
                session.hear(0.0, 30.0, pitch)
            runs.append([session.play() for _ in range(20)])
        self.assertEqual(runs[0], runs[1])

    def test_when_you_rest_it_sings_up_around_your_right_hand_and_its_left_hand_stays_down(self):
        for i, pitch in enumerate((43, 47, 50, 74, 76, 77, 79, 81)):
            self.session.hear(i * 0.1, 0.3, pitch)
        self.clock.t = 3.0                          # you rest
        self.session.turn(lead=6.0, answer=0, lowest=3, highest=6)
        notes = [self.session.play() for _ in range(60)]
        right = sorted(pitch for _, _, pitch, hand in notes if hand == 'right')
        # its right hand has F3 to B6: a quarter of its notes under D4 and C5 in the middle without it
        self.assertGreaterEqual(right[len(right) // 2], 80, right)
        self.assertLess(len([p for p in right if p < 66]), len(right) / 10, right)
        self.assertTrue([pitch for _, _, pitch, hand in notes if hand == 'left'])

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
        self.assertIn('apart', state['active'])
        self.assertIn('chord', state['active'])
        self.assertNotIn('echo', state['active'])
        self.clock.t = 2.0
        state = self.session.state()
        self.assertEqual(state['mode'], 'answering')
        self.assertIn('echo', state['active'])
        self.assertNotIn('apart', state['active'])
        self.assertIn('lead', state['active'])
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
        self.assertLessEqual(sum(t < self.session.heard_at - duo.MEMORY for t in mine), duo.PRUNE)
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_the_model_prefers_the_pitch_classes_you_are_playing(self):
        self.session.turn(silence=0)
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
                  [note[2] % 12 for note in (self.session.play() for _ in range(60)) if note]
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

    def test_it_answers_your_phrase_in_a_run_and_sequences_your_intervals_past_its_end(self):
        self.session.turn(run=10, min_step=5, max_rest=600, hands=1, echo=6.0, answer=1000)
        for i, pitch in enumerate((60, 62, 64)):
            self.session.hear(i * 0.25, 0.2, pitch)
        self.clock.t = 1.5   # your phrase has ended: it answers
        notes = []
        while len(notes) < 12:
            notes += self.fill()
            self.clock.t += 0.5
        onsets = [onset for onset, *_ in notes]
        self.assertLessEqual(max(b - a for a, b in zip(onsets, onsets[1:])), 0.1 + 1e-9)
        pitches = [pitch for *_, pitch, _ in notes]
        steps = [b - a for a, b in zip(pitches, pitches[1:])]
        self.assertGreater(sum(1 <= step <= 3 for step in steps[2:]), len(steps[2:]) / 2)   # your steps up, bent into C

    def test_it_colours_your_chord_with_its_tensions_even_outside_your_key(self):
        for pitch in (48, 52, 55, 59):   # Cmaj7, C in the bass
            self.session.hear(0.0, 2.0, pitch)
        bias = self.session._partner(30, [], (0, MAX_PITCH - 1), True)
        d, f_sharp, c_sharp = 62, 66, 61   # its 9th, its #11 (not in C major), and a wrong note
        self.assertGreater(bias[d], bias[c_sharp])
        self.assertGreater(bias[f_sharp], bias[c_sharp])
        self.session.turn(tension=0)
        self.assertAlmostEqual(float(self.session._partner(30, [], (0, MAX_PITCH - 1), True)[d]), float(bias[d]) - duo.TENSION, places=5)

if __name__ == '__main__':
    unittest.main()
