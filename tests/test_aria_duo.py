"""Needs torch and ariautils (pip install git+https://github.com/EleutherAI/aria-utils.git), not the model itself."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from music_transformer import aria_duo
    from music_transformer.aria_duo import NOTE_IDS, ONSET_IDS, SEGMENT, T, TOKENIZER, VELOCITIES, AriaSession
except ImportError as e:   # ariautils is not installed here
    aria_duo = None
    MISSING = str(e)


class UniformModel:
    """Every token equally likely; checks it is always fed where its cache ends."""

    def __init__(self):
        self.cached = 0
        self.starts = []

    def __call__(self, tokens, start):
        assert start <= self.cached, (start, self.cached)
        self.cached = start + len(tokens)
        self.starts.append(start)
        return torch.zeros(TOKENIZER.vocab_size)


class StuckModel(UniformModel):
    """Always wants pitch 61, as loud as can be, at once."""

    def __call__(self, tokens, start):
        logits = super().__call__(tokens, start)
        logits[NOTE_IDS[61, -1]] = 50
        logits[T] = -50
        return logits


class PeakedModel(UniformModel):
    """Wants pitch 62 most of all, and to rest almost as much."""

    def __call__(self, tokens, start):
        logits = super().__call__(tokens, start)
        logits[NOTE_IDS[62, -1]] = 8.4
        logits[T] = 7.8
        return logits


class ChordModel(StuckModel):
    """As StuckModel, and as soon as it may: it piles up chords."""

    def __call__(self, tokens, start):
        logits = super().__call__(tokens, start)
        logits[ONSET_IDS] = -torch.arange(SEGMENT).float()
        return logits


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def replay(session):
    """The notes Aria's tokens say, (tick, pitch, velocity, duration): they must be the session's events."""
    tokens, notes, segment = TOKENIZER.decode(session.seq[len(session.prefix):]), [], 0
    i = 0
    while i < len(tokens):
        if tokens[i] == TOKENIZER.time_tok:
            segment += 1
            i += 1
            continue
        (_, pitch, velocity), (_, onset), (_, duration) = tokens[i:i + 3]
        notes.append((segment * SEGMENT + onset // 10, pitch, velocity, duration // 10))
        i += 3
    return notes


@unittest.skipIf(aria_duo is None, "ariautils is not installed")
class AriaSessionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.clock, self.model = Clock(), UniformModel()
        self.session = AriaSession(self.model, clock=self.clock)

    def fill(self):
        notes = []
        while self.session.ahead() < aria_duo.LOOKAHEAD:
            notes.append(self.session.play())
        return notes

    def follow(self, count):
        notes = []
        for _ in range(count):
            notes.append(self.session.play())
            self.clock.t = notes[-1][0]
        return notes

    def events(self):
        return [(tick, pitch, velocity, duration) for tick, _, pitch, velocity, duration in self.session.events]

    def test_it_plays_after_now_in_order_without_long_rests_and_its_tokens_say_so(self):
        self.clock.t = 3.0
        self.session.hear(3.0, 0.3, 60, 90)
        notes = self.fill()
        onsets = [onset for onset, *_ in notes]
        self.assertEqual(onsets, sorted(onsets))
        self.assertGreaterEqual(onsets[0], 3.0 + aria_duo.MARGIN / 100 - 1e-9)
        self.assertLessEqual(max(b - a for a, b in zip([3.0] + onsets, onsets)), self.session.max_rest / 100 + 1e-9)
        self.assertTrue(all(velocity in VELOCITIES for *_, velocity in notes))
        self.assertTrue(all(hand == ('left' if pitch < aria_duo.SPLIT else 'right') for _, _, pitch, hand, _ in notes))
        self.assertEqual(replay(self.session), self.events())

    def test_long_rests_cross_segments_and_your_note_drops_only_what_has_not_sounded(self):
        self.session.turn(max_rest=800)
        self.session.hear(0.0, 0.3, 60, 70)
        self.follow(30)
        self.assertGreater(self.session.segment, 0)
        self.clock.t += 0.2
        before = len(self.session.events)
        cut = self.session.hear(self.clock.t, 0.3, 64, 100)
        mine = [e for e in self.session.events if e[1]]
        self.assertTrue(all(e[0] <= self.session.now() + aria_duo.MARGIN for e in mine))
        self.assertEqual(cut is None, len(self.session.events) == before + 1)
        self.assertEqual(replay(self.session), self.events())
        self.assertEqual(self.session.events[-1][2:4], (64, 100))
        self.follow(10)
        self.assertEqual(replay(self.session), self.events())

    def test_you_are_heard_when_you_played_even_before_its_notes_already_on_their_way(self):
        self.session.turn(commit=aria_duo.MARGIN)   # its plan does not stand
        self.session.hear(0.0, 0.3, 60, 80)
        notes = self.fill()
        coming = next(n for n in notes if n[0] > 0.5)
        self.clock.t = coming[0] - 0.05   # it sounds within MARGIN: it stays
        cut = self.session.hear(coming[0] - 0.08, 0.3, 64, 90)   # your key reached it a little late
        ticks = [e[0] for e in self.session.events]
        yours = next(i for i, (_, mine, pitch, *_) in enumerate(self.session.events) if not mine and pitch == 64)
        self.assertEqual(ticks[yours], round((coming[0] - 0.08) * 100))
        self.assertIn(round(coming[0] * 100), ticks[yours + 1:])
        self.assertEqual(ticks, sorted(ticks))
        dropped = [n for n in notes if n[0] > self.clock.t + aria_duo.MARGIN / 100 + 1e-9]
        self.assertEqual(cut, dropped[0][0] if dropped else None)
        self.assertEqual(replay(self.session), self.events())
        self.follow(10)
        self.assertEqual(replay(self.session), self.events())

    def test_when_you_let_go_it_hears_how_long_your_note_really_was(self):
        self.session.hear(0.0, 0.3, 60, 80)
        self.follow(20)
        self.session.release(0.0, 60, 2.0)
        self.assertEqual(self.session.events[0][4], 200)
        self.assertEqual(replay(self.session), self.events())
        self.assertEqual(self.session.listener.held, set())
        self.model.starts.clear()
        self.session.play()
        self.assertEqual(self.model.starts[0], len(self.session.prefix) + 2)   # fed again from its duration on

    def test_it_never_doubles_a_key_nor_starts_more_than_its_chord_at_once(self):
        session = AriaSession(StuckModel(), clock=self.clock, chord=2, min_step=20, space=20)
        session.hear(0.0, 0.3, 60, 80)
        notes = [session.play() for _ in range(30)]
        groups = []
        for onset, _, pitch, *_ in notes:
            if groups and onset - groups[-1][0] <= aria_duo.CHORD / 100 + 1e-9:
                groups[-1][1].append(pitch)
            else:
                groups.append((onset, [pitch]))
        for _, pitches in groups:
            self.assertLessEqual(len(pitches), 2)
            self.assertEqual(len(pitches), len(set(pitches)))
        times = [onset for onset, _ in groups]
        self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), 0.2 - 1e-9)

    def test_it_comps_in_chords_while_you_play_and_plays_lines_when_you_leave_it_room(self):
        session = AriaSession(ChordModel(), clock=self.clock, chord=4, solo_chord=1, space=20, min_step=10)
        session.hear(0.0, 0.3, 60, 80)
        comping = [session.play()[0] for _ in range(12)]   # the clock stays by your note: you are playing
        self.clock.t = 5.0                                  # you rest
        alone = [session.play()[0] for _ in range(12)]
        for onsets, most in ((comping, 4), (alone, 1)):
            groups = []   # onsets within CHORD of the first make one chord
            for onset in onsets:
                if groups and onset - groups[-1][0] <= aria_duo.CHORD / 100 + 1e-9:
                    groups[-1].append(onset)
                else:
                    groups.append([onset])
            self.assertEqual(max(len(group) for group in groups), most)
        self.assertEqual(session.state()['mode'], 'alone')
        self.assertIn('solo_chord', session.state()['active'])
        self.assertNotIn('chord', session.state()['active'])

    def test_your_hands_are_where_you_play_an_octave_apart(self):
        self.assertEqual(aria_duo.hands([52, 55, 59, 62, 76, 79, 83]), [(52, 62), (76, 83)])
        self.assertEqual(aria_duo.hands([60, 71]), [(60, 71)])   # a seventh apart: one hand, the keys between too
        self.assertEqual(aria_duo.hands([60, 72]), [(60, 60), (72, 72)])   # an octave: room between
        self.assertEqual(aria_duo.hands([]), [])

    def test_while_you_play_it_keeps_apart_from_your_hands_and_under_your_melody(self):
        self.session.turn(apart=4.0)
        for pitch in (40, 44, 47, 76, 79, 83):   # a left hand and a right hand, held
            self.session.hear(0.0, 30.0, pitch, 80)
        self.clock.t = 0.5                         # you are playing
        pitches = [self.session.play()[2] for _ in range(30)]
        self.assertFalse([p for p in pitches if 38 <= p <= 49 or p >= 74], pitches)
        self.assertTrue([p for p in pitches if 50 <= p <= 73], pitches)

    def test_while_you_only_comp_it_may_take_the_melody_over_you(self):
        self.session.turn(apart=4.0)
        for pitch in (48, 52, 55, 59):             # your left hand alone, held
            self.session.hear(0.0, 30.0, pitch, 80)
        self.clock.t = 0.5
        pitches = [self.session.play()[2] for _ in range(30)]
        self.assertFalse([p for p in pitches if 46 <= p <= 61], pitches)
        self.assertTrue([p for p in pitches if p >= 72], pitches)

    def test_when_your_hands_leave_it_no_room_it_plays_as_it_would(self):
        runs = []
        for apart in (0.0, 6.0):
            torch.manual_seed(0)
            session = AriaSession(UniformModel(), clock=Clock(), lowest=4, highest=5, apart=apart)
            for pitch in (60, 64, 67, 72, 76, 79):   # one hand over all of C4-B5, and your melody on top
                session.hear(0.0, 30.0, pitch, 80)
            runs.append([session.play() for _ in range(20)])
        self.assertEqual(runs[0], runs[1])

    def test_a_bass_under_its_octaves_neither_moves_it_nor_makes_it_rest(self):
        runs = []
        for apart in (0.0, 4.0):
            torch.manual_seed(0)
            session = AriaSession(UniformModel(), clock=Clock(), lowest=4, apart=apart)
            session.hear(0.0, 3.0, 29, 80)   # F1, held: under C4, its lowest
            runs.append([session.play() for _ in range(20)])
        self.assertEqual(runs[0], runs[1])

    def test_where_it_plays_never_changes_how_likely_it_is_to_rest(self):
        for temperature in (0.5, 0.95, 1.5):
            runs = []
            for apart in (0.0, 6.0):
                torch.manual_seed(0)
                session = AriaSession(PeakedModel(), clock=self.clock, apart=apart, temperature=temperature,
                                      max_rest=800)   # <T> competes
                for pitch in (60, 64):                  # your hand, round the key it wants most
                    session.hear(0.0, 3.0, pitch, 80)
                seen, sample = [], session._sample
                session._sample = lambda logits, seen=seen, sample=sample: seen.append(logits.clone()) or sample(logits)
                session.play()
                runs.append(seen)
            # whether it plays: the same logits, so the same chance after min_p, at any temperature
            self.assertTrue(torch.equal(runs[0][0], runs[1][0]))
            self.assertTrue(torch.isfinite(runs[0][0][T]))   # resting is one of its choices there
            # where: <T> out of it, and the key in your hand pushed away
            where = [[logits for logits in seen if logits[T] == -float('inf')][0] for seen in runs]
            self.assertLess(float(where[1][NOTE_IDS[62, -1]]), float(where[0][NOTE_IDS[62, -1]]) - 5)

    def test_when_you_rest_it_sings_up_around_your_right_hand(self):
        for i, pitch in enumerate((43, 47, 50, 74, 76, 77, 79, 81)):
            self.session.hear(i * 0.1, 0.3, pitch, 80)
        self.clock.t = 3.0                          # you rest
        self.session.turn(lead=4.0)
        pitches = [pitch for _, _, pitch, *_ in self.follow(40)]
        self.assertTrue(all(p >= 65 for p in pitches), pitches)   # an octave under the middle of your right hand, 77
        self.assertGreaterEqual(sorted(pitches)[len(pitches) // 2], 77)

    def test_when_you_rest_after_your_left_hand_alone_it_sings_from_c5(self):
        for i, pitch in enumerate((36, 43, 48, 52, 55)):
            self.session.hear(i * 0.1, 0.3, pitch, 80)
        self.clock.t = 3.0
        self.session.turn(lead=4.0)
        self.assertTrue(all(pitch > 60 for _, _, pitch, *_ in self.follow(40)))

    def test_it_plays_only_in_the_octaves_it_is_given(self):
        self.session.turn(lowest=5, highest=4)
        self.session.hear(0.0, 0.3, 30, 80)
        self.assertTrue(all(60 <= pitch <= 83 for _, _, pitch, *_ in self.follow(60)))

    def test_it_prefers_the_pitch_classes_you_are_playing(self):
        self.session.turn(chord_tone=3.0)
        for i, pitch in enumerate((60, 64, 67, 72)):
            self.session.hear(i * 0.2, 0.2, pitch, 80)
        self.clock.t = 0.6
        notes = []
        for _ in range(60):
            notes.append(self.session.play())
            self.clock.t = notes[-1][0]
            self.session.hear(self.clock.t, 0.2, (60, 64, 67)[len(notes) % 3], 80)
        in_chord = sum(pitch % 12 in (0, 4, 7) for _, _, pitch, *_ in notes) / len(notes)
        self.assertGreater(in_chord, 0.4)   # 3 pitch classes of 12: 25% without ears

    def test_a_full_context_keeps_the_last_moments_from_tick_zero_and_rebuilds_the_cache(self):
        context = aria_duo.CONTEXT
        aria_duo.CONTEXT = 300
        try:
            self.session.hear(0.0, 0.3, 60, 80)
            longest = 0
            for _ in range(200):
                self.follow(1)
                longest = max(longest, len(self.session.seq))
            self.assertGreater(self.session.base, 0)
            self.assertLessEqual(longest, 300)
            self.assertEqual(self.session.events[0][0], 0)
            self.assertEqual(replay(self.session), self.events())
        finally:
            aria_duo.CONTEXT = context

    def test_it_says_what_it_does(self):
        self.assertEqual(self.session.state()['mode'], 'listening')
        for i, pitch in enumerate((55, 59, 62, 65)):
            self.clock.t = i * 0.1
            self.session.hear(self.clock.t, 1.0, pitch, 80)
        self.clock.t = 0.35
        state = self.session.state()
        self.assertEqual((state['mode'], state['chord']), ('accompanying', 'G7'))
        self.assertIn('apart', state['active'])
        self.assertNotIn('lead', state['active'])
        self.clock.t = 5.0
        state = self.session.state()
        self.assertEqual(state['mode'], 'alone')
        self.assertIn('lead', state['active'])
        self.assertNotIn('apart', state['active'])

    def test_it_takes_a_breath_when_its_phrase_has_run_out_of_notes(self):
        session = AriaSession(StuckModel(), clock=self.clock, phrase=4, breath=120, solo_chord=1)
        session.hear(0.0, 0.3, 60, 80)
        self.clock.t = 5.0                          # you rest: every note of its is one of its phrase
        onsets = [session.play()[0] for _ in range(40)]
        phrases = [1]
        for a, b in zip(onsets, onsets[1:]):
            if b - a >= 1.2 - 1e-9:
                phrases.append(1)
            else:
                phrases[-1] += 1
        self.assertGreater(len(phrases), 6)
        self.assertLessEqual(max(phrases), 6)       # about 4: from 3 to 6

    def test_it_plays_on_after_your_last_note_then_waits_for_you(self):
        self.session.turn(timeout=200)
        self.session.hear(0.0, 0.3, 60, 80)
        for _ in range(80):
            self.clock.t += 0.1
            self.fill()
        self.assertEqual(self.session.state()['mode'], 'listening')
        self.assertEqual(self.fill(), [])
        self.assertLessEqual(self.session._last(), 200 + aria_duo.LOOKAHEAD + self.session.max_rest)
        self.session.hear(self.clock.t, 0.3, 62, 80)
        self.assertGreater(len(self.fill()), 0)

    def planned(self, *pitches, **knobs):
        """A session in C major, 0.3 s in, with its plan: `pitches` from 0.6 s on, 0.1 s apart."""
        session = AriaSession(UniformModel(), clock=self.clock, **knobs)
        for i, pitch in enumerate((60, 64, 67, 72, 62, 65, 69, 71) * 3):
            session.hear(i * 0.01, 0.1, pitch, 80)
        for i, pitch in enumerate(pitches):
            session._add((60 + 10 * i, True, pitch, 80, 10))
        self.clock.t = 0.3
        return session

    def test_its_plan_stands_when_your_note_does_not_contradict_it(self):
        session = self.planned(72, 74, 76, commit=100)
        self.assertIsNone(session.hear(0.3, 0.2, 64, 80))
        self.assertEqual([e[2] for e in session.events if e[1]], [72, 74, 76])
        self.assertEqual(replay(session), [(t, p, v, d) for t, _, p, v, d in session.events])

    def test_your_note_drops_its_plan_from_the_note_that_contradicts_it(self):
        session = self.planned(72, 73, 76, commit=100)   # Db in C major
        self.assertAlmostEqual(session.hear(0.3, 0.2, 64, 80), 0.7)
        self.assertEqual([e[2] for e in session.events if e[1]], [72])

    def test_its_plan_does_not_stand_past_commit(self):
        session = self.planned(72, 74, 76, commit=40)
        self.assertAlmostEqual(session.hear(0.3, 0.2, 64, 80), 0.8)

    def test_it_plays_as_loud_as_you_do(self):
        levels = []
        for velocity in (30, 110):
            torch.manual_seed(0)
            session = AriaSession(UniformModel(), clock=Clock(), follow=4.0)
            session.hear(0.0, 0.3, 60, velocity)
            notes = [session.play() for _ in range(40)]
            levels.append(sum(note[4] for note in notes) / len(notes))
        self.assertGreater(levels[1], levels[0] + 30)

    def test_when_your_phrase_ends_it_answers_with_your_intervals(self):
        for i, pitch in enumerate((72, 74, 76)):    # up a tone, and a tone
            self.session.hear(i * 0.3, 0.2, pitch, 80)
        self.clock.t = 2.0
        self.assertEqual(self.session.state()['mode'], 'answering')
        self.session._add((200, True, 67, 80, 10))   # its answer's first note: it goes up a tone from there
        answer = self.session._partner(215)
        self.session.turn(echo=0)
        self.assertEqual((answer - self.session._partner(215)).nonzero().tolist(), [[69]])

    def test_you_can_force_it_into_a_scale_and_fix_its_tempo(self):
        self.session.turn(root=5, scale=list(aria_duo.FORCED).index('blues') + 1, bpm=120)
        self.assertEqual(self.session.state()['key'], 'F blues')
        self.session.hear(0.0, 0.3, 65, 80)
        pitches = [pitch for _, _, pitch, *_ in self.follow(40)]
        self.assertTrue(all((pitch - 5) % 12 in aria_duo.FORCED['blues'] for pitch in pitches), pitches)
        self.assertEqual(self.session.state()['bpm'], 120)

    def test_it_plays_about_as_much_as_you_do(self):
        counts = []
        for yours in (1, 6):
            session = AriaSession(StuckModel(), clock=Clock(), busy=1.0, breath=0, chord=1)
            for i in range(yours):
                session.hear(i * 0.5, 0.3, 60 + i, 80)
            onsets = [session.play()[0] for _ in range(40)]
            counts.append(max(sum(a <= b < a + 8 for b in onsets) for a in onsets))   # in any 8 s
        self.assertEqual(counts, [2, 7])   # one for each of yours, and one more

    def test_it_moves_on_from_the_keys_it_has_just_played(self):
        session = AriaSession(PeakedModel(), clock=self.clock, variety=6.0, lowest=4, highest=4)
        session.hear(0.0, 0.3, 30, 80)
        self.clock.t = 5.0
        pitches = [session.play()[2] for _ in range(24)]
        self.assertLess(pitches.count(62), 8)        # the key the model wants most of all, every time
        session.turn(variety=0)
        self.assertGreater([session.play()[2] for _ in range(24)].count(62), 16)

    def test_over_a_chart_it_plays_the_chord_of_the_bar_at_the_charts_tempo(self):
        from music_transformer.chart import CHARTS, chord
        self.assertEqual([chord(name) for name in ('C-7', 'F7b9', 'B^7', 'Aø', 'Gm6', 'C7/E', 'Ebmaj9', 'F#o7')],
                         [(0, 'm7'), (5, '7'), (11, 'maj7'), (9, 'm7b5'), (7, 'm6'), (0, '7'), (3, 'maj7'), (6, 'dim7')])
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'autumn_leaves')
        self.assertEqual(len(CHARTS[tune].bars), 32)
        self.session.turn(chart=tune + 1, chord_tone=4.0, apart=0)
        self.assertEqual((self.session.bpm, self.session.metronome, self.session.pace), (0, 0, 120))   # your knobs
        self.assertEqual(self.session.state()['chord'], 'Cm7 > F7')
        self.clock.t = 10.2                          # bar 6 of 2 s each: D7, whose F# is not in G minor
        self.assertEqual(self.session.state()['key'], 'Autumn Leaves 6/32')
        self.session.hear(10.2, 0.3, 62, 80)
        bias = self.session._partner(self.session.now() + 30)
        self.assertGreater(bias[66], bias[65] + 3)   # F# over F
        self.assertLess(bias[65], -5)                # F is not in D7's scale here: it is not played
        self.session.turn(bpm=60)                    # half the tempo: the same bar, twice as long
        state = self.session.state()
        self.assertEqual((state['bpm'], state['chart']['bar'], state['chart']['bars'][5]), (60, 5, ['D7']))
        self.clock.t = 13.0
        self.assertEqual(self.session.state()['chart']['bar'], 5)
        self.assertAlmostEqual(self.session.began, 10.2 - 2 * 10.2)
        self.session.turn(chart=0)
        self.assertIsNone(self.session.listener.chart)
        self.assertNotIn('chart', self.session.state())

    def test_with_wait_it_takes_turns_with_you(self):
        self.session.turn(wait=80)
        self.session.hear(0.0, 0.3, 60, 80)
        notes = self.fill()
        self.assertGreaterEqual(min(note[0] for note in notes), 0.8 - 1e-9)   # once you have rested
        self.assertEqual(self.session.state()['mode'], 'waiting')
        self.clock.t = 0.5
        self.assertAlmostEqual(self.session.hear(0.5, 0.3, 62, 80), notes[0][0])   # it gives way: all of its plan
        self.assertGreaterEqual(min(note[0] for note in self.fill()), 1.3 - 1e-9)

    def test_knobs_hold_from_the_start_and_stay_within_their_range(self):
        session = AriaSession(UniformModel(), clock=self.clock, min_p=2.0, genre=None)
        self.assertEqual(session.knobs()['min_p'], 0.3)
        self.assertEqual(TOKENIZER.decode(session.prefix), [('prefix', 'instrument', 'piano'), TOKENIZER.bos_tok])


if __name__ == '__main__':
    unittest.main()
