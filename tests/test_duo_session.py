"""Needs torch and anticipation: run it where the model runs (python -m unittest tests.test_duo_session)."""
import os
import random
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
        self.enterContext(unittest.mock.patch('music_transformer.chart.COUNT', 0))   # charts begin at once here
        self.session = DuoSession(self.model, 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1)
        # a duet over Autumn Leaves, playing: 120 bpm, a bar is 2 s; it waits for your first note

    def tune(self, name):
        from music_transformer.chart import CHARTS
        return next(i for i, chart in enumerate(CHARTS) if chart.name == name) + 1

    def test_a_tune_begins_after_a_bar_of_count_in_and_it_starts_by_itself_when_it_comps(self):
        with unittest.mock.patch('music_transformer.chart.COUNT', 1):
            self.clock.t = 0.5
            self.session.turn(chart=self.tune('autumn_leaves'), playing=1, part=0)   # 120 bpm: a bar is 2 s
        self.assertEqual(self.session.began, 4.0)           # the next bar, and one of count-in
        self.assertEqual(self.session.state()['chart']['count'], 8)
        self.session.begin()
        self.assertIsNone(self.session.base)                # not yet
        self.clock.t = 3.2
        self.session.begin()
        notes = list(self.fill())
        self.assertGreater(len(notes), 0)
        self.assertGreaterEqual(min(n[0] for n in notes), 4.0 - 0.03)   # none in the count-in
        for _ in range(60):                                 # it plays on by itself
            self.clock.t += 0.1
            list(self.fill())
        self.session.hear(self.clock.t, 0.3, 60)            # and you join when you like
        self.assertGreaterEqual(self.session.heard_at, 0)
        duet = DuoSession(self.model, 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1)
        duet.begin()
        self.assertIsNone(duet.base)                        # in a duet it waits for you

    def test_a_tune_plays_only_between_play_and_stop(self):
        self.session.turn(chart=self.tune('autumn_leaves'), part=0, playing=0)   # chosen, not played
        for _ in range(30):
            self.clock.t += 0.1
            self.session.begin()
        self.assertIsNone(self.session.base)
        self.assertEqual(self.session.state()['chart']['bar'], -1)   # its page, no bar lit
        self.session.turn(playing=1)
        self.session.begin()
        self.assertGreater(len([n for n in self.fill() if n]), 0)
        self.session.turn(playing=0)
        self.assertLessEqual(self.session._last_time(), self.session.now())   # what it had not sounded is gone
        self.assertEqual([n for n in self.fill() if n], [])   # nothing more once stopped

    def test_trading_fours_you_solo_four_bars_then_it_does(self):
        self.session.turn(chart=self.tune('blues_in_f'), playing=1, part=3)
        bar = 4 * 60 / 110
        self.assertEqual([self.session._solos(k * bar + 0.1, True) for k in (0, 3, 4, 7, 8, 12, 15, 16)],
                         [False, False, True, True, False, True, True, False])   # on through the 12 bars' end
        self.session.hear(0.0, 0.3, 53)
        self.clock.t = 4.5 * bar
        self.assertEqual(self.session.state()['doing'], 'its turn')

    def test_over_a_tune_its_bass_walks_root_first_and_approaches_the_next_chord(self):
        self.session.turn(chart=self.tune('blues_in_f'), playing=1, part=0)   # F7 | Bb7: a beat is 0.545 s
        self.session.hear(0.0, 0.3, 65)
        beat = 60 / 110
        first = self.session._partner(self.session._tick(0.02), [], (36, 51), False)
        self.assertEqual(int(first[36:52].argmax()) + 36 in (41,), True)      # F2 on the first beat
        last = self.session._partner(self.session._tick(3 * beat), [], (36, 51), False)   # the bar's last beat
        self.assertIn(int(last[36:52].argmax()) + 36, (45, 47))               # A or B: a semitone from Bb
        second = self.session._partner(self.session._tick(beat), [(0, NOTE_OFFSET + 32 * MAX_PITCH + 41)], (36, 51), False)
        self.assertIn(int(second[36:52].argmax()) + 36, (39, 45))             # a chord tone near F: Eb or A
        hands = self.session._hands(False)
        self.assertEqual((hands[1][1] > 45, hands[1][2], hands[1][3][1]), (True, 1, 51))   # a note a beat, low

    def test_over_a_tune_key_follow_says_how_soon_it_leaves_the_tunes_chord_for_the_one_you_hold(self):
        self.session.turn(chart=self.tune('blues_in_f'), playing=1, part=0)   # F7 | Bb7 | F7: a beat is 0.545 s
        for pitch in (59, 63, 66, 69):                             # you hold B7 over them, for 4 s
            self.session.hear(0.0, 4.0, pitch)

        def bass(at):
            return int(self.session._partner(self.session._tick(at), [], (36, 51), False)[36:52].argmax()) + 36

        def leans(at):   # how much its right hand prefers your chord's F# to the tune's F, over your hands
            bias = self.session._partner(self.session._tick(at), [], (0, 127), True)
            return float(bias[78] - bias[77])
        self.session.turn(stay=0.995)                              # steady: the tune's chord, whatever you play
        self.assertEqual((bass(0.02), leans(0.02) < -1), (41, True))
        self.session.turn(stay=0.85)                               # quick: yours at once, its bass on B
        self.assertEqual((bass(0.02), leans(0.02) > 1), (47, True))
        self.assertEqual((bass(4.4), leans(4.4) < -1), (41, True))   # you let it go: F7 again, in the third bar
        self.session.turn(stay=0.97)                               # in between: once you have held it over a bar
        self.assertEqual((bass(0.02), bass(3.0)), (41, 47))

    def test_over_a_tune_its_bass_plays_every_beat_while_its_right_hand_plays_and_breathes(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=0,
                             phrase=4, breath=200)           # 120 bpm: a beat is 0.5 s
        session.hear(0.0, 0.3, 60)
        notes = []
        for _ in range(80):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        bass = sorted({round(n[0] * 2) for n in notes if n[3] == 'bass'})
        self.assertEqual(bass, list(range(bass[0], bass[-1] + 1)))   # no beat without its bass note
        self.assertGreater(len(bass), 12)
        self.assertTrue(all(abs(n[0] * 2 - round(n[0] * 2)) < 0.07 for n in notes if n[3] == 'bass'))
        self.assertGreater(sum(n[3] == 'right' for n in notes), 5)

    def test_over_a_bossa_its_bass_plays_root_and_fifth_on_one_the_and_of_two_three_and_the_and_of_four(self):
        from music_transformer.chart import CHARTS
        self.assertEqual(CHARTS[self.tune('black_orpheus') - 1].bass, 'bossa')
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('black_orpheus'), playing=1, part=0)
        session.hear(0.0, 0.3, 60)                               # 120 bpm: a beat is 0.5 s
        notes = []
        for _ in range(80):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        at = [n[0] * 2 % 4 for n in notes if n and n[3] == 'bass']   # in beats, in the bar
        self.assertGreater(len(at), 12)                         # 4 bars: 16 notes
        self.assertTrue(all(min(abs(a - p) for p in (0, 1.5, 2, 3.5, 4)) < 0.15 for a in at), at)
        for p in (0, 1.5, 2, 3.5):                               # the whole figure, not only its downbeats
            self.assertGreater(sum(abs(a - p) < 0.15 for a in at), 3, p)

        def bass(beat):                                          # Am7 | Bm7b5 E7b9 | ...
            return int(session._partner(session._tick(beat / 2), [], (36, 51), False)[36:52].argmax()) + 36
        self.assertEqual((bass(0.02) % 12, bass(1.5) % 12, bass(2) % 12), (9, 4, 4))   # A, then its fifth E
        self.assertEqual(bass(3.5) % 12, 11)                     # B: the next bar's root, ahead of it

    def test_over_a_bossa_its_bass_is_mostly_legato_and_moves_now_and_then(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('black_orpheus'), playing=1, part=0)
        session.hear(0.0, 0.3, 60)                               # 120 bpm: a beat is 0.5 s
        notes = []
        for _ in range(160):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        bass = [n for n in notes if n and n[3] == 'bass']
        gaps = [(b[0] - a[0], a[1]) for a, b in zip(bass, bass[1:])]
        self.assertTrue(all(lasts >= gap / 2 - 0.02 for gap, lasts in gaps), gaps)      # never a fanfare's staccato
        self.assertGreater(sum(abs(lasts - gap) < 0.02 for gap, lasts in gaps), len(gaps) / 2)   # mostly tied on
        self.assertLess(sum(abs(lasts - gap) < 0.02 for gap, lasts in gaps), len(gaps))   # but not always

        def second(beat):                                        # the pitch class it likes next best
            bias = session._partner(session._tick(beat / 2), [], (36, 51), False)[36:52]
            return [int(p) % 12 for p in bias.argsort(descending=True)[:2] + 36]
        self.assertEqual(second(1.5)[0], 4)                      # over Am7: its fifth E
        self.assertIn(second(1.5)[1], (0, 7))                    # or its third C, its seventh G
        self.assertEqual(second(3.5)[0], 11)                     # ahead of Bm7b5: B
        self.assertEqual(second(3.5)[1], 0)                      # or C, a semitone over it

    def test_over_a_tune_its_right_hand_plays_on_though_the_model_would_rather_walk_its_bass_alone(self):
        class Bassist(UniformModel):
            def __call__(self, inp, past_key_values=None, use_cache=True):
                out = super().__call__(inp, past_key_values, use_cache)
                out.logits[..., duo.TIME_OFFSET:duo.TIME_OFFSET + duo.MAX_TIME] += torch.arange(duo.MAX_TIME)   # on the
                out.logits[..., NOTE_OFFSET + duo.LEFT * MAX_PITCH:NOTE_OFFSET + (duo.LEFT + 1) * MAX_PITCH] += 30   # beat, the bass
                return out

        for part, heard in ((0, False), (1, True), (1, False)):   # it comps; a duet, you playing or resting
            session = DuoSession(Bassist(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=part,
                                 max_rest=100, breath=60)
            start = self.clock.t
            session.hear(0.0, 0.3, 60)
            notes = []
            for step in range(200):
                self.clock.t += 0.1
                if heard and step % 5 == 0:
                    session.hear(self.clock.t - start - 0.05, 0.3, 60 + step % 7)
                while session.ahead() < duo.LOOKAHEAD:
                    notes.append(session.play())
            right = sorted(n[0] for n in notes if n and n[3] == 'right' and n[0] > 3)
            self.assertGreater(len(right), 8, part)
            self.assertLess(max(b - a for a, b in zip(right, right[1:])), 3.5, part)   # a second, a breath, a bar: it
                                                                                      # takes the last moment it may
            self.assertGreater(sum(bool(n) and n[3] == 'bass' for n in notes), 30, part)   # and its bass walks on
            self.clock.t = start

    def test_soloing_over_its_walking_bass_it_comps_a_chord_now_and_then_under_its_line(self):
        from music_transformer import chart
        for knobs in ({'part': 1, 'licks': 0}, {'part': 1, 'licks': 1}, {'part': 1, 'licks': 0.5, 'grid': 0, 'bpm': 200},
                      {'part': 0}, {'part': 2}):
            clock = Clock()
            session = DuoSession(UniformModel(), 'cpu', clock=clock, chart=self.tune('autumn_leaves'), playing=1, **knobs)
            session.hear(0.0, 0.3, 60)                       # then you rest: in a duet it solos
            notes = []
            for _ in range(300):
                clock.t += 0.1
                session.begin()
                while session.ahead() < duo.LOOKAHEAD:
                    notes.append(session.play())
            beat = 60 / session.pace
            first = session.began + 8 * beat                 # your note is long past: it no longer comps under you
            notes = [note for note in notes if note and first <= note[0] < 30]
            under = [n for n in notes if n[3] == 'right' and sum(m[0] == n[0] and m[3] == 'right' and
                                                                 duo.UNDER[0] <= m[2] <= duo.UNDER[1] for m in notes) >= 3
                     and duo.UNDER[0] <= n[2] <= duo.UNDER[1]]
            chords = sorted({n[0] for n in under})
            if knobs['part'] != 1:                           # it comps, or solos for you to comp: none of these
                self.assertEqual([n for n in notes if n[2] < 60 and n[3] == 'right' and knobs['part'] == 2], [])
                self.assertEqual(len(session._my_line(session.comp)), 0)
                continue
            bars = (30 - first) / (4 * beat)
            self.assertTrue(0.5 * bars < len(chords) < 1.6 * bars, (len(chords), bars))   # about one a bar
            for onset in chords:                             # where a pianist's left hand comps, three notes, short
                bar, spot = divmod((onset - session.began) / beat + 0.02, 4)
                self.assertTrue(any(abs(spot - 0.02 - at) < 0.2 for at in chart.stabs(session, int(bar))), (onset, spot))
                self.assertIn(len({n[2] for n in under if n[0] == onset}), (3, 4))   # and a note of its line, at times
                self.assertGreaterEqual(sum(n[1] <= 1.5 * beat + 0.01 for n in under if n[0] == onset), 3)
            line = [n for n in notes if n[3] == 'right' and n not in under]
            self.assertGreater(len(line), 2 * len(chords))   # its line goes on over them, and its bass under
            self.assertGreater(sum(n[3] == 'bass' for n in notes), 0.8 * 4 * bars)

    def test_over_a_tune_its_part_says_what_its_hands_do_whatever_the_other_knobs(self):
        self.session.turn(chart=self.tune('blues_in_f'), playing=1, part=0)
        comps = self.session._hands(False, self.session._solos(1.0, False))
        self.assertEqual([(most, register) for _, _, most, register in comps], [(3, (53, 76)), (1, (28, 51))])
        self.session.turn(part=2)
        solos = self.session._hands(False, self.session._solos(1.0, False))
        self.assertEqual([(most, register) for _, _, most, register in solos], [(1, (60, 96))])   # a line, no bass
        self.session.turn(part=0, chord=1)           # whatever its chord knob: three notes, and its bass walks
        self.assertEqual([(most, register) for _, _, most, register in self.session._hands(False, False)],
                         [(3, (53, 76)), (1, (28, 51))])
        self.session.turn(part=1)                    # a duet: chords under you, a line when you rest, the bass on
        self.assertEqual([hand[2] for hand in self.session._hands(True, self.session._solos(1.0, True))], [3, 1])
        self.assertEqual([hand[2] for hand in self.session._hands(False, self.session._solos(1.0, False))], [1, 1])
        self.session.turn(lowest=5, highest=6)       # its octaves still hold its line, C5 to B6; its bass keeps its own
        self.assertEqual([hand[3] for hand in self.session._hands(False, True)], [(72, 95), (28, 51)])

    def test_over_a_tune_its_octaves_bound_its_right_hand_and_its_bass_walks_on_every_beat(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=0,
                             lowest=5, highest=7)        # 120 bpm: a beat is 0.5 s
        session.hear(0.0, 0.3, 60)
        notes = []
        for _ in range(80):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        bass = sorted({round(n[0] * 2) for n in notes if n and n[3] == 'bass'})
        self.assertEqual(bass, list(range(bass[0], bass[-1] + 1)))   # no beat without its bass note
        self.assertGreater(len(bass), 12)
        self.assertTrue(all(n[2] < 52 for n in notes if n and n[3] == 'bass'))
        self.assertTrue(all(n[2] >= 72 for n in notes if n and n[3] == 'right'))

    def test_over_a_tune_it_comps_more_than_four_chords_before_a_breath(self):
        class Eager(UniformModel):   # every note as soon as it may: its chords full
            def __call__(self, inp, past_key_values=None, use_cache=True):
                out = super().__call__(inp, past_key_values, use_cache)
                out.logits[0, -1, TIME_OFFSET:TIME_OFFSET + 10000] = -torch.arange(10000).float()
                return out

        session = DuoSession(Eager(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=0, chord=4)
        session.hear(0.0, 0.3, 60)
        notes = []
        for _ in range(100):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        onsets = sorted(n[0] for n in notes if n and n[3] == 'right')
        breath = next(b for a, b in zip(onsets, onsets[1:]) if b - a > 1.4)   # a breath is 1.5 s
        self.assertGreater(sum(onsets.count(onset) == 4 for onset in set(onsets) if onset < breath), 4)

    def solo(self, seconds, **knobs):
        """(the session, its notes) after it soloed alone over Autumn Leaves (120 bpm: a bar is 2 s) for `seconds`."""
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=2, **knobs)
        session.begin()
        notes = []
        for _ in range(round(seconds * 10)):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        return session, [note for note in notes if note]

    def test_over_a_tune_harmony_outside_leaves_the_chords_scale_and_inside_keeps_to_it(self):
        from music_transformer import chart
        outside = {}
        for out_of_key in (0.0, 5.0):                # harmony fully outside, fully inside
            with unittest.mock.patch('music_transformer.chart.way', return_value='scale'):
                session, notes = self.solo(16, licks=0, arc=0, out_of_key=out_of_key)
            outside[out_of_key] = sum(not session.listener.chord_scale(n[0])[n[2] % 12] for n in notes)
            session.out_of_key, at = out_of_key, [k * 2 * 4 * 60 / session.pace for k in range(2000)]
            outside[out_of_key, 'ways'] = sum(chart.way(session, seconds) in ('bebop', 'blues') for seconds in at)
        self.assertGreater(outside[0.0], 5)
        self.assertEqual(outside[5.0], 0)
        self.assertGreater(outside[0.0, 'ways'], outside[5.0, 'ways'] + 150)   # and more often bebop or blues

    def test_over_a_tune_keeping_apart_from_your_hands_moves_its_right_hand_only(self):
        self.session.turn(chart=self.tune('blues_in_f'), playing=1, part=1)
        for pitch in (41, 45, 48, 65, 69):           # you play in its bass's register, and in its right hand's
            self.session.hear(0.0, 2.0, pitch)
        tick = self.session._tick(0.6)

        def bias(apart, *hand):
            self.session.turn(apart=apart)
            return self.session._partner(tick, [], *hand)

        for hand in (((28, 51), False), ((52, 69), False, True)):   # its walking bass, the chords under its solo
            self.assertTrue(torch.equal(bias(6.0, *hand), bias(0.0, *hand)))
        self.assertLess(float((bias(6.0, (53, 76), True) - bias(0.0, (53, 76), True)).min()), -5)

    def test_soloing_it_quotes_a_phrase_played_over_the_same_chords_in_the_key_of_the_tune(self):
        from music_transformer import licks
        session, notes = self.solo(2.5, licks=1, out_of_key=0)   # the first bar: Cm7, then F7; outside: as it was played
        self.assertIn(' · ', session.state()['chart'].get('quote', ''))
        quoted = session.quotes[0]
        played = [(onset, pitch) for onset, _, pitch, _ in notes if onset <= quoted[-1][0] + 0.02]
        self.assertGreater(len(played), 1)               # (it came in after its first notes: the duo had just begun)
        for onset, pitch in played:                      # note for note
            self.assertTrue(any(abs(onset - at) < 0.08 and pitch == note for at, note, _ in quoted), (onset, pitch))
        ours = lambda lick: [(round(at + (1 / 6 if abs(at % 1 - 0.5) < 0.04 else 0), 2), pitch % 12) for at, _, pitch in lick]
        mine = [(round(at * 2, 2), pitch % 12) for at, pitch, _ in quoted]
        self.assertTrue(any(ours(lick) == mine for chords in licks.DATA['licks'] if chords.startswith('min.0.4')
                            for _, lick in licks.DATA['licks'][chords]))   # on C: some soloist's, from a ii on
        self.assertEqual(licks.quote(licks.DATA, random.Random(0), [(0, 'dim7')] * 3 + [(1, 'm7')] * 5, 60, 96, 72), None)

    def test_its_licks_knob_says_how_much_of_its_solo_it_quotes(self):
        quoted = {}
        for share in (0.0, 0.5, 1.0):
            session, _ = self.solo(0.5, licks=share)
            bars = []
            for bar in range(200):                           # the bars it comes to, one after another
                session.quotes = {b: notes for b, notes in session.quotes.items() if b >= bar}
                session._lick(session._tick(session.began + bar * 2.0 + 0.01))
                bars.append(bool(session.quotes[bar]))
            quoted[share] = sum(bars) / len(bars)
        self.assertEqual((quoted[0.0], quoted[1.0]), (0, 1))   # its own lines only; one phrase after another
        self.assertTrue(0.45 < quoted[0.5] < 0.55, quoted)     # half of its bars

    def test_it_quotes_a_phrase_over_two_chords_wherever_in_the_tune_they_begin(self):
        from music_transformer import licks
        asked, real = [], licks.quote

        def quote(library, rng, chords, *rest):
            asked.append(chords)
            return real(library, rng, chords[:4] if len(asked) == 1 else chords[:8], *rest)   # one bar first: Cm7, then two

        with unittest.mock.patch('music_transformer.licks.quote', quote), unittest.mock.patch('music_transformer.licks.LONGER', 1):
            session, notes = self.solo(7, licks=1, vary=0)
        # the next begins at the second bar, off the tune's pairs of bars: F7 | Bbmaj7, then Ebmaj7 | Am7b5
        self.assertEqual([[chords[0], chords[4]] for chords in asked[:3]],
                         [[(0, 'm7'), (5, '7')], [(5, '7'), (10, 'maj7')], [(3, 'maj7'), (9, 'm7b5')]])
        self.assertEqual([len(notes_) > 3 for bar, notes_ in sorted(session.quotes.items())], [True] * len(session.quotes))
        f7 = [pitch for onset, _, pitch, _ in notes if 2.0 <= onset < 4.0]
        fits = [lick for _, lick in licks.DATA['licks']['dom.0.4 maj.5.4'] if all(
            any((pitch - key - 5) % 12 == 0 for *_, key in lick) for pitch in f7)]
        self.assertGreaterEqual(len(fits), 1)            # on F: a V-I of some soloist
        self.assertGreater(len(f7), 0)

    def test_it_quotes_one_soloist_whole_over_a_progression_of_as_many_as_four_bars(self):
        from music_transformer import licks
        with unittest.mock.patch('music_transformer.licks.LONGER', 1):
            session, notes = self.solo(9, licks=1, out_of_key=0)   # Cm7 | F7 | Bbmaj7 | Ebmaj7: 8 s, as they were played
        whole = [[pitch % 12 for *_, pitch in lick] for _, lick in licks.DATA['licks']['min.0.4 dom.5.4 maj.10.4 maj.3.4']]
        self.assertIn([pitch % 12 for _, pitch, _ in session.quotes[0]], whole)   # its ii-V-I-IV, one soloist's four bars
        self.assertTrue(all(session.quotes[bar] is session.quotes[0] for bar in range(4)))
        self.assertGreater(sum(4.0 <= onset < 8.0 for onset, *_ in notes), 2)   # and it plays them through
        # a soloist it has just quoted is quoted again over the next chords, when they played them
        with unittest.mock.patch('music_transformer.licks.SAME', 1):
            name = licks.quote(licks.DATA, random.Random(1), [(9, 'm7b5')] * 4 + [(2, '7')] * 4, 60, 96, 72, 'John Coltrane')[0]
        self.assertEqual(name.split(' · ')[0], 'John Coltrane')
        with unittest.mock.patch('music_transformer.licks.LONGER', 1):
            self.assertEqual(licks.quote(licks.DATA, random.Random(1), [(0, 'm7')] * 4 + [(5, '7')] * 4 + [(10, 'maj7')] * 4, 60, 96, 72)[2], 3)

    def test_a_pickup_leads_into_the_next_phrase_by_step_with_a_chromatic_approach_or_encloses_it(self):
        from music_transformer import licks
        c_major = [pc in (0, 2, 4, 5, 7, 9, 11) for pc in range(12)]
        lead = lambda last, target, room, chromatic=True: licks.pickup(last, target, room, c_major, chromatic)
        self.assertEqual(lead(71, 74, 2), [72, 73])        # B | C C# | D: a passing note, then the chromatic approach
        self.assertEqual(lead(71, 74, 1), [73])            # room for one note: the approach alone
        self.assertEqual(lead(71, 74, 3), [72, 73])        # no further back than the note it left
        self.assertEqual(lead(65, 74, 5), [71, 72, 73])    # three notes at most
        self.assertEqual(lead(77, 74, 2), [76, 75])        # from over it: F | E Eb | D
        self.assertEqual(lead(72, 74, 2), [76, 73])        # too near to walk: an enclosure, C | E C# | D
        self.assertEqual(lead(74, 74, 3), [76, 73])        # and the same note again
        self.assertEqual(lead(71, 72, 2), [])              # a semitone from it already: nothing to add
        self.assertEqual(lead(71, 74, 0), [])              # no room
        self.assertEqual(lead(69, 74, 1, False), [72])     # the approach falls on the beat: a step of the scale

    def test_now_and_then_it_leads_from_a_phrase_it_quotes_into_the_next(self):
        first, later = [(0, 0.5, 72), (1, 0.5, 70), (2, 0.5, 69)], [(0, 0.5, 72), (1, 0.5, 75)]   # ends on A; begin on C

        def quoting():   # one bar each, as they are here, whatever the chords
            phrases = iter([first])
            return unittest.mock.patch('music_transformer.licks.quote',
                                       lambda *_: ('someone · a tune', next(phrases, later), 1))

        with quoting():
            session, notes = self.solo(2.5, licks=1, pickup=1, vary=0)   # a bar is 2 s, a beat 0.5 s
        # over Cm7, after the A on three: Bb on four, B on its "and", swung, and the C on one
        way = [(onset, pitch) for onset, _, pitch, hand in notes if hand == 'right' and pitch >= 70 and 1.2 < onset < 1.95]
        self.assertEqual([pitch for _, pitch in way], [70, 71], way)
        self.assertTrue(abs(way[0][0] - 1.5) < 0.03 and abs(way[1][0] - 1.5 - 0.5 * 2 / 3) < 0.03, way)
        self.assertTrue(any(abs(onset - 2.0) < 0.03 and pitch == 72 for onset, _, pitch, _ in notes))
        with quoting():
            session, _ = self.solo(2.5, licks=1, pickup=0, vary=0)   # at 0, the phrases as they were played
        self.assertEqual([pitch for _, pitch, _ in session.quotes[1]], [72, 75])
        with quoting():
            session, _ = self.solo(0.5, licks=1, pickup=0.5, vary=0)
            led = []
            for bar in range(1, 200):                            # the bars it comes to, one after another
                session._lick(session._tick(session.began + bar * 2.0 + 0.01))
                led.append(len(session.quotes[bar]) > len(later))
        self.assertTrue(0.35 < sum(led) / len(led) < 0.65, sum(led))   # half of the times

    def test_a_phrase_it_quotes_goes_on_from_where_its_line_left_off(self):
        from music_transformer import licks
        lick = [[0, 0.5, 7], [0.5, 0.5, 9], [1, 0.5, 12]]    # from G, over a chord on C
        self.assertEqual(licks.place(lick, 0, 60, 96, 68)[1][0][2], 67)   # after Ab4: G4, a semitone under it
        self.assertEqual(licks.place(lick, 0, 60, 96, 81)[1][0][2], 79)   # after A5: G5
        self.assertEqual(licks.place(lick, 0, 60, 66, 68), None)          # no octave keeps it under F#4
        ii_v = [(0, 'm7')] * 4 + [(5, '7')] * 4
        leaps = [abs(licks.quote(licks.DATA, random.Random(seed), ii_v, 60, 96, near)[1][0][2] - near)
                 for seed in range(40) for near in (65, 77, 89)]
        self.assertLess(sum(leap > 4 for leap in leaps), len(leaps) / 10)   # of those it draws: seldom a leap

    def test_a_phrase_it_quotes_is_fitted_to_the_scale_of_the_chords_it_sounds_over(self):
        from music_transformer import licks
        c_major = lambda at: [pc in (0, 2, 4, 5, 7, 9, 11) for pc in range(12)]
        fitted = lambda *notes: [pitch for *_, pitch in licks.fit(list(notes), c_major)]
        # a Db on the beat, as over an altered dominant, moves on the way the line was going; off the beat it stays
        self.assertEqual(fitted((0, 0.5, 71), (0.5, 0.5, 72), (1, 0.5, 73), (1.5, 0.5, 76)), [71, 72, 74, 76])
        self.assertEqual(fitted((0, 0.5, 76), (0.5, 0.5, 75), (1, 0.5, 73), (1.5, 0.5, 71)), [76, 75, 72, 71])
        self.assertEqual(fitted((0, 0.5, 72), (0.5, 0.5, 73), (1, 0.5, 74)), [72, 73, 74])   # a passing note
        self.assertEqual(fitted((0, 0.5, 76), (1, 0.5, 73), (2, 0.5, 72)), [76, 74, 72])     # not to the note after it
        self.assertEqual(fitted((0.5, 1.5, 73)), [72])                                       # off the beat, but held
        session, _ = self.solo(0.5, licks=1, pickup=0)
        for bar in range(64):                                # the bars it comes to: over the tune, inside
            session._lick(session._tick(session.began + bar * 2.0 + 0.01))
            for at, pitch, _ in session.quotes[bar]:
                on = abs((at - session.began) * 2 - round((at - session.began) * 2)) < 0.05
                self.assertTrue(not on or session.listener.chord_scale(at + 0.01)[pitch % 12], (bar, at, pitch))

    def test_now_and_then_it_plays_the_phrase_it_quoted_last_again_to_develop_it(self):
        from music_transformer import licks
        lick = [[0, 0.5, 0], [1, 0.5, 4], [2, 0.5, 7], [3, 0.5, 12]]
        ways = {tuple((at, pitch) for at, _, pitch in licks.vary(lick, 1, random.Random(seed))) for seed in range(40)}
        self.assertEqual(ways, {((0, 0), (1, 4), (2, 7), (3, 12)),       # as it was: over other chords, a sequence
                                ((1, 0), (2, 4), (3, 7)),                # a beat later, what its bar still holds
                                ((0, 0), (1, 4), (2, 7)),                # its first half, three notes at least
                                ((0, 0), (1, -4), (2, -7), (3, -12))})   # upside down from its first note
        shares = {}
        for vary in (0, 0.5, 1):
            asked = []
            quote = lambda *_: asked.append(1) or ('someone · a tune', [(0, 0.5, 72), (1.5, 0.5, 75), (2, 0.5, 79)], 1)
            with unittest.mock.patch('music_transformer.licks.quote', quote):
                session, _ = self.solo(0.5, licks=1, pickup=0, vary=vary)
                for bar in range(1, 200):                    # the bars it comes to, one after another
                    session._lick(session._tick(session.began + bar * 2.0 + 0.01))
            shares[vary] = 1 - len(asked) / 200
            self.assertTrue(all(len(session.quotes[bar]) >= 3 for bar in range(196, 200)))
        self.assertEqual((shares[0], shares[1]), (0, 1 - 1 / 200))     # none; all but the first
        self.assertTrue(0.4 < shares[0.5] < 0.6, shares)

    def test_it_develops_your_last_phrase_over_the_chords_that_follow_it(self):
        from music_transformer import licks
        self.enterContext(unittest.mock.patch.dict(licks.DATA, {'licks': {}}))   # no soloist to quote
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=2,
                             licks=1, vary=1, recall=0.01, pickup=0)
        for k, pitch in enumerate((72, 75, 79, 77)):         # over Cm7, the first bar
            session.hear(0.5 * k, 0.3, pitch)
        self.clock.t = 3.5                                   # F7 is under way: your phrase has ended
        session._lick(session._tick(4.01))                   # the third bar, Bbmaj7: nobody played over it
        self.assertEqual(session.motif[0], 'you')
        self.assertEqual(session.quoting[0], 'you')
        self.assertIn(len(session.quotes[2]), (3, 4))        # yours, whole or in part, on Bb
        session.turn(recall=0)
        session.motif = None
        session.hear(8.0, 0.3, 72), session.hear(8.5, 0.3, 75), session.hear(9.0, 0.3, 79)
        self.clock.t = 11.5
        session._lick(session._tick(12.01))
        self.assertEqual(session.motif, None)                # it recalls nothing of you: the soloists' only

    def test_its_solo_builds_over_a_chorus_to_its_height_three_quarters_through(self):
        from music_transformer import chart
        session = self.session                               # Autumn Leaves: 32 bars of 2 s
        session.turn(part=2, arc=1)
        self.assertEqual([round(chart.build(session, at), 2) for at in (0, 24, 48, 56, 63.99)], [0, 0.5, 1, 0.5, 0])
        feels = lambda at: [chart.feel(session, at) for session.varied in range(300)]
        busy = lambda at: sum(feel in (4, 8) for feel in feels(at))
        self.assertGreater(feels(0).count(2), 250)           # it begins in eighths
        self.assertGreater(busy(48), 200)                    # at its height: sixteenths, or faster
        self.assertLess(busy(0), 10)
        self.assertEqual([session._hands(False, True, at)[0][3] for at in (0, 48)], [(60, 84), (72, 96)])   # and higher
        session.turn(arc=0)                                  # the same all through
        self.assertTrue(abs(busy(0) - busy(48)) < 50 and busy(0) > 50)
        self.assertEqual([session._hands(False, True, at)[0][3] for at in (0, 48)], [(60, 96), (60, 96)])
        session.turn(chart=self.tune('blues_in_f'), playing=1, arc=1)   # twelve bars: it builds over three choruses
        bar = 4 * 60 / session.pace
        self.assertEqual(round(chart.build(session, session.began + 27 * bar), 2), 1)

    def test_a_phrase_you_play_over_a_ii_v_comes_back_at_another_as_much_as_it_recalls_you(self):
        from music_transformer import licks
        self.enterContext(unittest.mock.patch.dict(licks.DATA, {'licks': {}}))   # no soloist to quote: you only
        tune = self.tune('all_the_things_you_are')           # at 240 bpm a bar is 1 s
        first = list(zip((1.5, 1.75, 2.0, 2.25, 2.5, 2.75), (65, 68, 72, 70, 67, 63)))        # over Bbm7 | Eb7
        second = list(zip((16.5, 16.75, 17.0, 17.25, 17.5, 17.75), (76, 72, 69, 66, 64, 62)))  # over Am7 | D7
        for part, recall in ((0, duo.EXACT), (2, duo.EXACT), (2, 0)):   # it comps, it solos
            clock = Clock()
            session = DuoSession(UniformModel(), 'cpu', clock=clock, chart=tune, playing=1, bpm=240, part=part, licks=1,
                                 recall=recall, echo=6, vary=0, max_rest=50)   # (no long rest: it is there as a phrase begins)
            yours, notes, whose = first + second, [], set()
            while clock.t < 27.9:
                clock.t += 0.1
                session.begin()
                while yours and yours[0][0] <= clock.t:
                    session.hear(yours[0][0], 0.2, yours.pop(0)[1])
                while session.ahead() < duo.LOOKAHEAD:
                    notes.append(session.play())
                whose.add(session.state()['chart'].get('quote'))
            back = lambda phrase, later, up: any(all(any(
                note and abs(note[0] - at - later) < 0.08 and note[2] == pitch + shift for note in notes)
                for at, pitch in phrase) for shift in (up, up + 12))
            # eight bars on, over Fm7 | Bb7, the first a fifth up, note for note; and at the next Bbm7 | Eb7 the one
            # you played last over a ii-V, a semitone up: its `echo` is at the top. The piano roll says whose it is
            self.assertEqual((back(first, 8, 7), back(second, 9, 1), 'you' in whose), (recall > 0,) * 3, (part, recall))

    def test_over_a_tune_its_eighths_are_swung(self):
        session = DuoSession(UniformModel(), 'cpu', clock=Clock(), chart=self.tune('autumn_leaves'), playing=1, part=0,
                             breath=0, min_step=5)
        session.hear(0.0, 0.3, 60)
        onsets = sorted({session.play()[0] for _ in range(40)})
        places = {round(o % 0.5 / 0.5 * 3) % 3 for o in onsets}   # which third of the beat (a beat is 0.5 s)
        self.assertEqual(places, {0, 2})                           # on the beat and on its last third, never its middle

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
        self.assertLessEqual({hand for *_, hand in notes}, {'bass', 'right'})

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

    def soloist(self, **knobs):
        """A session that solos alone over Autumn Leaves, its own lines only, wanting the earliest note it may play."""
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1, part=2,
                             licks=0, **knobs)
        session.begin()
        return session

    def test_it_takes_a_breath_when_its_phrase_has_run_out_of_notes(self):
        session = self.soloist(phrase=4, breath=120)
        onsets = sorted({session.play()[0] for _ in range(30)})
        gaps = [b - a for a, b in zip(onsets, onsets[1:])]
        self.assertGreaterEqual(max(gaps), 1.2)
        self.assertGreater(sum(gap >= 1.2 for gap in gaps), 2)

    def test_it_samples_nothing_while_it_breathes(self):
        session = self.soloist(phrase=2, breath=400)
        while session._breath() == 0:
            session.play()
        self.assertGreaterEqual(session.ahead(), 400 - session.now())

    def test_you_can_come_back_after_a_silence_longer_than_the_context_lasts(self):
        self.session.hear(0.0, 0.3, 60)
        list(self.fill())
        self.clock.t = 500.0
        self.session.hear(500.0, 0.3, 62)
        self.assertLess(self.session.seq[-3] - TIME_OFFSET, 10000)
        self.assertGreater(len(list(self.fill())), 0)

    def planned(self, *pitches):
        """A session over Cm7, the first bar of Autumn Leaves, with its plan after 0.3 s: `pitches`, 0.1 s apart."""
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1,
                             commit=100)
        session.hear(0.0, 0.2, 60)
        session.seq += [token for i, pitch in enumerate(pitches) for token in
                        (TIME_OFFSET + 30 + 10 * i, DUR_OFFSET + 10, NOTE_OFFSET + 4 * MAX_PITCH + pitch)]
        session.spoken = [0.3 + i / 10 for i in range(len(pitches))]
        return session

    def test_its_plan_stands_when_your_note_does_not_contradict_it(self):
        session = self.planned(72, 74, 75)
        self.clock.t = 0.05
        self.assertIsNone(session.hear(0.05, 0.2, 63))
        self.assertEqual(len(session._events()), 5)

    def test_your_note_drops_its_plan_from_the_note_that_contradicts_it(self):
        session = self.planned(72, 73, 75)     # Db over Cm7
        self.clock.t = 0.05
        self.assertAlmostEqual(session.hear(0.05, 0.2, 63), 0.4)
        self.assertEqual(session.spoken, [0.3])
        self.assertEqual(len(session._events()), 3)

    def test_its_plan_does_not_stand_past_commit(self):
        session = self.planned(72, 74, 75)
        session.turn(commit=40)
        self.clock.t = 0.05
        self.assertAlmostEqual(session.hear(0.05, 0.2, 63), 0.5)

    def test_over_a_chart_it_plays_the_chord_of_the_bar_at_the_charts_tempo(self):
        from music_transformer.chart import CHARTS
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'autumn_leaves')
        self.session.turn(chart=tune + 1, playing=1)
        self.assertEqual((self.session.pace, self.session.state()['chord']), (120, 'Cm7 > F7'))
        self.clock.t = 10.2                          # bar 6 of 2 s each: D7, whose F# is not in G minor
        self.session.hear(10.2, 0.3, 62)
        self.assertEqual(self.session.state()['key'], 'Autumn Leaves 6/32')
        bias = self.session._partner(self.session.now() + 30, [], (0, 127), True)
        self.assertGreater(bias[66], bias[65] + 1)

    def test_soloing_over_a_tune_it_approaches_a_chord_tone_by_a_semitone_and_lands_on_it(self):
        self.session.turn(chart=self.tune('blues_in_f'), playing=1, part=2, apart=0, lead=0)   # F7 in the first bar
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

    def test_soloing_as_bebop_players_do_its_line_moves_by_step_and_hears_the_next_chord_coming(self):
        from music_transformer.chart import CHARTS
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'blues_in_f')   # 110 bpm: F7, then Bb7
        self.session.turn(chart=tune + 1, playing=1, part=2, apart=0, lead=0)
        self.session.hear(0.0, 0.3, 53)
        self.enterContext(unittest.mock.patch('music_transformer.chart.way', return_value='bebop'))
        line = [(20, NOTE_OFFSET + 4 * MAX_PITCH + 72)]             # it has just played C5
        bias = self.session._partner(self.session._tick(0.4), line, (0, 127), True)
        self.assertGreater(min(bias[70], bias[74]), max(bias[72], bias[84], bias[60]))   # a step, not C again or a leap
        self.assertGreater(bias[70], -4)                            # Bb, F7's fourth: bebop plays it, the scales held it back
        free = self.session._partner(self.session._tick(0.4), [], (0, 127), True)
        late = self.session._partner(self.session._tick(2.1), [], (0, 127), True)   # Bb7 is a tenth of a second away
        self.assertGreater(late[68] - late[69], free[68] - free[69])   # Ab, its seventh, gains on A

    def test_soloing_over_a_tune_it_runs_in_triplets_or_sixteenths_for_a_phrase(self):
        from music_transformer.chart import CHARTS
        tune = next(i for i, chart in enumerate(CHARTS) if chart.name == 'autumn_leaves')   # 120 bpm: a beat is 0.5 s
        for division, step in ((3, 1 / 6), (4, 0.125)):
            session = DuoSession(UniformModel(), 'cpu', clock=Clock(), chart=tune + 1, playing=1, part=2, breath=0, licks=0)
            session.hear(0.0, 0.3, 60)
            with unittest.mock.patch('music_transformer.chart.feel', return_value=division), \
                    unittest.mock.patch('music_transformer.chart.way', return_value='bebop'):
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
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1)
        session.hear(0.0, 0.3, 60)
        notes = [session.play() for _ in range(40)]
        for hand, most, step in (('right', duo.CHORD, duo.MIN_STEP), ('bass', 1, 40)):   # its chords, its walking bass
            onsets = {}
            for onset, _, pitch, played in notes:
                if played == hand:
                    onsets.setdefault(onset, []).append(pitch)
            for pitches in onsets.values():
                self.assertEqual(len(pitches), len(set(pitches)))
                self.assertLessEqual(len(pitches), most)
            times = sorted(onsets)
            self.assertGreater(len(times), 1)
            self.assertGreaterEqual(min(b - a for a, b in zip(times, times[1:])), step / 100 - 1e-9)
            pitches = [pitch for _, _, pitch, played in notes if played == hand]
            runs = [tuple(pitches[i:i + duo.NO_REPEAT]) for i in range(len(pitches) - duo.NO_REPEAT + 1)]
            self.assertEqual(len(runs), len(set(runs)))

    def test_your_hands_are_where_you_play_an_octave_apart(self):
        self.assertEqual(hands([52, 55, 59, 62, 76, 79, 83]), [(52, 62), (76, 83)])
        self.assertEqual(hands([60, 71]), [(60, 71)])   # a seventh apart: one hand, the keys between too
        self.assertEqual(hands([60, 72]), [(60, 60), (72, 72)])   # an octave: room between
        self.assertEqual(hands([]), [])

    def test_a_knob_turned_while_it_plays_changes_its_next_notes(self):
        session = DuoSession(StuckModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), playing=1)
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
        pitches = [pitch for _, _, pitch, hand in (self.session.play() for _ in range(60)) if hand == 'right']
        self.assertTrue(pitches and all(60 <= p <= 83 for p in pitches), pitches)   # its bass keeps its own register
        self.assertEqual(self.session.octaves(), (60, 83))

    def test_it_says_what_it_does(self):
        self.assertEqual(self.session.state()['doing'], 'listening')
        for i, pitch in enumerate((60, 64, 67, 72)):
            self.clock.t = i * 0.25
            self.session.hear(self.clock.t, 0.2, pitch)
        self.assertEqual(self.session.state()['doing'], 'accompanying')
        self.clock.t = 2.0
        self.assertEqual(self.session.state()['doing'], 'answering')
        self.clock.t = 10.0
        self.assertEqual(self.session.state()['doing'], 'alone')

    def test_it_colours_the_chord_of_the_bar_with_its_tensions(self):
        self.session.hear(0.0, 0.3, 60)
        comps = lambda: self.session._partner(30, [], (0, MAX_PITCH - 1), False, True)   # the chords under its solo
        bias = comps()
        d, c_sharp = 62, 61                          # over Cm7: its 9th, and a wrong note
        self.assertGreater(bias[d], bias[c_sharp] + 5)
        self.session.turn(tension=0)
        self.assertAlmostEqual(float(comps()[d]), float(bias[d]) - duo.TENSION, places=5)

    def test_while_you_play_it_keeps_apart_from_your_hands_and_under_your_melody(self):
        self.session.turn(apart=6.0)
        for pitch in (40, 44, 47, 76, 79, 83):       # a left hand and a right hand, held
            self.session.hear(0.0, 30.0, pitch)
        bias = self.session._register(10, True, (36, 96), True)
        self.assertTrue(bool((bias[38:50] == -6).all()) and bool((bias[74:] == -6).all()))   # your hands, and over your melody
        self.assertTrue(bool((bias[50:74] == 0).all()))
        self.assertTrue(bool((self.session._register(10, True, (28, 51), False) == 0).all()))   # its bass walks where it walks
        # your hands over all of its register leave it no room: it plays as it would
        self.assertTrue(bool((self.session._register(10, True, (76, 83), True) == 0).all()))

    def test_while_you_only_comp_it_may_take_the_melody_over_you(self):
        self.session.turn(apart=6.0)
        for pitch in (48, 52, 55, 59):               # your left hand alone, held
            self.session.hear(0.0, 30.0, pitch)
        bias = self.session._register(10, True, (36, 96), True)
        self.assertTrue(bool((bias[46:62] == -6).all()) and bool((bias[62:] == 0).all()))

    def test_when_you_rest_its_line_sings_up_around_your_right_hand(self):
        for i, pitch in enumerate((43, 47, 50, 74, 76, 77, 79, 81)):
            self.session.hear(i * 0.1, 0.3, pitch)
        self.session.turn(lead=6.0)
        bias = self.session._register(300, False, (60, 96), True)   # you rest
        self.assertEqual((float(bias[65]), float(bias[77])), (-6.0, 0.0))   # an octave under your melody's middle; there
        self.assertLess(float(bias[65]), float(bias[71]))
        self.assertTrue(bool((self.session._register(300, False, (28, 51), False) == 0).all()))   # its bass stays down

    def test_there_is_always_a_tune_and_it_waits_for_play(self):
        from music_transformer.chart import CHARTS
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock)
        self.assertEqual(session.state()['chart']['title'], CHARTS[0].title)   # the first, until you choose another
        self.assertEqual((session.state()['chart']['bar'], session.state()['doing']), (-1, 'listening'))
        session.hear(0.0, 0.3, 60)
        self.assertGreaterEqual(session.ahead(), duo.LOOKAHEAD)   # stopped: nothing to sample
        session.turn(chart=0)
        self.assertEqual(session.chart, 1)                         # no tune is no choice
        self.assertEqual(set(duo.KNOBS) & {'timeout', 'hands', 'scale', 'root', 'silence', 'space'}, set())

    def test_knobs_given_at_the_start_hold_from_the_first_note(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, self_window=60.4, stay=0.9)
        self.assertEqual(session.self_window, 60)
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
        self.assertLessEqual(sum(t < self.session.heard_at - duo.SELF_WINDOW for t in mine), duo.PRUNE)
        self.session.play()
        self.assertEqual(self.model.fed, len(self.session.seq) - 1)

    def test_it_answers_your_phrase_in_a_run_and_sequences_your_intervals_past_its_end(self):
        self.session.turn(part=2, licks=0, run=10, min_step=5, max_rest=600, echo=6.0, answer=1000)   # it solos
        for i, pitch in enumerate((60, 62, 64)):
            self.session.hear(i * 0.25, 0.2, pitch)
        self.clock.t = 1.5   # your phrase has ended: it answers
        notes = []
        while len(notes) < 12:
            notes += self.fill()
            self.clock.t += 0.5
        onsets = [onset for onset, *_ in notes]
        self.assertLessEqual(max(b - a for a, b in zip(onsets, onsets[1:])), 0.1 + 1e-9)

        def echoed(*pitches):   # the pitch its `echo` adds to, once it has answered with `pitches`
            line = [(200 + 10 * i, NOTE_OFFSET + 4 * MAX_PITCH + pitch) for i, pitch in enumerate(pitches)]
            biases = []
            for echo in (6.0, 0.0):
                self.session.turn(echo=echo)
                biases.append(self.session._partner(260, line, (60, 96), True))
            return int((biases[0] - biases[1]).argmax())
        self.assertEqual(echoed(67), 69)             # your step up, from its G
        self.assertEqual(echoed(67, 69), 72)         # and again: B is not in G minor, the tune's key: bent to C

if __name__ == '__main__':
    unittest.main()
