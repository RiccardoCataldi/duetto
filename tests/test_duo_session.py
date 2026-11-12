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
        self.session = DuoSession(self.model, 'cpu', clock=self.clock)
        self.enterContext(unittest.mock.patch('music_transformer.chart.COUNT', 0))   # charts begin at once here

    def tune(self, name):
        from music_transformer.chart import CHARTS
        return next(i for i, chart in enumerate(CHARTS) if chart.name == name) + 1

    def test_a_tune_begins_after_a_bar_of_count_in_and_it_starts_by_itself_when_it_comps(self):
        with unittest.mock.patch('music_transformer.chart.COUNT', 1):
            self.clock.t = 0.5
            self.session.turn(chart=self.tune('autumn_leaves'), part=0)   # 120 bpm: a bar is 2 s
        self.assertEqual(self.session.began, 4.0)           # the next bar, and one of count-in
        self.assertEqual(self.session.state()['chart']['count'], 8)
        self.session.begin()
        self.assertIsNone(self.session.base)                # not yet
        self.clock.t = 3.2
        self.session.begin()
        notes = list(self.fill())
        self.assertGreater(len(notes), 0)
        self.assertGreaterEqual(min(n[0] for n in notes), 4.0 - 0.03)   # none in the count-in
        for _ in range(60):                                 # it plays on by itself, its memory of you still empty
            self.clock.t += 0.1
            list(self.fill())
        self.session.hear(self.clock.t, 0.3, 60)            # and you join when you like
        self.assertGreaterEqual(self.session.heard_at, 0)
        duet = DuoSession(self.model, 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'))
        duet.begin()
        self.assertIsNone(duet.base)                        # in a duet it waits for you

    def test_trading_fours_you_solo_four_bars_then_it_does(self):
        self.session.turn(chart=self.tune('blues_in_f'), part=3)
        bar = 4 * 60 / 110
        self.assertEqual([self.session._solos(k * bar + 0.1, True) for k in (0, 3, 4, 7, 8, 12, 15, 16)],
                         [False, False, True, True, False, True, True, False])   # on through the 12 bars' end
        self.session.hear(0.0, 0.3, 53)
        self.clock.t = 4.5 * bar
        self.assertEqual(self.session.state()['mode'], 'its turn')

    def test_over_a_tune_its_bass_walks_root_first_and_approaches_the_next_chord(self):
        self.session.turn(chart=self.tune('blues_in_f'), part=0)   # F7 | Bb7: a beat is 0.545 s
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

    def test_over_a_tune_its_bass_plays_every_beat_while_its_right_hand_plays_and_breathes(self):
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), part=0,
                             phrase=4, breath=200)           # 120 bpm: a beat is 0.5 s
        session.hear(0.0, 0.3, 60)
        notes = []
        for _ in range(80):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        bass = sorted({round(n[0] * 2) for n in notes if n[3] == 'left'})
        self.assertEqual(bass, list(range(bass[0], bass[-1] + 1)))   # no beat without its bass note
        self.assertGreater(len(bass), 12)
        self.assertTrue(all(abs(n[0] * 2 - round(n[0] * 2)) < 0.07 for n in notes if n[3] == 'left'))
        self.assertGreater(sum(n[3] == 'right' for n in notes), 5)

    def test_over_a_tune_its_right_hand_plays_on_though_the_model_would_rather_walk_its_bass_alone(self):
        class Bassist(UniformModel):
            def __call__(self, inp, past_key_values=None, use_cache=True):
                out = super().__call__(inp, past_key_values, use_cache)
                out.logits[..., duo.TIME_OFFSET:duo.TIME_OFFSET + duo.MAX_TIME] += torch.arange(duo.MAX_TIME)   # on the
                out.logits[..., NOTE_OFFSET + duo.LEFT * MAX_PITCH:NOTE_OFFSET + (duo.LEFT + 1) * MAX_PITCH] += 30   # beat, the bass
                return out

        for part, heard in ((0, False), (1, True), (1, False)):   # it comps; a duet, you playing or resting
            session = DuoSession(Bassist(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), part=part,
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
            self.assertGreater(sum(bool(n) and n[3] == 'left' for n in notes), 30, part)   # and its bass walks on
            self.clock.t = start

    def test_soloing_over_its_walking_bass_it_comps_a_chord_now_and_then_under_its_line(self):
        from music_transformer import chart
        for knobs in ({'part': 1, 'licks': 0}, {'part': 1, 'licks': 1}, {'part': 1, 'licks': 0.5, 'grid': 0, 'bpm': 200},
                      {'part': 0}, {'part': 2}):
            clock = Clock()
            session = DuoSession(UniformModel(), 'cpu', clock=clock, chart=self.tune('autumn_leaves'), **knobs)
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
            self.assertGreater(sum(n[3] == 'left' for n in notes), 0.8 * 4 * bars)

    def test_over_a_tune_its_part_says_what_its_hands_do_whatever_the_other_knobs(self):
        self.session.turn(chart=self.tune('blues_in_f'), part=0)
        comps = self.session._hands(False, self.session._solos(1.0, False))
        self.assertEqual([(most, register) for _, _, most, register in comps], [(3, (53, 76)), (1, (28, 51))])
        self.session.turn(part=2)
        solos = self.session._hands(False, self.session._solos(1.0, False))
        self.assertEqual([(most, register) for _, _, most, register in solos], [(1, (60, 96))])   # a line, no bass
        self.session.turn(part=0, hands=1, chord=1, solo_chord=1)   # as "it solos" leaves them: its bass walks all the same
        self.assertEqual([(most, register) for _, _, most, register in self.session._hands(False, False)],
                         [(3, (53, 76)), (1, (28, 51))])
        self.session.turn(part=1)                    # a duet: chords under you, a line when you rest, the bass on
        self.assertEqual([hand[2] for hand in self.session._hands(True, self.session._solos(1.0, True))], [3, 1])
        self.assertEqual([hand[2] for hand in self.session._hands(False, self.session._solos(1.0, False))], [1, 1])
        self.session.turn(lowest=5, highest=6)       # its octaves still hold: C5 to B6, and no room for a bass there
        self.assertEqual([hand[3] for hand in self.session._hands(False, True)], [(72, 95)])

    def solo(self, seconds, **knobs):
        """(the session, its notes) after it soloed alone over Autumn Leaves (120 bpm: a bar is 2 s) for `seconds`."""
        session = DuoSession(UniformModel(), 'cpu', clock=self.clock, chart=self.tune('autumn_leaves'), part=2, **knobs)
        session.begin()
        notes = []
        for _ in range(round(seconds * 10)):
            self.clock.t += 0.1
            while session.ahead() < duo.LOOKAHEAD:
                notes.append(session.play())
        return session, [note for note in notes if note]

    def test_soloing_it_quotes_a_phrase_played_over_the_same_chords_in_the_key_of_the_tune(self):
        from music_transformer import licks
        session, notes = self.solo(2.5, licks=1)             # the first bar: Cm7, then F7
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
            session, notes = self.solo(7, licks=1)
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
            session, notes = self.solo(9, licks=1)           # Cm7 | F7 | Bbmaj7 | Ebmaj7: 8 s
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

    def test_a_phrase_you_play_over_a_ii_v_comes_back_at_another_as_much_as_it_recalls_you(self):
        from music_transformer import licks
        self.enterContext(unittest.mock.patch.dict(licks.DATA, {'licks': {}}))   # no soloist to quote: you only
        tune = self.tune('all_the_things_you_are')           # at 240 bpm a bar is 1 s
        first = list(zip((1.5, 1.75, 2.0, 2.25, 2.5, 2.75), (65, 68, 72, 70, 67, 63)))        # over Bbm7 | Eb7
        second = list(zip((16.5, 16.75, 17.0, 17.25, 17.5, 17.75), (76, 72, 69, 66, 64, 62)))  # over Am7 | D7
        for part, recall in ((0, duo.EXACT), (2, duo.EXACT), (2, 0)):   # it comps, it solos
            clock = Clock()
            session = DuoSession(UniformModel(), 'cpu', clock=clock, chart=tune, bpm=240, part=part, licks=1,
                                 recall=recall, echo=6, max_rest=50)   # (no long rest: it is there as a phrase begins)
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
        session = DuoSession(UniformModel(), 'cpu', clock=Clock(), chart=self.tune('autumn_leaves'), part=0, hands=1,
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
        self.session.turn(chart=self.tune('blues_in_f'), part=2, hands=1, apart=0, lead=0)   # F7 in the first bar
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
        self.session.turn(chart=tune + 1, part=2, hands=1, apart=0, lead=0)
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
            session = DuoSession(UniformModel(), 'cpu', clock=Clock(), chart=tune + 1, part=2, hands=1, breath=0, licks=0)
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
