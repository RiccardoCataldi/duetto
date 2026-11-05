"""
The jam_bot's ways of playing together (Blanchard, Naseck et al., "The jam_bot, a Real-Time System for Collaborative
Free Improvisation with Music Language Models", ISMIR 2025), with AMT as it was pretrained: the model starts again at
chosen moments from only the last few notes, and is left free to play on from them, without the duo's rules.

    trade    at regular intervals: you and it take turns of `bars` bars; in its turn it carries on from the last notes
             of you both as one piano, in yours it listens
    gesture  at each of your gestures: `wait` after your last note it drops what it has not played yet and starts
             again from the last notes; with `lock` your last gesture is held ahead of it over and over, as AMT's
             anticipated controls, and it follows you strictly
    request  when you cue it: a key under `trigger` makes it start again from your last notes; in between it plays on

It plays with the server as a DuoSession does. Times are session seconds; in the context, ticks from the prompt's
first note.
"""

import time

import torch
from anticipation.config import CONTEXT_SIZE, DELTA, MAX_DUR, MAX_PITCH, MAX_TIME, TIME_RESOLUTION
from anticipation.vocab import (ADUR_OFFSET, ANOTE_OFFSET, ANTICIPATE, ATIME_OFFSET, AUTOREGRESS, DUR_OFFSET,
                                NOTE_OFFSET, TIME_OFFSET)
from anticipation.sample import safe_logits, nucleus

from music_transformer.duo import KNOBS, LOOKAHEAD, MARGIN, NEVER, PLAYING, RIGHT_BOTTOM, _duration, knob
from music_transformer.listener import Listener

STRATEGIES = ('duo', 'trade', 'gesture', 'request')   # the strategy knob's choices; the duo's is DuoSession
LEAD_S = 0.25      # seconds before its turn it hears the last notes and starts, to sound on time
BEAT_S = 0.5       # a beat while your tempo is unclear (120 bpm)
BAR = 4            # beats in a bar
GESTURE_S = 0.15   # your notes this close before your last one are one gesture with it: a chord
OLDEST_S = 60      # seconds back a prompt reaches at most: AMT's times end at 100 s
IDLE = 1000        # ticks it waits while it has nothing to play; your next note wakes it sooner
# the knobs that shape its notes, in each strategy
ALWAYS = {'strategy', 'context', 'temperature', 'top_p', 'lowest', 'highest', 'max_rest', 'bpm'}
IN_EFFECT = {'trade': {'bars'}, 'gesture': {'wait', 'lock', 'timeout'}, 'request': {'trigger', 'timeout'}}


class JamSession:
    def __init__(self, model, device, you=0, me=4, clock=time.monotonic, started=None, **knobs):
        self.model, self.device = model, device
        self.you, self.me = you, me    # General MIDI programs: piano, and its own in the context
        self.listener = Listener()
        for name, (default, *_) in KNOBS.items():
            setattr(self, name, default)
        self.turn(**knobs)
        self.clock, self.started = clock, clock() if started is None else started
        self.base = None               # not None once you have played: the server waits for you until then
        self.notes = []                # (onset, duration, pitch, its) of you both
        self.heard = None              # when you last played
        self.due = None                # when it starts again from your last notes
        self.cut = None                # from when its notes already sent are dropped, for the server to say
        self.seq = None                # the prompt and what it has played since; None while it is silent
        self.past, self.fed = None, 0  # the KV cache covers seq[:fed]
        self.origin = 0.0              # when the prompt's time 0 is
        self.last, self.since = 0.0, 0 # its last onset, and its notes since the prompt
        self.gesture, self.held = [], 0.0   # lock: your last gesture's pitches, and when it is next held ahead
        self.turns, self.prompted = None, False   # trade: (start, end) of its next or present turn; begun

    @property
    def way(self):
        return STRATEGIES[self.strategy]

    def turn(self, **knobs):
        """Sets knobs (see KNOBS), kept within their range; they take effect from the model's next note."""
        for name, value in knobs.items():
            setattr(self, name, knob(name, value))
        if 'stay' in knobs:
            self.listener.stay(self.stay)
        if 'bpm' in knobs:
            self.listener.fix(self.bpm)

    def state(self):
        """What it hears and what it is doing, as DuoSession.state()."""
        now = self._now()
        if self.way == 'trade':
            mode = ('listening' if self.turns is None else
                    'its turn' if self.turns[0] <= now < self.turns[1] else 'your turn')
        elif self.seq is None or self._timed_out(now):
            mode = 'listening'
        else:
            mode = 'accompanying' if now - self.heard < PLAYING / TIME_RESOLUTION else 'alone'
        key, pulse = self.listener.said_key(), self.listener.pulse(now)
        active = ALWAYS | IN_EFFECT[self.way] | ({'every'} if self.lock and self.way == 'gesture' else set())
        return {'mode': mode, 'key': self.listener.name(key) if key is not None else None,
                'chord': self.listener.chord_label(now) if self.listener.chord(now) else None,
                'bpm': round(60 / self.listener.beat(pulse[0])) if pulse else None,
                'active': sorted(active | ({'metronome'} if self.bpm else set()))}

    def ahead(self):
        """Ticks of music already sampled beyond now; the caller samples more while this is under LOOKAHEAD, and
        otherwise waits as long as it is over."""
        now = self._now()
        if self.due is not None:
            return 0 if now >= self.due else LOOKAHEAD + _ticks(self.due - now)
        if self.way == 'trade':
            self._next_turn(now)
            if self.turns is None:
                return LOOKAHEAD + IDLE
            start, end = self.turns
            if not self.prompted:
                return 0 if now >= start - LEAD_S else LOOKAHEAD + _ticks(start - LEAD_S - now)
            if self.seq is None:
                return LOOKAHEAD + max(_ticks(end - now), 1)
        elif self.seq is None or self._timed_out(now):
            return LOOKAHEAD + IDLE
        return _ticks(self.last - now)

    def hear(self, onset, duration, pitch, velocity=None):
        """Your note, pressed at `onset`. It starts again from it when its strategy says; the server learns what of
        its own it drops from recut()."""
        self.listener.hear(onset, duration, pitch)
        self.notes.append((onset, duration, pitch, False))
        self.base, self.heard = 0, onset
        if self.way == 'trade' and self.turns is None:
            bar = BAR * self._beat(onset)
            start = (round(onset / bar) * bar if self.bpm else onset) + self.bars * bar   # on the metronome's bars
            self.turns, self.prompted = (start, start + self.bars * bar), False
        elif self.way == 'gesture':
            self.due = onset + self.wait / TIME_RESOLUTION
        elif self.way == 'request' and pitch < self.trigger:
            self.due = onset
        return None

    def release(self, onset, pitch, duration):
        """Your note pressed at `onset` stopped sounding after `duration`: its next prompt hears how long it was."""
        self.listener.release(onset, pitch, duration)
        self.notes = [(o, duration, p, its) if (o, p, its) == (onset, pitch, False) else (o, d, p, its)
                      for o, d, p, its in self.notes]

    def recut(self):
        """The time from which its notes already sent are dropped, once; None if none are."""
        cut, self.cut = self.cut, None
        return cut

    @torch.no_grad()
    def play(self):
        """Samples the model's next note: (onset, duration, pitch, hand) in session seconds; or None when it stays
        silent."""
        now = self._now()
        soonest = now + MARGIN / TIME_RESOLUTION
        if self.due is not None and now >= self.due:
            self.due = None
            self._again(soonest, soonest, yours=self.way == 'request')
        elif self.way == 'trade':
            self._next_turn(now)
            if self.turns and not self.prompted and now >= self.turns[0] - LEAD_S:
                self.prompted = True
                self._again(soonest, max(self.turns[0], soonest))
        if self.seq is None or (self.way != 'trade' and self._timed_out(now)):
            return None
        if (self.since >= self.context or len(self.seq) + 3 * (1 + 4 * len(self.gesture)) > CONTEXT_SIZE
                or self._at(now) + LOOKAHEAD + self.max_rest >= MAX_TIME):
            self._again(self.last + 1 / TIME_RESOLUTION, self.last, cut=False)   # its context stays short
        earliest = max(self._at(self.last), self._at(soonest))
        latest = earliest + self.max_rest
        if self.way == 'trade':
            latest = min(latest, self._at(self.turns[1]))
            if earliest >= latest:
                self.seq = None   # its turn is over
                return None
        self._anticipate(earliest)
        voice = self.you if self.way == 'trade' else self.me
        low, high = sorted((self.lowest, self.highest))
        low, high = 12 * (low + 1), min(12 * (high + 1) + 11, MAX_PITCH - 1)
        for slot in range(3):
            logits = self._logits()
            mask = torch.full_like(logits, -float('inf'))
            if slot == 0:
                mask[TIME_OFFSET + earliest:TIME_OFFSET + latest + 1] = 0
            elif slot == 1:
                mask[DUR_OFFSET + 1:DUR_OFFSET + MAX_DUR] = 0
            else:
                mask[NOTE_OFFSET + voice * MAX_PITCH + low:NOTE_OFFSET + voice * MAX_PITCH + high + 1] = 0
            token = int(torch.multinomial(torch.softmax(nucleus(logits + mask, self.top_p), -1), 1))
            self.seq.append(token)
            if slot == 0 and self.way == 'trade' and token - TIME_OFFSET >= self._at(self.turns[1]):
                self.seq = None   # it ends its turn here
                return None
        time_, duration, note = self.seq[-3:]
        onset, pitch = self.origin + (time_ - TIME_OFFSET) / TIME_RESOLUTION, (note - NOTE_OFFSET) % MAX_PITCH
        duration = (duration - DUR_OFFSET) / TIME_RESOLUTION
        self.notes.append((onset, duration, pitch, True))
        self.last, self.since = onset, self.since + 1
        return onset, duration, pitch, 'left' if pitch < RIGHT_BOTTOM else 'right'

    def _again(self, upto, start, cut=True, yours=False):
        """A new prompt from the last `context` notes before `upto` (with `yours`, of yours only), to play on from
        `start`; unless `cut` is False, what it has sampled from `upto` on is dropped."""
        if cut and any(its and onset >= upto for onset, _, _, its in self.notes):
            self.cut = upto
            self.notes = [n for n in self.notes if not (n[3] and n[0] >= upto)]
        notes = sorted(n for n in self.notes if upto - OLDEST_S <= n[0] < upto and not (yours and n[3]))
        notes = notes[-self.context:]
        self.origin = notes[0][0] if notes else start
        locked = self.lock and self.way == 'gesture'
        self.seq = [ANTICIPATE if locked else AUTOREGRESS]
        for onset, duration, pitch, its in notes:
            voice = self.me if its and self.way != 'trade' else self.you
            self.seq += [TIME_OFFSET + self._at(onset), DUR_OFFSET + _duration(duration),
                         NOTE_OFFSET + voice * MAX_PITCH + pitch]
        if cut:
            heard = [n for n in self.notes if not n[3]]
            last = heard[-1][0] if heard else start
            self.gesture = sorted({p for o, _, p, its in heard if o >= last - GESTURE_S}) if locked else []
        self.held = start              # held ahead again from here: the controls of the old prompt are gone
        self.past, self.fed = None, 0
        self.last, self.since = start, 0

    def _anticipate(self, tick):
        """Lock: your last gesture, held again every `every` ticks from where it starts again, joins the context as
        AMT's anticipated controls, DELTA seconds ahead of their time."""
        while self.gesture and self._at(self.held) - DELTA * TIME_RESOLUTION <= tick:
            for pitch in self.gesture:
                self.seq += [ATIME_OFFSET + self._at(self.held), ADUR_OFFSET + self.every,
                             ANOTE_OFFSET + self.you * MAX_PITCH + pitch]
            self.held += self.every / TIME_RESOLUTION

    def _next_turn(self, now):
        """Trade: once its turn is over, its next after yours; none if you have not played since your last began."""
        if self.turns is None or now < self.turns[1]:
            return
        start, end = self.turns
        if self.heard is None or self.heard < start - (end - start):
            self.turns, self.seq = None, None
            return
        length = self.bars * BAR * self._beat(now)
        self.turns, self.prompted, self.seq = (end + length, end + 2 * length), False, None

    def _beat(self, now):
        pulse = self.listener.pulse(now)
        return self.listener.beat(pulse[0]) if pulse else BEAT_S

    def _timed_out(self, now):
        """Whether it has played on long enough after your last note."""
        return self.timeout < NEVER and self.heard is not None and now - self.heard >= self.timeout / TIME_RESOLUTION

    def _now(self):
        return self.clock() - self.started

    def _at(self, seconds):
        """`seconds` as a time in the context."""
        return _ticks(seconds - self.origin)

    def _logits(self):
        inp = torch.tensor([self.seq[self.fed:]], device=self.device)
        out = self.model(inp, past_key_values=self.past, use_cache=True)
        self.past, self.fed = out.past_key_values, len(self.seq)
        return safe_logits(out.logits[0, -1].float() / self.temperature, len(self.seq) - 1)


def _ticks(seconds):
    return round(seconds * TIME_RESOLUTION)
