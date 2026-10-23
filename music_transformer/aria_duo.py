"""
The real-time duo with Aria (EleutherAI): a model of expressive solo piano, ~60k hours of transcribed recordings.
https://github.com/EleutherAI/aria

Aria hears one piano: your notes and its own are one performance in its context, four hands on one keyboard. It
plays both its hands and how hard it strikes each key. As with AMT (duo.py), your notes enter the context the moment
you press them and the part of its line that has not sounded yet is thrown away and sampled again.

Aria's tokens: each note is (piano, pitch, velocity), then its onset in ms within a 5 s segment, then its duration;
<T> starts the next segment. Here times are ticks of 10 ms (Aria's step) since `base`; on the wire, seconds since
the connection opened.
"""

import time

import torch
from ariautils.tokenizer import AbsTokenizer

from music_transformer.listener import STAY, Listener

NAME = 'aria'
TOKENIZER = AbsTokenizer()
STEP_MS = TOKENIZER.time_step_ms                         # 10 ms: one tick
SEGMENT = TOKENIZER.abs_time_step_ms // STEP_MS          # ticks in a segment, between two <T>
MAX_DUR = TOKENIZER.max_dur_ms // STEP_MS
VELOCITIES = TOKENIZER.velocity_quantizations[1:]        # 10 to 120; 0 is not a note
NOTE_IDS = torch.tensor([[TOKENIZER.tok_to_id[('piano', p, v)] for v in VELOCITIES] for p in range(128)])
ONSET_IDS = torch.tensor([TOKENIZER.tok_to_id[('onset', t * STEP_MS)] for t in range(SEGMENT)])
DUR_IDS = torch.tensor([TOKENIZER.tok_to_id[('dur', t * STEP_MS)] for t in range(1, MAX_DUR + 1)])
T = TOKENIZER.tok_to_id[TOKENIZER.time_tok]

CONTEXT = 4096    # tokens of the KV cache (Aria was trained on up to 8192)
WINDOW = 3000     # ticks of the duo it keeps hearing when the context fills up
LOOKAHEAD = 100   # ticks it plays ahead of now (1 s)
MARGIN = 15       # ticks: a new note of its sounds no earlier than this, to cover sampling and the network
PLAYING = 100     # ticks after your last note that you still count as playing
CHORD = 3         # ticks within which its onsets make one chord (a pianist's chord is never quite together)
SPLIT = 60        # under middle C a note is its left hand's (for the client's dynamics and the piano roll only)

# the knobs you can turn while you play (the piano roll shows them): default, lowest, highest, unit, what it does.
# Aria is a pianist already: they nudge it less than AMT's, and it needs no rules against loops or doubled notes.
KNOBS = {
    'min_step': (0, 0, 60, 'ticks', "time between its onsets when you rest"),
    'space': (15, 0, 150, 'ticks', "time between its onsets while you play"),
    'chord': (5, 1, 8, 'notes', "most notes it starts at once while you play: its comping"),
    'solo_chord': (2, 1, 8, 'notes', "most notes it starts at once when you leave it room: lines and arpeggios"),
    'out_of_key': (0.5, 0.0, 6.0, '', "how strongly it keeps to your key"),
    'chord_tone': (0.5, 0.0, 4.0, '', "how much it leans on the chord you play"),
    'tension': (0.5, 0.0, 4.0, '', "how much it colours your chord with its 9th, 11th and 13th"),
    'under': (0.5, 0.0, 4.0, '', "how firmly it stays below you while you play"),
    'grid': (20, 0, 20, 'ticks', "how far off your pulse its onsets may land (20: free, as Aria plays)"),
    'stay': (STAY, 0.80, 0.995, '', "how sure it must be you modulated (lower = sooner)"),
    'max_rest': (300, 50, 800, 'ticks', "the longest silence it may leave"),
    'temperature': (0.95, 0.5, 1.5, '', "how adventurous its choices are"),
    'min_p': (0.035, 0.0, 0.3, '', "how unlikely a note may be, against the likeliest, and still be played"),
    'lowest': (0, 0, 8, 'octave_from', "the lowest octave it may play"),
    'highest': (8, 0, 8, 'octave_to', "the highest octave it may play"),
}
IN_EFFECT = {'listening': set(), 'accompanying': {'space', 'under', 'chord'}, 'alone': {'min_step', 'solo_chord'}}
ALWAYS = {'out_of_key', 'chord_tone', 'tension', 'stay', 'max_rest', 'temperature', 'min_p', 'lowest', 'highest'}


def knob(name, value):
    """`value` for the knob `name`, kept within its range."""
    default, low, high, *_ = KNOBS[name]
    value = min(max(value, low), high)
    return round(value) if isinstance(default, int) else float(value)


class AriaModel:
    """Aria on CUDA with its static KV cache: `model(tokens, start)` feeds `tokens` at positions `start`... and gives
    the logits after the last. Feeding at a position overwrites what the cache held from there on."""

    def __init__(self, checkpoint, context=CONTEXT):
        from safetensors.torch import load_file
        from aria.config import load_model_config
        from aria.inference.model_cuda import TransformerLM
        from aria.model import ModelConfig
        config = ModelConfig(**load_model_config(name='medium'))
        config.set_vocab_size(TOKENIZER.vocab_size)
        self.model = TransformerLM(config)
        self.model.load_state_dict(load_file(checkpoint), strict=False)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = self.model.to('cuda', self.dtype).eval()
        self.model.setup_cache(batch_size=1, max_seq_len=context, dtype=self.dtype)
        self.device = 'cuda'

    @torch.inference_mode()
    def __call__(self, tokens, start):
        idxs = torch.tensor([tokens], device=self.device)
        positions = torch.arange(start, start + len(tokens), device=self.device)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return self.model(idxs=idxs, input_pos=positions)[0, -1].float()


class AriaSession:
    def __init__(self, model, genre='jazz', clock=time.monotonic, started=None, **knobs):
        self.model = model
        prefix = ([('prefix', 'genre', genre)] if genre else []) + [('prefix', 'instrument', 'piano')]
        self.prefix = TOKENIZER.encode(prefix + [TOKENIZER.bos_tok])
        self.listener = Listener()
        for name, (default, *_) in KNOBS.items():
            setattr(self, name, default)
        self.turn(**knobs)
        self.heard_at = None           # tick of your last note
        self.clock, self.started = clock, clock() if started is None else started
        self.base = None               # session tick of tick 0: your first note
        self.events = []               # (tick, mine, pitch, velocity, duration), in onset order
        self.seq = list(self.prefix)   # Aria's tokens of the events
        self.segment = 0               # <T> tokens in seq
        self.fed = 0                   # the cache holds seq[:fed]

    def turn(self, **knobs):
        """Sets knobs (see KNOBS), kept within their range; they take effect from its next note."""
        for name, value in knobs.items():
            if name in KNOBS:
                setattr(self, name, knob(name, value))
        if 'stay' in knobs:
            self.listener.stay(self.stay)

    def knobs(self):
        return {name: getattr(self, name) for name in KNOBS}

    def state(self):
        """What it hears and what it is doing: {'mode', 'key', 'chord', 'bpm', 'active': the knobs in effect}."""
        if self.base is None or self.heard_at is None:
            return {'mode': 'listening', 'key': None, 'chord': None, 'bpm': None, 'active': sorted(ALWAYS)}
        now = self.now()
        at = self._seconds(now)
        mode = 'accompanying' if now - self.heard_at < PLAYING else 'alone'
        key, sure = self.listener.key()
        pulse = self.listener.pulse(at)
        active = ALWAYS | IN_EFFECT[mode] | ({'grid'} if pulse else set())
        return {'mode': mode, 'key': self.listener.name(key) if sure > 0.3 else None,
                'chord': self.listener.chord_label(at),
                'bpm': round(60 / self.listener.beat(pulse[0])) if pulse else None, 'active': sorted(active)}

    def octaves(self):
        """The lowest and highest pitch it may play, from C of its lowest octave to B of its highest (C4 = 60)."""
        low, high = sorted((self.lowest, self.highest))
        return 12 * (low + 1), min(12 * (high + 1) + 11, 127)

    def now(self):
        return round((self.clock() - self.started) * 100) - self.base

    def ahead(self):
        """Ticks of music already sampled beyond now; the caller samples more while this is under LOOKAHEAD."""
        return self._last() - self.now()

    def hear(self, onset, duration, pitch, velocity=80):
        """Your note, pressed at `onset`: Aria hears it when you played it, before those of its notes already on their
        way that come after it. Returns the time from which its unplayed notes are dropped, or None."""
        self.listener.hear(onset, duration, pitch)
        tick = round(onset * 100)
        if self.base is None:
            self.base = tick
        self.heard_at = max(tick - self.base, 0)
        horizon, later, cut = self.now() + MARGIN, self._unwind(self.heard_at), None
        self._add((self.heard_at, False, pitch, _velocity(velocity), _duration(duration)))
        for event in later:
            if event[1] and event[0] > horizon:   # its own, not sounded yet: sampled again, now hearing you
                cut = event[0] if cut is None else cut
            else:
                self._add(event)
        return None if cut is None else self._seconds(cut)

    def release(self, onset, pitch, duration):
        """Your note pressed at `onset` stopped sounding after `duration`, the pedal's ring and all (Aria learnt the
        pedal as longer notes): it hears how long it really was from its next note on."""
        self.listener.release(onset, pitch, duration)
        if self.base is None:
            return
        tick, ticks, at = max(round(onset * 100) - self.base, 0), _duration(duration), len(self.prefix)
        for i, (when, mine, played, velocity, lasted) in enumerate(self.events):
            while self.seq[at] == T:   # at: where each event's tokens start
                at += 1
            if (when, mine, played) == (tick, False, pitch):
                if lasted != ticks:
                    self.events[i] = (when, mine, played, velocity, ticks)
                    self.seq[at + 2] = int(DUR_IDS[ticks - 1])
                    self.fed = min(self.fed, at + 2)
                return
            at += 3

    @torch.no_grad()
    def play(self):
        """Samples its next note: (onset, duration, pitch, hand, velocity), times in session seconds."""
        self._fit()
        now, last = self.now(), self._last()
        you_play = self.heard_at is not None and now - self.heard_at < PLAYING
        earliest = max(last, now + MARGIN)
        latest = max(earliest, last + self.max_rest)
        mine = [e for e in self.events if e[1]]
        step, most = (self.space, self.chord) if you_play else (self.min_step, self.solo_chord)
        gap = None
        if mine:
            group = [e for e in mine if e[0] >= mine[-1][0] - CHORD]   # its last chord
            start = group[0][0]
            if len(group) >= most:
                earliest = max(earliest, start + max(step, CHORD + 1))
            elif step > CHORD:
                gap = (start + CHORD + 1, start + step)               # no onset between its chord and step
            latest = max(latest, start + step)
        onsets = self._onsets(earliest, latest, gap, you_play)
        while self.segment < earliest // SEGMENT:                      # time goes by up to the earliest it may play
            self._push([T])
        # its note, or <T>: it rests on into the next segment
        while True:
            here = [t for t in onsets if t // SEGMENT == self.segment]
            later = any(t // SEGMENT > self.segment for t in onsets)
            if not here:
                self._push([T])
                continue
            logits = self._logits()
            mask = torch.full_like(logits, -float('inf'))
            low, high = self.octaves()
            notes = NOTE_IDS[low:high + 1].to(logits.device)
            bias = self._partner(here[0], you_play)[low:high + 1]
            for e in mine:
                if e[0] >= here[0] - CHORD and low <= e[2] <= high:   # a key it is already striking now
                    bias[e[2] - low] = -float('inf')
            mask[notes] = bias.to(logits.device)[:, None].expand_as(notes)
            if later:
                mask[T] = 0
            token = self._sample(logits + mask)
            if token == T:
                self._push([T])
                continue
            index = (notes == token).nonzero()[0]
            pitch, velocity = low + int(index[0]), VELOCITIES[int(index[1])]
            break
        # when, within this segment
        self._push([token])
        logits = self._logits()
        mask = torch.full_like(logits, -float('inf'))
        mask[ONSET_IDS[[t % SEGMENT for t in here]].to(logits.device)] = 0
        onset = int((ONSET_IDS.to(logits.device) == self._sample(logits + mask)).nonzero()[0]) + self.segment * SEGMENT
        self.seq.append(int(ONSET_IDS[onset % SEGMENT]))
        # how long
        logits = self._logits()
        mask = torch.full_like(logits, -float('inf'))
        mask[DUR_IDS.to(logits.device)] = 0
        duration = int((DUR_IDS.to(logits.device) == self._sample(logits + mask)).nonzero()[0]) + 1
        self.seq.append(int(DUR_IDS[duration - 1]))
        self.events.append((onset, True, pitch, velocity, duration))
        return (self._seconds(onset), duration / 100, pitch, 'left' if pitch < SPLIT else 'right', velocity)

    def _onsets(self, earliest, latest, gap, you_play):
        """The ticks it may start its next note on: from earliest to latest, not in the gap, on your pulse if any."""
        ticks = [t for t in range(earliest, latest + 1) if gap is None or not gap[0] <= t < gap[1]]
        pulse = self.listener.pulse(self._seconds(self.now()))
        if pulse is None or self.grid >= KNOBS['grid'][2]:
            return ticks or [latest]
        unit = (self.listener.beat(pulse[0]) if you_play else pulse[0]) * 100
        phase = pulse[1] * 100 - self.base
        on = [t for t in ticks if abs((t - phase + unit / 2) % unit - unit / 2) <= self.grid]
        return on or ticks or [latest]

    def _partner(self, tick, you_play):
        """Logits added to each of the 128 pitches at `tick`: your chord, its tensions and your key; under you while
        you play."""
        if self.heard_at is None:
            return torch.zeros(128)
        at = self._seconds(tick)
        bias = torch.tensor(self.listener.bias(at, self.chord_tone, self.tension, self.out_of_key), dtype=torch.float)
        if you_play:
            recent = at - PLAYING / 100 - 1
            low = min((n[2] for n in self.listener.notes if n[0] >= recent), default=128)
            bias[low:] -= self.under
        return bias

    def _sample(self, logits):
        probs = torch.softmax(logits / self.temperature, -1)
        probs[probs < self.min_p * probs.max()] = 0
        return int(torch.multinomial(probs / probs.sum(), 1))

    def _logits(self):
        logits = self.model(self.seq[self.fed:], self.fed)
        self.fed = len(self.seq)
        return logits

    def _seconds(self, tick):
        return (tick + self.base) / 100

    def _last(self):
        return self.events[-1][0] if self.events else 0

    def _push(self, tokens):
        self.seq += tokens
        self.segment += tokens.count(T)

    def _add(self, event):
        """Appends `event`, after the <T> that bring the context to its segment."""
        tick, _, pitch, velocity, duration = event
        while self.segment < tick // SEGMENT:
            self._push([T])
        self.seq += TOKENIZER.encode([('piano', pitch, velocity), ('onset', tick % SEGMENT * STEP_MS),
                                      ('dur', duration * STEP_MS)])
        self.events.append(event)

    def _unwind(self, tick):
        """Takes out of the context the events that start after `tick`, and the <T> past its segment; returns them
        in onset order. The cache is fed again from there."""
        later = []
        while True:
            if self.seq[-1] == T and (self.segment > tick // SEGMENT or self._last() > tick):
                self.seq.pop()
                self.segment -= 1
            elif self.events and self._last() > tick:
                later.insert(0, self.events.pop())
                del self.seq[-3:]
            else:
                break
        self.fed = min(self.fed, len(self.seq))
        return later

    def _fit(self):
        """Keeps the context under CONTEXT tokens: the last WINDOW ticks, moved to start at 0, the cache rebuilt."""
        if len(self.seq) + 40 <= CONTEXT:
            return
        now = self.now()
        kept = [e for e in self.events if e[0] >= now - WINDOW][-(CONTEXT // 2) // 3:]
        shift = kept[0][0] if kept else now
        self.base += shift
        if self.heard_at is not None:
            self.heard_at -= shift
        self.events, self.seq, self.segment, self.fed = [], list(self.prefix), 0, 0
        for tick, *rest in kept:
            self._add((tick - shift, *rest))


def _velocity(velocity):
    """Your velocity as Aria hears it: to its nearest step, 10 at the least."""
    return min(VELOCITIES, key=lambda v: abs(v - velocity))


def _duration(seconds):
    """Your note's duration as Aria hears it: in ticks, from one to its longest."""
    return min(max(round(seconds * 100), 1), MAX_DUR)
