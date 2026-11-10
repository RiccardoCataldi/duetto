"""
The real-time duo with Aria (EleutherAI): a model of expressive solo piano, ~60k hours of transcribed recordings.
https://github.com/EleutherAI/aria

Aria hears one piano: your notes and its own are one performance in its context, four hands on one keyboard. It
plays both its hands and how hard it strikes each key. As with AMT (duo.py), your notes enter the context the moment
you press them and the part of its line that has not sounded yet is thrown away and sampled again.

It jams as a partner does: it plays about as much as you do, in phrases with a breath between them, keeps the plan
it has begun when you play,
plays as loud as you do, answers your phrase with your intervals when you rest, and stops a while after you do.
Aria learnt one pianist at a time: with `wait` it takes turns with you instead, playing once you rest and giving way
as you play again.

Aria's tokens: each note is (piano, pitch, velocity), then its onset in ms within a 5 s segment, then its duration;
<T> starts the next segment. Here times are ticks of 10 ms (Aria's step) since `base`; on the wire, seconds since
the connection opened.
"""

import random
import time

import torch
from ariautils.tokenizer import AbsTokenizer

from music_transformer import chart
from music_transformer.listener import FORCED, NAMES, STAY, Listener, hands

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
SPLIT = 60        # under middle C a note is a left hand's (its own: for the client's dynamics and the piano roll)
# where it plays against you, as a second pianist: clear of your hands while you play, up where a melody sings when
# you rest
RECENT = 200      # ticks of your playing, up to your last note, that say where your hands are
AROUND = 2        # semitones it keeps clear under and over each of your hands
TOP = 72          # once your top hand reaches C5 you play the melody: while you play, the keys over it are yours too
ROOM = 12         # free keys it needs in its octaves to keep apart at full strength; with fewer, it keeps apart less
MELODY = (72, 79) # where its line sings when you rest: the middle of your right hand, from C5 to G5
PITCHES = torch.arange(128, dtype=torch.float)
NEVER = 6000      # a timeout this long: it plays on for good
UNDER = 0.85      # how loud it plays while you play, against you: it accompanies
LOUDER = 30       # steps of velocity away from yours that cost it one `follow`
HEARD = 400       # ticks of your playing, up to your last note, that say how loud you play
RECENTLY = 16     # notes of its own it looks back on, not to play the same keys over and over
BALANCE = 800     # ticks over which it plays about as much as you did, up to your last note

# the knobs you can turn while you play (the piano roll shows them): default, lowest, highest, unit, what it does.
# Aria is a pianist already: they nudge it less than AMT's, and it needs no rules against loops or doubled notes.
KNOBS = {
    'min_step': (0, 0, 60, 'ticks', "time between its onsets when you rest"),
    'space': (15, 0, 150, 'ticks', "time between its onsets while you play"),
    'chord': (3, 1, 8, 'notes', "most notes it starts at once while you play: its comping"),
    'solo_chord': (2, 1, 8, 'notes', "most notes it starts at once when you leave it room: lines and arpeggios"),
    'out_of_key': (4.0, 0.0, 6.0, '', "how strongly it keeps to your key"),
    'chord_tone': (0.5, 0.0, 4.0, '', "how much it leans on the chord you play"),
    'tension': (0.5, 0.0, 4.0, '', "how much it colours your chord with its 9th, 11th and 13th"),
    'apart': (2.0, 0.0, 6.0, '', "how firmly it keeps apart from your hands, and under your melody, while you play"),
    'lead': (3.0, 0.0, 6.0, '', "how firmly it takes the melody, up where yours was, when you rest"),
    'grid': (20, 0, 20, 'ticks', "how far off your pulse its onsets may land (20: free, as Aria plays)"),
    'stay': (0.93, 0.80, 0.995, '', "how sure it must be you modulated (lower = sooner)"),
    'max_rest': (300, 50, 800, 'ticks', "the longest silence it may leave"),
    'wait': (0, 0, 300, 'ticks', "how long you rest before it plays: 0, it plays along with you; more, you take turns"),
    'variety': (1.5, 0.0, 6.0, '', "how much it avoids the keys it has just played: each time it played one costs a quarter of this"),
    'busy': (2.0, 0.5, 8.0, '', "how many notes or chords it plays for each one of yours; at the top, as many as it likes"),
    'phrase': (12, 2, 64, 'notes', "about how many notes or chords it plays before it takes a breath"),
    'breath': (150, 0, 800, 'ticks', "the silence that ends its phrase"),
    'timeout': (1000, 100, NEVER, 'ticks', "how long it plays on after your last note; at the top, for good"),
    'commit': (50, MARGIN, LOOKAHEAD, 'ticks', "how far ahead its plan stands when you play: low, it rethinks at every note"),
    'answer': (400, 0, 1000, 'ticks', "how long after your phrase it answers it"),
    'echo': (1.5, 0.0, 6.0, '', "how closely its answer follows your intervals"),
    'follow': (1.0, 0.0, 4.0, '', "how closely it plays as loud as you do, a little under you while you play"),
    'temperature': (0.95, 0.5, 1.5, '', "how adventurous its choices are"),
    'min_p': (0.035, 0.0, 0.3, '', "how unlikely a note may be, against the likeliest, and still be played"),
    'lowest': (0, 0, 8, 'octave_from', "the lowest octave it may play"),
    'highest': (8, 0, 8, 'octave_to', "the highest octave it may play"),
    'root': (0, 0, 11, 'choice:' + '/'.join(NAMES), "the root of the scale you force it into"),
    'scale': (0, 0, len(FORCED), 'choice:' + '/'.join(['yours', *FORCED]),
              "the scale it keeps to: your key as it hears it, or one you force it into"),
    'bpm': (0, 0, 240, 'bpm', "the tempo its grid keeps to: yours as it hears it, or one you fix"),
    'metronome': (0, 0, 1, 'choice:off/on', "a click on each beat of the tempo you fix"),
    'chart': chart.KNOB(),
}
IN_EFFECT = {'listening': set(), 'accompanying': {'space', 'apart', 'chord', 'commit'}, 'waiting': set(),
             'answering': {'min_step', 'solo_chord', 'lead', 'answer', 'echo', 'timeout'},
             'alone': {'min_step', 'solo_chord', 'lead', 'answer', 'timeout'}}
ALWAYS = {'out_of_key', 'chord_tone', 'tension', 'stay', 'max_rest', 'temperature', 'min_p', 'lowest', 'highest',
          'phrase', 'breath', 'follow', 'scale', 'bpm', 'wait', 'busy', 'variety', 'chart'}


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
        self.clock, self.started = clock, clock() if started is None else started
        self.turn(**knobs)
        self.heard_at = None           # tick of your last note
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
        if 'root' in knobs or 'scale' in knobs:
            self.listener.force(self.root, list(FORCED)[self.scale - 1] if self.scale else None)
        if 'bpm' in knobs or 'chart' in knobs:
            chart.follow(self, 'chart' in knobs)

    def recut(self):
        """Its notes already sent are dropped only when you play (hear() says from when): never after play()."""
        return None

    def knobs(self):
        return {name: getattr(self, name) for name in KNOBS}

    def state(self):
        """What it hears and what it is doing: {'mode', 'key', 'chord', 'bpm', 'active': the knobs in effect}."""
        if self.base is None or self.heard_at is None:
            return chart.reading(self, {'mode': 'listening', 'key': self.listener.forced_name(), 'chord': None,
                                  'bpm': self.bpm or None, 'active': sorted(ALWAYS)})
        now = self.now()
        at = self._seconds(now)
        if now - self.heard_at < (self.wait or PLAYING):
            mode = 'waiting' if self.wait else 'accompanying'
        elif self._timed_out(now):
            mode = 'listening'
        elif now - self.heard_at < self.answer and len(self.listener.phrase(at)) > 1:
            mode = 'answering'
        else:
            mode = 'alone'
        key = self.listener.said_key()
        pulse = self.listener.pulse(at)
        active = ALWAYS | IN_EFFECT[mode] | ({'grid'} if pulse else set())
        active = active - {'stay'} | {'root'} if self.scale else active   # forced: your key no longer counts
        active |= {'metronome'} if self.bpm else set()
        return chart.reading(self, {
            'mode': mode, 'key': self.listener.forced_name() or (self.listener.name(key) if key is not None else None),
            'chord': self.listener.chord_label(at),
            'bpm': round(60 / self.listener.beat(pulse[0])) if pulse else None, 'active': sorted(active)})

    def octaves(self):
        """The lowest and highest pitch it may play, from C of its lowest octave to B of its highest (C4 = 60)."""
        low, high = sorted((self.lowest, self.highest))
        return 12 * (low + 1), min(12 * (high + 1) + 11, 127)

    def now(self):
        return round((self.clock() - self.started) * 100) - self.base

    def ahead(self):
        """Ticks of music already sampled beyond now; the caller samples more while this is under LOOKAHEAD. Once
        you have rested for `timeout` it samples no more until you play."""
        now, last = self.now(), self._last()
        return LOOKAHEAD + 5 if self.heard_at is not None and self._timed_out(max(last, now)) else last - now

    def hear(self, onset, duration, pitch, velocity=80):
        """Your note, pressed at `onset`: Aria hears it when you played it, before those of its notes already on their
        way that come after it. Returns the time from which its unplayed notes are dropped, or None: its plan stands
        for `commit` ticks, as long as your note does not put one of its notes out of your key; not at all when it
        takes turns with you (`wait`)."""
        self.listener.hear(onset, duration, pitch)
        tick = round(onset * 100)
        if self.base is None:
            self.base = tick
        self.heard_at = max(tick - self.base, 0)
        now, later, cut = self.now(), self._unwind(self.heard_at), None
        bias = self.listener.bias(self._seconds(now), 0, 0, self.out_of_key) if later else None
        self._add((self.heard_at, False, pitch, _velocity(velocity), _duration(duration)))
        for event in later:
            if event[1] and event[0] > now + MARGIN and (
                    cut is not None or self.wait or event[0] > now + self.commit
                    or bias[event[2]] < -self.out_of_key / 2):
                cut = event[0] if cut is None else cut   # its own, not sounded yet: sampled again, now hearing you
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
        if self.wait and self.heard_at is not None:   # its turn: once you have rested `wait`, on its own
            earliest, you_play = max(earliest, self.heard_at + self.wait), False
        latest = max(earliest, last + self.max_rest)
        mine = [e for e in self.events if e[1]]
        step, most = (self.space, self.chord) if you_play else (self.min_step, self.solo_chord)
        gap = None
        if mine:
            group = [e for e in mine if e[0] >= mine[-1][0] - CHORD]   # its last chord
            start = group[0][0]
            # its phrase has run out, or it has played as much as you: it keeps silent, its chord as it stands
            silent = max(self._breath(), self._room(mine))
            if silent > start + step:
                earliest, most = max(earliest, silent), 0
            if len(group) >= most:
                earliest = max(earliest, start + max(step, CHORD + 1))
            elif step > CHORD:
                gap = (start + CHORD + 1, start + step)               # no onset between its chord and step
            latest = max(latest, start + step, earliest)
        onsets = self._onsets(earliest, latest, gap, you_play)
        closed = range(*gap) if gap else ()   # where it would play if it could: there, it has left its chord
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
            bias = self._partner(here[0])[low:high + 1]
            for e in mine[-RECENTLY:]:
                if low <= e[2] <= high:   # the keys it has just played, over and over: it moves on
                    bias[e[2] - low] -= self.variety / 4
            for e in mine:
                if e[0] >= here[0] - CHORD and low <= e[2] <= high:   # a key it is already striking now
                    bias[e[2] - low] = -float('inf')
            bias = bias.to(logits.device)
            if later:   # whether it plays now or rests on: as it would wherever it played
                mask[notes] = bias[:, None].expand_as(notes)
                mask[T] = 0
                if self._sample(logits + mask) == T:
                    self._push([T])
                    continue
                mask[T] = -float('inf')
            register = self._register(here[0], you_play)[low:high + 1].to(logits.device)   # then where, and how hard
            mask[notes] = (bias + register)[:, None] + self._loudness(you_play).to(logits.device)
            token = self._sample(logits + mask)
            index = (notes == token).nonzero()[0]
            pitch, velocity = low + int(index[0]), VELOCITIES[int(index[1])]
            break
        # when, within this segment
        self._push([token])
        logits = self._logits()
        mask = torch.full_like(logits, -float('inf'))
        after = [t for t in here if gap and t >= gap[1]]
        wished = here + [t for t in closed if t // SEGMENT == self.segment] if after else here
        mask[ONSET_IDS[[t % SEGMENT for t in wished]].to(logits.device)] = 0
        onset = int((ONSET_IDS.to(logits.device) == self._sample(logits + mask)).nonzero()[0]) + self.segment * SEGMENT
        if onset in closed:   # not with its chord: at the first tick it may play after it, never piled on the chord
            onset = after[0]
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

    def _partner(self, tick):
        """Logits added to each of the 128 pitches at `tick`: your chord, its tensions and your key; after your
        phrase, its answer: from its first note on, the pitch that continues your intervals, over and over."""
        if self.heard_at is None:
            return torch.zeros(128)
        at = self._seconds(tick)
        chord_tone = max(self.chord_tone, chart.CHANGES) if self.chart else self.chord_tone
        bias = torch.tensor(self.listener.bias(at, chord_tone, self.tension, self.out_of_key), dtype=torch.float)
        if PLAYING <= tick - self.heard_at < self.answer:
            phrase = self.listener.phrase(at)
            line = [e[2] for e in self.events[-64:] if e[1] and e[0] > self.heard_at and e[2] >= SPLIT]
            if line and len(phrase) > 1:
                step = (len(line) - 1) % (len(phrase) - 1) + 1
                echo, in_key = line[-1] + phrase[step] - phrase[step - 1], self.listener.in_key()
                if in_key[echo % 12] < 0.5:   # your interval, bent into your key as a sequence would be
                    echo += 1 if in_key[(echo + 1) % 12] >= in_key[(echo - 1) % 12] else -1
                if 0 <= echo < 128:
                    bias[echo] += self.echo
        return bias

    def _loudness(self, you_play):
        """Logits added to each velocity it may strike a key with: as loud as you have just played, a little under
        you while you play."""
        yours = [e[3] for e in self.events[-64:] if not e[1] and e[0] >= (self.heard_at or 0) - HEARD]
        if not yours:
            return torch.zeros(len(VELOCITIES))
        level = sum(yours) / len(yours) * (UNDER if you_play else 1)
        return -self.follow * (torch.tensor(VELOCITIES, dtype=torch.float) - level).abs() / LOUDER

    def _timed_out(self, tick):
        """Whether it has played on long enough, by `tick`, after your last note."""
        return self.timeout < NEVER and tick - self.heard_at >= self.timeout

    def _room(self, mine):
        """The tick before which it starts no new note or chord: in any BALANCE ticks it plays `busy` of them for each
        one you played in the BALANCE up to your last note, and one more."""
        if self.heard_at is None or self.busy >= KNOBS['busy'][2]:
            return 0
        yours = _attacks(e[0] for e in self.events if not e[1] and e[0] >= self.heard_at - BALANCE)
        mine, most = _attacks(e[0] for e in mine), int(self.busy * len(yours)) + 1
        return mine[-most] + BALANCE if len(mine) >= most else 0

    def _breath(self):
        """The tick before which it keeps silent: a breath after its last note, once its phrase has run out. Its
        phrase is its notes since its last silence of a breath, a chord counted once; about `phrase` of them, a
        different number each time."""
        phrase = []   # their ticks, the last first
        for tick, mine, *_ in reversed(self.events):
            if mine and phrase and phrase[-1] - tick >= self.breath:
                break
            if mine and (not phrase or phrase[-1] - tick > CHORD):
                phrase.append(tick)
        if not self.breath or len(phrase) < self.phrase * random.Random(phrase[-1] if phrase else 0).uniform(0.6, 1.4):
            return 0
        return phrase[0] + self.breath

    def _register(self, tick, you_play):
        """Logits added to each of the 128 pitches for where it plays at `tick`, as a second pianist: while you play,
        -apart on your hands and over your melody, less when that leaves it little room; when you rest, down to -lead
        an octave under where a melody sings (the middle of your right hand, from C5 to G5)."""
        register = torch.zeros(128)
        if self.heard_at is None:
            return register
        notes = self.listener.notes
        if you_play:
            at, since = self._seconds(tick), self._seconds(self.heard_at - RECENT)
            playing = [p for o, d, p in notes if o >= since or o + d >= at or (o, p) in self.listener.held]
            yours = hands(playing)
            for lowest, highest in yours:
                register[max(lowest - AROUND, 0):highest + AROUND + 1] = -self.apart
            if yours and yours[-1][1] >= TOP:   # the melody is yours: it does not comp over it
                register[yours[-1][1] + AROUND + 1:] = -self.apart
            low, high = self.octaves()
            return register * min(int((register[low:high + 1] == 0).sum()) / ROOM, 1)
        right = sorted(p for _, _, p in notes if p >= SPLIT)
        middle = min(max(right[len(right) // 2] if right else 0, MELODY[0]), MELODY[1])
        return -self.lead * ((middle - PITCHES) / 12).clamp(0, 1)

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


def _attacks(ticks):
    """Of `ticks` in order, those that start a note or a chord: more than CHORD after the start of the one before."""
    starts = []
    for tick in ticks:
        if not starts or tick - starts[-1] > CHORD:
            starts.append(tick)
    return starts


def _velocity(velocity):
    """Your velocity as Aria hears it: to its nearest step, 10 at the least."""
    return min(VELOCITIES, key=lambda v: abs(v - velocity))


def _duration(seconds):
    """Your note's duration as Aria hears it: in ticks, from one to its longest."""
    return min(max(round(seconds * 100), 1), MAX_DUR)
