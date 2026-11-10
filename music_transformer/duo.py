"""
A real-time duo with the Anticipatory Music Transformer: you play one instrument, the model improvises on another
at the same time. Your notes enter the context the moment you press them; the part of the model's line that has
not sounded yet is thrown away and sampled again, now hearing you.

Times on the wire are seconds since the connection opened; inside the context they are ticks since `base`.
"""

import math
import random
import time

import numpy as np
import torch
from anticipation.config import CONTEXT_SIZE, MAX_DUR, MAX_PITCH, MAX_TIME, TIME_RESOLUTION
from anticipation.vocab import AUTOREGRESS, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET
from anticipation.sample import safe_logits, nucleus

from music_transformer.listener import FORCED, NAMES, STAY, Listener, hands
from music_transformer.memory import Memory

NAME = 'amt'
LOOKAHEAD = 100   # ticks the model plays ahead of now (1 s)
MARGIN = 15       # ticks: a new note of the model sounds no earlier than this, to cover sampling and the network
MAX_REST = 200    # ticks: the longest silence the model may leave; it reads long silences as a sparse piece and stops
LONGEST_REST = 800   # ticks: the most the max_rest knob allows
# it breathes: silence is something it does, not a long gap it happens to sample
PHRASE = 16       # notes it plays, about, before it takes a breath
BREATH = 150      # ticks of silence that end a phrase
TIMEOUT = 1000    # ticks it plays on after your last note; then it is silent until you play
NEVER = 6000      # a timeout this long: it plays on for good
SILENCE = 0.1     # chance that it leaves out a slice of its memory and stays silent as long as the slice lasts
EXACT = 10.0      # recall at this strength: it plays what it remembers note for note
COMMIT = 50       # ticks ahead of now its plan stands when you play, as long as your note does not contradict it
# left free, the model plays like a full Lakh MIDI track: dense chords and the same arpeggio over and over.
# These keep it a pianist: it comps in chords while you play and plays lines when you rest.
CHORD = 3         # most notes its right hand may start at once while you play
SOLO_CHORD = 2    # and when you rest
MIN_STEP = 20     # ticks between two of its onsets that are not a chord
NO_REPEAT = 3     # it may not play again a run of this many pitches it played in its last LINE notes
LINE = 32
# it listens to you more than to itself:
MEMORY = 200      # ticks of its own playing it remembers; all of yours within WINDOW stays
SPACE = 50        # ticks between its onsets while you are playing: it leaves you room
PLAYING = 100     # ticks after your last note that you still count as playing
# it listens like a partner (listener.py): your key, your chord, your pulse, your phrases.
CHORD_TONE = 1.0  # logit added to the tones of the chord you are playing
TENSION = 1.0     # logit added to the tensions of that chord (its 9th, 11th or #11th, 13th), as a jazz pianist colours it
OUT_OF_KEY = 2.0  # logit taken from a pitch class outside your key; in between while you are modulating
ANSWER = 400      # ticks after your phrase ends in which it answers it, following its intervals
ECHO = 3.0        # logit added to the pitch that continues your phrase's intervals, over and over as a sequence
RUN = MAX_REST    # ticks: the longest silence it may leave while it answers; low, it answers in a run of notes
GRID = 2          # ticks off your pulse that its onsets may land (your beat while you play, your tatum otherwise)
WINDOW = 2000     # ticks of the duo the model keeps hearing when the context fills up or the clock runs out
KEEP = 600        # tokens kept at most in that case
PRUNE = 6         # its notes past MEMORY wait to be this many: forgetting them feeds the whole context again
# two hands, as a pianist plays: its line over a left hand of its own, another instrument in the context so that
# the model hears them as two parts. Each hand keeps to its register and its own rhythm; both sound as the model.
HANDS = 2
LEFT = 32         # the General MIDI program the left hand is in the context: Lakh MIDI's bass lines move like one
LEFT_STEP = 40    # ticks between two of the left hand's onsets that are not a chord
LEFT_CHORD = 3    # most notes the left hand may start at once
LEFT_TOP = 64     # the highest pitch of the left hand (E4)
RIGHT_BOTTOM = 53 # the lowest pitch of the right hand (F3), while there are two
# where it plays against you, as a second pianist: clear of your hands while you play, up where a melody sings when
# you rest
APART = 2.0       # logit taken from the keys of your hands while you play, and from those over your melody
LEAD = 3.0        # when you rest: logit taken from its right hand's pitches an octave or more under your melody
RECENT = 200      # ticks of your playing, up to your last note, that say where your hands are
AROUND = 2        # semitones it keeps clear under and over each of your hands
TOP = 72          # once your top hand reaches C5 you play the melody: while you play, the keys over it are yours too
ROOM = 12         # free keys a hand of its needs to keep apart at full strength; with fewer, it keeps apart less
SPLIT = 60        # your notes from middle C up are your right hand's
MELODY = (72, 79) # where its line sings when you rest: the middle of your right hand, from C5 to G5
PITCHES = torch.arange(MAX_PITCH, dtype=torch.float)

# the knobs you can turn while you play (the piano roll shows them): default, lowest, highest, unit, what it does
KNOBS = {
    'min_step': (MIN_STEP, 5, 60, 'ticks', "time between its onsets when you rest"),
    'space': (SPACE, 10, 150, 'ticks', "time between its onsets while you play"),
    'out_of_key': (OUT_OF_KEY, 0.0, 6.0, '', "how strongly it keeps to your key"),
    'chord_tone': (CHORD_TONE, 0.0, 4.0, '', "how much it leans on the chord you play"),
    'tension': (TENSION, 0.0, 4.0, '', "how much it colours your chord with its 9th, 11th and 13th"),
    'chord': (CHORD, 1, 6, 'notes', "most notes it starts at once while you play: its comping"),
    'solo_chord': (SOLO_CHORD, 1, 6, 'notes', "most notes it starts at once when you rest: lines"),
    'apart': (APART, 0.0, 6.0, '', "how firmly it keeps apart from your hands, and under your melody, while you play"),
    'lead': (LEAD, 0.0, 6.0, '', "how firmly it takes the melody, up where yours was, when you rest"),
    'answer': (ANSWER, 0, 1000, 'ticks', "how long after your phrase it answers it"),
    'echo': (ECHO, 0.0, 6.0, '', "how closely its answer follows your intervals"),
    'run': (RUN, 5, LONGEST_REST, 'ticks', "the longest silence it may leave while it answers: low, it answers in a run"),
    'grid': (GRID, 0, 20, 'ticks', "how far off your pulse its onsets may land"),
    'stay': (STAY, 0.80, 0.995, '', "how sure it must be you modulated (lower = sooner)"),
    'no_repeat': (NO_REPEAT, 2, 8, 'notes', "the length of runs it may not repeat"),
    'phrase': (PHRASE, 2, 64, 'notes', "about how many notes it plays before it takes a breath"),
    'breath': (BREATH, 0, LONGEST_REST, 'ticks', "the silence that ends its phrase"),
    'timeout': (TIMEOUT, 0, NEVER, 'ticks', "how long it plays on after your last note; at the top, for good"),
    'recall': (4.0, 0.0, EXACT, '', "how closely it follows its memory of you; 0: it only invents, at the top: it quotes"),
    'silence': (SILENCE, 0.0, 0.9, '', "how often it leaves out what it remembers and stays silent instead"),
    'sparse': (0, 0, 1, 'choice:off/on', "with nothing to remember: on, it keeps silent; off, it invents"),
    'decay': (30.0, 5.0, 120.0, 's', "how long an idea of where you are lasts unheard"),
    'least': (1.5, 0.5, 8.0, '', "how sure it must be of where you are before it follows its memory"),
    'continuity': (1.5, 1.0, 4.0, '', "how much it prefers to carry on along what it is remembering"),
    'jump': (8, 2, 32, 'slices', "how far along one memory it may go before it must jump"),
    'melodic': (1.0, 0.0, 3.0, '', "how much your line wakes its memory"),
    'harmonic': (1.0, 0.0, 3.0, '', "how much your harmony wakes its memory"),
    'itself': (0.5, 0.0, 3.0, '', "how much its own line wakes its memory"),
    'probabilistic': (0, 0, 1, 'choice:strongest/by chance', "the memory it follows: always the strongest, or by chance"),
    'commit': (COMMIT, MARGIN, LOOKAHEAD, 'ticks', "how far ahead its plan stands when you play: low, it rethinks at every note"),
    'memory': (MEMORY, 50, 2000, 'ticks', "how much of its own playing it remembers"),
    'max_rest': (MAX_REST, 50, LONGEST_REST, 'ticks', "the longest silence it may leave"),
    'hands': (HANDS, 1, 2, 'hands', "one hand, a single line; or two: its line over a left hand of its own"),
    'left_step': (LEFT_STEP, 10, 200, 'ticks', "time between its left hand's onsets"),
    'temperature': (1.0, 0.5, 1.5, '', "how adventurous its choices are"),
    'top_p': (0.98, 0.80, 1.0, '', "how many unlikely notes it still considers"),
    'lowest': (0, 0, 8, 'octave_from', "the lowest octave it may play"),
    'highest': (8, 0, 8, 'octave_to', "the highest octave it may play"),
    'root': (0, 0, 11, 'choice:' + '/'.join(NAMES), "the root of the scale you force it into"),
    'scale': (0, 0, len(FORCED), 'choice:' + '/'.join(['yours', *FORCED]),
              "the scale it keeps to: your key as it hears it, or one you force it into"),
    'bpm': (0, 0, 240, 'bpm', "the tempo it keeps to, its beats on the metronome's: yours as it hears it, or one you fix"),
    'metronome': (0, 0, 1, 'choice:off/on', "a click on each beat of the tempo you fix"),
    # the jam_bot's ways (jam.py): it starts again from your last notes when its strategy says, and plays on freely
    'strategy': (0, 0, 3, 'choice:duo/trade/gesture/request',
                 "the duo's own way, or the jam_bot's: taking turns, starting again after each of your gestures, or when you cue it"),
    'context': (50, 10, 200, 'notes', "jam_bot: how many of the last notes it starts again from"),
    'bars': (2, 1, 8, 'bars', "trade: the bars each of you plays in turn"),
    'wait': (10, 0, 150, 'ticks', "gesture: how long after your last note it starts again from it (longer: it waits for a held note)"),
    'lock': (0, 0, 1, 'choice:off/on', "gesture: your last gesture held ahead of it over and over, for it to follow strictly"),
    'every': (50, 10, 200, 'ticks', "gesture, locked: how often your last gesture is held again"),
    'trigger': (36, 21, 60, 'pitch', "request: your keys under this one cue it to start again from your last notes"),
}
RECALLED = ('decay', 'least', 'continuity', 'jump', 'melodic', 'harmonic', 'itself', 'probabilistic')   # memory.py's
# what it is doing, and the knobs that shape its notes then; the others wait for their moment
IN_EFFECT = {
    'listening': set(),
    'accompanying': {'space', 'chord', 'apart', 'commit'},         # you are playing
    'answering': {'min_step', 'solo_chord', 'lead', 'answer', 'echo', 'run', 'timeout'},   # your phrase just ended
    'alone': {'min_step', 'solo_chord', 'lead', 'answer', 'timeout'},   # you rest, and it has answered
}
ALWAYS = {'out_of_key', 'chord_tone', 'tension', 'stay', 'no_repeat', 'memory', 'max_rest', 'temperature', 'top_p',
          'lowest', 'highest', 'hands', 'scale', 'bpm', 'phrase', 'breath', 'recall', 'silence', 'sparse', 'strategy',
          *RECALLED}


def knob(name, value):
    """`value` for the knob `name`, kept within its range."""
    default, low, high, *_ = KNOBS[name]
    value = min(max(value, low), high)
    return round(value) if isinstance(default, int) else float(value)


class DuoSession:
    def __init__(self, model, device, you=0, me=4, left=LEFT, clock=time.monotonic, started=None, corpus=(), **knobs):
        self.model, self.device = model, device
        self.you, self.me, self.left = you, me, left   # General MIDI programs: piano; its right and left hands
        self.listener, self.mind = Listener(), Memory()
        self.mind.load(corpus)
        for name, (default, *_) in KNOBS.items():
            setattr(self, name, default)
        self.turn(**{name: getattr(self, name) for name in RECALLED})
        self.turn(**knobs)
        self.heard_at = None           # context time of your last note
        self.clock, self.started = clock, clock() if started is None else started
        self.base = None               # session tick of time 0 in the context: your first note
        self.seq = [AUTOREGRESS]       # complete events in time order, yours and the model's
        self.past, self.fed = None, 0  # the KV cache covers seq[:fed]
        self.spoken = []               # session seconds of its last notes that were not cut
        self.plan, self.loose = [], 0  # (seconds, pitch, duration) it remembers and has yet to play; ticks its first
                                       # note may wait for your pulse: it has just jumped
        self.cursor = 0.0              # session seconds of its next slice: a rest in its memory passes here
        self.free, self.owed = True, False   # it invents, its memory silent; your note is still to be answered

    def turn(self, **knobs):
        """Sets knobs (see KNOBS), kept within their range; they take effect from the model's next note."""
        for name, value in knobs.items():
            setattr(self, name, knob(name, value))
        for name in RECALLED:
            if name in knobs:
                setattr(self.mind, name, getattr(self, name))
        if 'stay' in knobs:
            self.listener.stay(self.stay)
        if 'root' in knobs or 'scale' in knobs:
            self.listener.force(self.root, list(FORCED)[self.scale - 1] if self.scale else None)
        if 'bpm' in knobs:
            self.listener.fix(self.bpm)

    def state(self):
        """What it hears and what it is doing: {'mode', 'key', 'chord', 'bpm', 'active': the knobs in effect}."""
        if self.base is None or self.heard_at is None:
            return {'mode': 'listening', 'key': self.listener.forced_name(), 'chord': None, 'bpm': self.bpm or None,
                    'active': sorted(ALWAYS)}
        now = self.now()
        at = self._seconds(now)
        if now - self.heard_at < PLAYING:
            mode = 'accompanying'
        elif self._timed_out(now):
            mode = 'listening'
        elif now - self.heard_at < self.answer and len(self.listener.phrase(at)) > 1:
            mode = 'answering'
        else:
            mode = 'alone'
        key = self.listener.said_key()
        chord = self.listener.chord(at)
        pulse = self.listener.pulse(at)
        active = ALWAYS | IN_EFFECT[mode] | ({'grid'} if pulse else set()) | ({'left_step'} if self.hands == 2 else set())
        active = active - {'stay'} | {'root'} if self.scale else active   # forced: your key no longer counts
        active |= {'metronome'} if self.bpm else set()
        heard = self.listener.name(key) if key is not None else None
        return {'mode': mode, 'key': self.listener.forced_name() or heard,
                'chord': self.listener.chord_label(at) if chord else None,
                'bpm': round(60 / self.listener.beat(pulse[0])) if pulse else None, 'active': sorted(active)}

    def octaves(self):
        """The lowest and highest pitch it may play, from C of its lowest octave to B of its highest (C4 = 60)."""
        low, high = sorted((self.lowest, self.highest))
        return 12 * (low + 1), min(12 * (high + 1) + 11, MAX_PITCH - 1)

    def recut(self):
        """Its notes already sent are dropped only when you play (hear() says from when): never after play()."""
        return None

    def knobs(self):
        return {name: getattr(self, name) for name in KNOBS}

    def now(self):
        return round((self.clock() - self.started) * TIME_RESOLUTION) - self.base

    def ahead(self):
        """Ticks of music already sampled beyond now; the caller samples more while this is under LOOKAHEAD."""
        now, last = self.now(), max(self._last_time(), self._breath(), self._tick(self.cursor))
        return LOOKAHEAD + 5 if self._timed_out(max(last, now)) and not self.owed else last - now

    def hear(self, onset, duration, pitch, velocity=None):
        """Your note, pressed at `onset` (AMT does not hear how hard): the model hears it when you played it, before
        those of its notes already on their way that come after it. Returns the time from which the model's unplayed
        notes are dropped, or None."""
        self.listener.hear(onset, duration, pitch)
        self.mind.hear(onset, pitch, duration, self._phase(onset))
        tick = round(onset * TIME_RESOLUTION)
        if self.base is None:
            self.base = tick
        self._fit()   # it may have kept silent for longer than a context lasts
        waited = self.heard_at is None or not self.timeout or self._timed_out(max(self.now(), self._last_time()))
        self.heard_at = max(tick - self.base, 0)
        now, later, cut = self.now(), self._unwind(self.heard_at), None
        self.owed = True
        if waited:                     # the trigger: you are what it was waiting for
            self.cursor = min(self.cursor, self._seconds(now + MARGIN))
        bias = self.listener.bias(self._seconds(now), self.chord_tone, self.tension, self.out_of_key) if later else None
        self.seq += [TIME_OFFSET + self.heard_at, DUR_OFFSET + _duration(duration),
                     NOTE_OFFSET + self.you * MAX_PITCH + pitch]
        for event in later:
            tick_, pitch = event[0] - TIME_OFFSET, (event[2] - NOTE_OFFSET) % MAX_PITCH
            if self._instrument(event[2]) in self._mine() and tick_ > now + MARGIN and (
                    cut is not None or tick_ > now + self.commit or bias[pitch] < -self.out_of_key / 2):
                cut = self._seconds(tick_) if cut is None else cut   # not sounded yet: sampled again
            else:
                self.seq += event
        if cut is not None:
            self.spoken = [onset for onset in self.spoken if onset < cut]
            self.plan, self.cursor = [], min(self.cursor, cut)
        # otherwise its own line, two or three times denser than yours, drowns you out
        events = self._events()
        kept = [e for e in events
                if self._instrument(e[2]) not in self._mine() or e[0] - TIME_OFFSET >= self.heard_at - self.memory]
        if len(events) - len(kept) >= PRUNE:
            self._restart(kept)
        return cut

    def release(self, onset, pitch, duration):
        """Your note pressed at `onset` stopped sounding after `duration`, the pedal's ring and all: the model hears
        how long it really was from its next note on."""
        self.listener.release(onset, pitch, duration)
        self.mind.release(onset, pitch, duration)
        if self.base is None:
            return
        time_ = TIME_OFFSET + max(round(onset * TIME_RESOLUTION) - self.base, 0)
        for i in range(1, len(self.seq) - 2, 3):
            if (self.seq[i], self.seq[i + 2]) == (time_, NOTE_OFFSET + self.you * MAX_PITCH + pitch):
                if self.seq[i + 1] != DUR_OFFSET + _duration(duration):
                    self.seq[i + 1] = DUR_OFFSET + _duration(duration)
                    self._crop(i + 1)
                return

    @torch.no_grad()
    def play(self):
        """Samples the model's next note: (onset, duration, pitch, hand) in session seconds, hand 'left' or 'right';
        or None when it stays silent."""
        self._fit()
        now, last = self.now(), self._last_time()
        you_play = self.heard_at is not None and now - self.heard_at < PLAYING
        self.owed = False
        if not self.plan:
            self._step(now, last, you_play)
        if self.plan:
            return self._remembered(now, last, you_play)
        if not self.free:
            return None
        answering = self.heard_at is not None and not you_play and now - self.heard_at < self.answer
        after = max(last, self._breath())
        earliest = max(after, now + MARGIN)
        latest = min(max(earliest, after + (min(self.max_rest, self.run) if answering else self.max_rest)), MAX_TIME - 1)
        # when each hand may play next: (instrument, its line, its register, its onsets as a mask over times)
        hands = []
        for instrument, step, most, register in self._hands(you_play):
            line, window = self._my_line(instrument), torch.full((MAX_TIME,), -float('inf'))
            start, end = earliest, latest
            if line:
                mine = line[-1][0]
                chord = [note for time_, note in line if time_ == mine]
                start = max(start, mine + (step if len(chord) >= most else 0))
                end = min(max(end, mine + step), MAX_TIME - 1)
            window[start:end + 1] = 0
            if line:
                window[mine + 1:mine + step] = -float('inf')
            hands.append((instrument, line, register, window))
        for slot in range(3):
            logits = self._logits()
            mask = torch.full_like(logits, -float('inf'))
            if slot == 0:
                times = torch.stack([window for *_, window in hands]).max(0).values.to(mask.device)
                mask[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = times
                mask = self._on_your_pulse(mask, earliest, int(torch.nonzero(torch.isfinite(times)).max()), you_play)
            elif slot == 1:
                mask[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = 0
            else:
                tick = self.seq[-2] - TIME_OFFSET
                for instrument, line, (low, high), window in hands:
                    if not torch.isfinite(window[tick]):
                        continue   # this hand may not play now
                    first = NOTE_OFFSET + instrument * MAX_PITCH
                    mask[first + low:first + high + 1] = 0
                    for note in self._banned(line, tick):
                        mask[note] = -float('inf')
                    if self.heard_at is not None:
                        mask[first:first + MAX_PITCH] += self._partner(tick, line, (low, high), instrument == self.me)
            token = int(torch.multinomial(torch.softmax(nucleus(logits + mask, self.top_p), -1), 1))
            self.seq.append(token)
        return self._played()

    def _step(self, now, last, you_play):
        """What its memory says comes next: its next slice into the plan, or a rest; `free` when it says nothing.
        It turns to its memory at a share of its steps as large as recall is of EXACT, and invents at the others."""
        self.free = True
        if not self.recall or (not self.sparse and random.random() >= self.recall / EXACT):
            return
        self.mind.update(self._seconds(now))
        at = self._seconds(max(now + MARGIN, last, self._tick(self.cursor)))
        event = self.mind.step(at, self._phase(at))
        if self.sparse and (event is None or not event[2]):
            self.free, self.cursor = False, self._seconds(now + LOOKAHEAD + 10)
        if event is None or (self.sparse and not event[2]):
            return
        slice_, shift, _ = event
        # a rest it remembers lasts no longer than one it would leave on its own
        self.free, self.cursor = False, at + min(self.mind.duration(slice_), self.max_rest / TIME_RESOLUTION)
        if random.random() < self.silence:
            return
        low, high = self.octaves()
        notes = self.mind.notes(slice_, shift)[-((self.chord if you_play else self.solo_chord) + LEFT_CHORD * (self.hands == 2)):]
        self.plan = [(at, min(max(p, low + (p - low) % 12), high - (high - p) % 12), d) for p, d in notes]
        self.loose = 0 if self.mind.run else round(self.listener.beat(pulse[0]) * TIME_RESOLUTION) \
            if (pulse := self.listener.pulse(at)) else 0

    def _remembered(self, now, last, you_play):
        """The next note of the plan: when its memory says and about what it says, as strictly as `recall`; AMT fills
        in the rest, and may change a note unless recall is EXACT."""
        at, pitch, duration = self.plan.pop(0)
        tick = min(max(self._tick(at), last, now + MARGIN), MAX_TIME - 1)
        end, self.loose = min(tick + self.loose, MAX_TIME - 1), 0
        instrument = self.left if self.hands == 2 and pitch < RIGHT_BOTTOM else self.me
        register, line = self.octaves(), self._my_line(instrument)
        for slot in range(3):
            logits = self._logits()
            mask = torch.full_like(logits, -float('inf'))
            if slot == 0:
                mask[TIME_OFFSET + tick:TIME_OFFSET + end + 1] = 0
                mask = self._on_your_pulse(mask, tick, end, you_play) if end > tick else mask
            elif slot == 1:
                length = _duration(duration)
                mask[DUR_OFFSET + max(length // 2, 1):DUR_OFFSET + min(2 * length, MAX_DUR - 1) + 1] = 0
            elif self.recall >= EXACT:
                mask[NOTE_OFFSET + instrument * MAX_PITCH + pitch] = 0
            else:
                first = NOTE_OFFSET + instrument * MAX_PITCH
                mask[first + register[0]:first + register[1] + 1] = 0
                for note in self._banned(line, tick):
                    mask[note] = -float('inf')
                mask[first:first + MAX_PITCH] += self._partner(tick, line, register, instrument == self.me)
                mask[first + pitch] += self.recall
            token = int(torch.multinomial(torch.softmax(nucleus(logits + mask, self.top_p), -1), 1))
            self.seq.append(token)
        return self._played()

    def _played(self):
        """The note just sampled, once it has joined what it has said and woken its memory of itself."""
        time_, duration, note = self.seq[-3:]
        onset, pitch = self._seconds(time_ - TIME_OFFSET), (note - NOTE_OFFSET) % MAX_PITCH
        self.spoken = self.spoken[-256:] + [onset]
        self.cursor = max(self.cursor, onset)
        self.mind.own(onset, pitch)
        return onset, (duration - DUR_OFFSET) / TIME_RESOLUTION, pitch, 'left' if self._instrument(note) == self.left else 'right'

    def _tick(self, seconds):
        return round(seconds * TIME_RESOLUTION) - self.base

    def _phase(self, seconds):
        """Where `seconds` falls in your beat, 0 to 1; None while your rhythm is unclear."""
        pulse = self.listener.pulse(seconds)
        return ((seconds - pulse[1]) / self.listener.beat(pulse[0])) % 1 if pulse else None

    def _timed_out(self, tick):
        """Whether it has played on long enough, by `tick`, after your last note."""
        return self.timeout < NEVER and tick - self.heard_at >= self.timeout

    def _breath(self):
        """The tick before which it keeps silent: a breath after its last note, once its phrase has run out of notes.
        Its phrase is its notes since its last silence of a breath; about `phrase` of them, a different number each."""
        if not self.spoken or not self.free:
            return 0
        start, gap = len(self.spoken) - 1, self.breath / TIME_RESOLUTION
        while start > 0 and self.spoken[start] - self.spoken[start - 1] < gap:
            start -= 1
        notes = self.phrase * random.Random(self.spoken[start]).uniform(0.6, 1.4)
        if len(self.spoken) - start < notes:
            return 0
        return round(self.spoken[-1] * TIME_RESOLUTION) - self.base + self.breath

    def _hands(self, you_play):
        """(instrument, step, most notes at once, (lowest, highest) pitch) of each hand it plays with."""
        low, high = self.octaves()
        right = (self.me, self.space if you_play else self.min_step, self.chord if you_play else self.solo_chord)
        if self.hands == 1:
            return [(*right, (low, high))]
        hands = [(*right, (max(low, RIGHT_BOTTOM), high)), (self.left, self.left_step, LEFT_CHORD, (low, min(high, LEFT_TOP)))]
        hands = [hand for hand in hands if hand[3][0] <= hand[3][1]]   # its octaves may leave room for one hand only
        return hands or [(*right, (low, high))]

    def _seconds(self, tick):
        return (tick + self.base) / TIME_RESOLUTION

    def _on_your_pulse(self, mask, earliest, latest, you_play):
        """Keeps only the onsets on your beat (while you play) or your tatum; unchanged while your rhythm is unclear."""
        pulse = self.listener.pulse(self._seconds(self.now()))
        if pulse is None:
            return mask
        unit = self.listener.beat(pulse[0]) if you_play else pulse[0]
        first = math.ceil((self._seconds(earliest) - pulse[1]) / unit - self.grid / (unit * TIME_RESOLUTION))
        grid = torch.full_like(mask, -float('inf'))
        for k in range(first, first + math.ceil((latest - earliest) / (unit * TIME_RESOLUTION)) + 2):
            tick = round((pulse[1] + k * unit) * TIME_RESOLUTION) - self.base
            grid[TIME_OFFSET + max(tick - self.grid, earliest):TIME_OFFSET + min(tick + self.grid, latest) + 1] = 0
        on = mask + grid
        return on if torch.isfinite(on).any() else mask

    def _partner(self, tick, line, register, right):
        """Logits added to the pitches of one of its hands at `tick` (its `line`, its `register`; `right`: its right
        hand): your chord and key; where it plays against you (_register); after your phrase, its answer."""
        at, you_play = self._seconds(tick), tick - self.heard_at < PLAYING
        in_key = self.listener.in_key()
        expected = self.mind.expect(at)[0] if self.recall else None   # the harmony it remembers comes next, not yours now
        bias = torch.tensor(self.listener.bias(at, self.chord_tone, self.tension, self.out_of_key)[:MAX_PITCH]
                            + (0 if expected is None else self.recall * np.tile(expected, 11)[:MAX_PITCH]),
                            dtype=torch.float) + self._register(tick, you_play, register, right)
        if not you_play and right and line and tick - self.heard_at < self.answer:
            phrase = self.listener.phrase(at)
            answered = sum(time_ > self.heard_at for time_, _ in line)
            if answered > 0 and len(phrase) > 1:   # past your phrase's end it plays its intervals again, a sequence
                step = (answered - 1) % (len(phrase) - 1) + 1
                echo = (line[-1][1] - NOTE_OFFSET) % MAX_PITCH + phrase[step] - phrase[step - 1]
                if in_key[echo % 12] < 0.5:   # your interval, bent into your key as a sequence would be
                    echo += 1 if in_key[(echo + 1) % 12] >= in_key[(echo - 1) % 12] else -1
                if 0 <= echo < MAX_PITCH:
                    bias[echo] += self.echo
        return bias.to(self.device)

    def _register(self, tick, you_play, register, right):
        """Logits added to its pitches for where it plays at `tick`, as a second pianist: while you play, -apart on
        your hands and over your melody, less when that leaves the hand (its `register`) little room; when you rest,
        its right hand down to -lead an octave under where a melody sings (the middle of your right hand, from C5
        to G5). AMT samples when before where: this moves its notes, never how often it plays."""
        bias = torch.zeros(MAX_PITCH)
        notes = self.listener.notes
        if you_play:
            at, since = self._seconds(tick), self._seconds(self.heard_at - RECENT)
            playing = [p for o, d, p in notes if o >= since or o + d >= at or (o, p) in self.listener.held]
            yours = hands(playing)
            for lowest, highest in yours:
                bias[max(lowest - AROUND, 0):highest + AROUND + 1] = -self.apart
            if yours and yours[-1][1] >= TOP:   # the melody is yours: it does not comp over it
                bias[yours[-1][1] + AROUND + 1:] = -self.apart
            low, high = register
            return bias * min(int((bias[low:high + 1] == 0).sum()) / ROOM, 1)
        if not right:
            return bias
        melody = sorted(p for _, _, p in notes if p >= SPLIT)
        middle = min(max(melody[len(melody) // 2] if melody else 0, MELODY[0]), MELODY[1])
        return -self.lead * ((middle - PITCHES) / 12).clamp(0, 1)

    def _logits(self):
        inp = torch.tensor([self.seq[self.fed:]], device=self.device)
        out = self.model(inp, past_key_values=self.past, use_cache=True)
        self.past, self.fed = out.past_key_values, len(self.seq)
        return safe_logits(out.logits[0, -1].float() / self.temperature, len(self.seq) - 1)

    def _last_time(self):
        return self.seq[-3] - TIME_OFFSET if len(self.seq) > 1 else 0

    def _my_line(self, instrument=None):
        """(time, note token) of the last LINE notes of one of its hands (its right by default), oldest first."""
        instrument = self.me if instrument is None else instrument
        line, complete = [], len(self.seq) - (len(self.seq) - 1) % 3   # skip an event still being sampled
        for i in range(complete - 3, 0, -3):
            if self._instrument(self.seq[i + 2]) == instrument:
                line.append((self.seq[i] - TIME_OFFSET, self.seq[i + 2]))
                if len(line) == LINE:
                    break
        return line[::-1]

    def _banned(self, line, tick):
        """Notes the model may not play at `tick`: one it is already playing there, or one that repeats a run."""
        banned = {note for time_, note in line if time_ == tick}   # Lakh MIDI is full of doubled notes
        notes = [note for _, note in line]
        run = notes[len(notes) - self.no_repeat + 1:]
        for i in range(len(notes) - self.no_repeat + 1):
            if notes[i:i + self.no_repeat - 1] == run:
                banned.add(notes[i + self.no_repeat - 1])
        return banned

    def _unwind(self, tick):
        """Takes out of the context the events that start after `tick`; returns them, oldest first."""
        later = []
        while len(self.seq) > 1 and self._last_time() > tick:
            later.insert(0, self.seq[-3:])
            del self.seq[-3:]
        self._crop(len(self.seq))
        return later

    def _crop(self, length):
        """The KV cache keeps at most seq[:length]; the rest is fed again."""
        if self.fed > length:
            self.past.crop(length)
            self.fed = length

    def _fit(self):
        """Keeps the context under CONTEXT_SIZE tokens and its times under MAX_TIME."""
        now = self.now()
        if len(self.seq) + 3 <= CONTEXT_SIZE and now + LOOKAHEAD + LONGEST_REST < MAX_TIME:
            return
        self._restart([e for e in self._events() if e[0] - TIME_OFFSET >= now - WINDOW][-(KEEP // 3):], now)

    def _events(self):
        """Complete events in the context, leaving out one still being sampled."""
        return [self.seq[i:i + 3] for i in range(1, len(self.seq) - 2, 3)]

    def _restart(self, events, now=None):
        """A new context from `events`, moved to start at time 0 as the model was trained; the cache is rebuilt."""
        shift = events[0][0] - TIME_OFFSET if events else (now if now is not None else self.now())
        self.base += shift
        if self.heard_at is not None:
            self.heard_at -= shift
        self.seq = [AUTOREGRESS] + [token for t, d, n in events for token in (t - shift, d, n)]
        self.past, self.fed = None, 0

    def _mine(self):
        """The instruments of its hands in the context (both, even while it plays with one: it played with two)."""
        return {self.me, self.left}

    @staticmethod
    def _instrument(note):
        return (note - NOTE_OFFSET) // MAX_PITCH


def _duration(seconds):
    """Your note's duration as the model hears it: in ticks, from one to its longest."""
    return min(max(round(seconds * TIME_RESOLUTION), 1), MAX_DUR - 1)
