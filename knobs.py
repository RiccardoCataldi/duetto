"""
The controls beside the piano roll. On top, always in sight, the player, as in iReal Pro: Play or Stop, the Tune and
where it is, a light on each beat of the bar (amber through the count-in), the tempo and the metronome. Below, two
tabs:
- Play: the Tune, its Style (swing, bossa, even 8ths: yours to choose, whatever the tune), the Part it plays; the
  Character (density, harmony, role...), each control turning some of the model's Knobs (see
  music_transformer/server.py, "knobs"); the octaves it may play
- Sound: the band (its solo, its chords and its bass, each on an instrument of its own: a preset, or one by one) and
  the sound of each and of your piano in FluidSynth on this laptop (sound.py)
Drag a slider (Shift: finely), click a stepper's arrows, press the arrow keys or turn the wheel with Ctrl over
either; a double click puts it back. The wheel scrolls the tab.
"""
import json
import math
import os
import socket
import threading
import time

import pygame
import pygame.gfxdraw

import sound
from theme import AI, BASS, CHORDS, FAINT, HOVER, LINE, MUTED, RAISED, SELECTED, SURFACE, TEXT, YOU, capital, fit, mix, \
    text, wrap

PANEL_W = 380
PAD = 20                  # the panel's margins
PLAYER_H = 206            # the player, on top
HEADER_H = 56             # the tabs, under it
FOOTER_H = 96             # what the control under the mouse does
TITLE_H = 40              # a section's title
SLIDER_H = 52
MACRO_H = 66              # a musical control: a slider with its two ends under it
STEPPER_H = 70
CARD_H = 40               # Play, the metronome, a Part
ARROW_W = 30              # a stepper's arrows
THUMB_R = 7
DRAG = 220                # pixels of drag across a stepper's whole range
STEPS = 50                # wheel steps across a knob's range
DOUBLE_CLICK_S = 0.35
STEP_KEYS = {pygame.K_LEFT: -1, pygame.K_DOWN: -1, pygame.K_RIGHT: 1, pygame.K_UP: 1}
# the model's knobs as last turned
MIND_FILE = os.path.expanduser('~/.config/ai-duet/knobs.json')

# the musical controls: label, its two ends, what it does, and the model's knobs it turns, each at (0, middle, 1);
# the middle is the duo's default
MACROS = [
    ('density', ('sparse', 'dense'), "how many notes it plays",
     {'min_step': (60, 20, 8), 'max_rest': (500, 200, 100), 'phrase': (6, 16, 40), 'breath': (300, 150, 50)}),
    ('harmony', ('inside', 'outside'), "how far it strays from the scale of the bar's chord, and how it colours the chord",
     {'out_of_key': (5.0, 2.0, 0.0), 'chord_tone': (3.0, 1.0, 0.0), 'tension': (0.0, 1.0, 3.0)}),
    ('role', ('accompanies', 'leads'), "while you play: it keeps apart from your hands and under your melody, or goes "
                                       "anywhere; when you rest: it comps on, or takes the melody, up where yours was",
     {'apart': (4.0, 2.0, 0.0), 'lead': (0.0, 3.0, 6.0)}),
    ('dialogue', ('ignores you', 'answers you'), "when your phrase ends: how soon and how closely it answers it",
     {'answer': (0, 400, 1000), 'echo': (0.0, 3.0, 6.0)}),
    ('groove', ('loose', 'tight'), "how closely its notes sit on the beat",
     {'grid': (20, 2, 0)}),
    ('risk', ('safe', 'adventurous'), "how surprising its choices are",
     {'temperature': (0.7, 1.0, 1.4), 'top_p': (0.85, 0.98, 1.0)}),
    ('focus', ('on you', 'on itself'), "whose playing it builds on: yours, or its own line too",
     {'self_window': (50, 200, 2000)}),
    ('memory', ('the soloists', 'you'), "whose phrases it quotes: the great soloists', or the ones you played over the tune",
     {'recall': (0.0, 4.0, 10.0)}),
    ('key follow', ('steady', 'quick'), "how soon it leaves the tune's chord for one you hold",
     {'stay': (0.995, 0.97, 0.85)}),
]
# the Parts over the Tune: name, the value of the knob `part`, what it plays
PARTS = [('comps', 0, "it comps, with a walking bass, for you to solo"),
         ('duet', 1, "it comps while you play and solos when you rest"),
         ('solos', 2, "it solos as a jazz player does, for you to comp")]
# the Styles, in the order of the knob `style` (music_transformer/chart.py, STYLES): name, how it plays any tune
STYLES = [('swing', "swung eighths over a walking bass"),
          ('bossa', "even eighths over a bossa's bass: root and fifth on 1, the and of 2, 3, the and of 4"),
          ('even 8ths', "even eighths over a walking bass")]
SLOWEST = 40              # bpm the player's tempo goes down to
LIGHTS_H = 12             # the player's beat lights

# who plays, in the Sound tab: the channel (sound.py), who it is, its colour (the piano roll's)
VOICES = [(sound.YOU, 'you', YOU), (sound.AI, 'solo', AI), (sound.CHORDS, 'chords', CHORDS), (sound.BASS, 'bass', BASS)]


class Knob:
    """`turned(knob)` is called whenever what it shows changes (a musical control: whenever it moves)."""

    def __init__(self, name, value, low, high, default, unit, what, whole, turned, ends=None, title=None, accent=None):
        self.name, self.unit, self.what, self.whole, self.turned, self.ends = name, unit, what, whole, turned, ends
        self.title, self.accent = title, accent   # what it is called, if not its name; its colour, if not its tab's
        self.value, self.low, self.high, self.default = float(value), float(low), float(high), float(default)

    def set(self, value):
        before = self.shown()
        self.value = min(max(value, self.low), self.high)
        if self.shown() != before or self.unit == 'macro':
            self.turned(self)

    def step(self):
        """How far one turn of the wheel moves it."""
        if self.unit == 'program':
            return 1
        span = (self.high - self.low) / STEPS
        return max(1, round(span)) if self.whole else span

    def shown(self):
        value = round(self.value) if self.whole else self.value
        if self.unit == 'macro':
            return self.ends[0] if value < 0.4 else self.ends[1] if value > 0.6 else 'balanced'
        if self.unit == 'ticks':
            return f"{value / 100:.2f} s"
        if self.unit == 'notes':
            return f"{value} {self.unit}"
        if self.unit == 'program':
            return sound.GM[value]
        if self.unit == 'bpm':
            return f"{value} bpm" if value else "the tune's"
        if self.unit.startswith('choice:'):
            return self.unit[7:].split('/')[value - round(self.low)].replace('_', ' ')
        if self.unit in ('octave_from', 'octave_to'):
            return f"{'C' if self.unit == 'octave_from' else 'B'}{value}"
        if self.unit == 'pan':
            side = round((value - 64) / 63 * 100)
            return 'centre' if side == 0 else f"{'L' if side < 0 else 'R'} {abs(side)}%"
        if self.unit == 'ms':
            return f"{value} ms"
        if self.unit == 'times':
            return f"x{2 ** (value / 1200):.2f}"
        if self.unit == 'level':   # a MIDI level out of 127, or a share
            return f"{value / 127:.0%}" if self.whole else f"{value:.0%}"
        if self.unit == 'amount':
            return f"{(value - self.low) / (self.high - self.low):.0%}"
        return f"{value:.3f}" if self.high - self.low < 0.5 else f"{value:.2f}"

    def label(self):
        return self.title or ('tune' if self.name == 'chart' else self.name.replace('_', ' '))

    def share(self):
        return (self.value - self.low) / (self.high - self.low)


def along(anchors, share):
    """The value at `share` (0 to 1) of a line through anchors at 0, 0.5 and 1."""
    a, b, c = anchors
    return a + (b - a) * share * 2 if share <= 0.5 else b + (c - b) * (share - 0.5) * 2


class Mind:
    """The model's knobs over the knobs connection to the AMT server, and what it says it is doing.
    They are kept in MIND_FILE and given back to the server each time it connects; it reconnects whenever lost
    (with no `address`, it does not connect: its knobs are given to it)."""

    def __init__(self, address, path=MIND_FILE):
        self.address, self.path = address, path
        self.knobs = {}           # name -> Knob, once connected
        self.state = None         # the last DuoSession.state() heard
        self.heard = 0.0          # when, in time.monotonic() seconds
        self.sock = None
        self.lock = threading.Lock()
        try:
            with open(path) as f:
                self.saved = json.load(f)
        except (OSError, ValueError):
            self.saved = {}
        self.macros = [Knob(label, 0.5, 0, 1, 0.5, 'macro', what, False, self._macro, ends)
                       for label, ends, what, _ in MACROS]
        self.turns = {label: turns for label, _, _, turns in MACROS}
        for macro in self.macros:
            macro.value = self._saved('macros').get(macro.name, 0.5)
        if address:
            threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        host, port = self.address.rsplit(':', 1)
        while True:
            try:
                sock = socket.create_connection((host, int(port)), timeout=3)
                sock.sendall(b"knobs\n")
                lines = sock.makefile('r')
                knobs = {}
                for line in lines:
                    if line.strip() == 'ready':
                        break
                    _, name, value, low, high, default, unit, what = line.split(maxsplit=7)
                    knobs[name] = Knob(name, value, low, high, default, unit.strip('-'), what.strip(),
                                       '.' not in default, self.send)
                sock.settimeout(None)
                with self.lock:
                    self.knobs, self.sock = knobs, sock
                self._restore(knobs)
                for line in lines:
                    if line.startswith('state '):
                        self.state, self.heard = json.loads(line[6:]), time.monotonic()
            except (OSError, ValueError):
                pass
            with self.lock:
                self.knobs, self.sock, self.state = {}, None, None
            time.sleep(2)

    def _restore(self, knobs):
        """Gives the server back its knobs as you left them; one you never turned starts where its control stands."""
        saved = self._saved('knobs')
        for name, knob in knobs.items():
            if name in saved:
                knob.value = saved[name]
                self.send(knob, keep=False)
        for macro in self.macros:
            for name, anchors in self.turns[macro.name].items():
                if name in knobs and name not in saved:   # not kept: it follows its control until you turn it
                    knobs[name].value = along(anchors, macro.value)
                    self.send(knobs[name], keep=False)

    def send(self, knob, keep=True):
        """Turns `knob` on the server and, unless `keep` is False, keeps it for next time."""
        with self.lock:
            sock = self.sock
        if sock is None:
            return
        try:
            sock.sendall(f"set {knob.name} {knob.value:.4f}\n".encode())
        except OSError:
            return
        if knob.name == 'chart' and 'playing' in self.knobs:
            self.knobs['playing'].set(0)   # another tune waits for Play
            if 'bpm' in self.knobs:
                self.knobs['bpm'].set(0)   # at its own tempo, until you fix another
        if keep and knob.name != 'playing':   # a tune never starts by itself
            self._saved('knobs')[knob.name] = knob.value
            self._save()
            if knob.name == 'chart' and 'metronome' in self.knobs:
                self.knobs['metronome'].set(1)   # a chart starts with its click: the knob says so, and turns it off

    def transport(self):
        """Plays the tune, or stops it."""
        if 'playing' in self.knobs:
            self.knobs['playing'].set(1 - self.knobs['playing'].value)

    def click(self):
        """Turns the metronome on, or off."""
        if 'metronome' in self.knobs:
            self.knobs['metronome'].set(1 - self.knobs['metronome'].value)

    def tempo(self, by=None):
        """Fixes the tempo `by` bpm faster (slower, when negative) than the one it keeps to; None: the tune's again."""
        if 'bpm' not in self.knobs:
            return
        bpm = self.knobs['bpm']
        if by is None:
            bpm.set(0)
            return
        now = bpm.value or (self.state or {}).get('bpm') or 120
        bpm.set(min(max(round(now) + by, SLOWEST), bpm.high))

    def beat(self):
        """(the beat of the bar it is on, from 0, and whether it is the count-in's), or None while the tune is
        stopped: from the last state heard, run on at the tempo since."""
        state = self.state or {}
        at = (state.get('chart') or {}).get('at')
        if at is None or not state.get('bpm'):
            return None
        at += (time.monotonic() - self.heard) * state['bpm'] / 60
        return int(at % 4), at < 0

    def _macro(self, macro):
        for name, anchors in self.turns[macro.name].items():
            if name in self.knobs:
                self.knobs[name].set(along(anchors, macro.value))
        self._saved('macros')[macro.name] = macro.value
        self._save()

    def _saved(self, what):
        """What you left turned: 'knobs' or 'macros'."""
        return self.saved.setdefault(what, {})

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.saved, f, indent=1)

    def readout(self):
        """Where the tune is, for the top of the piano roll: (what, as it stands) for its title and bar, its chord
        and the next, and its tempo; None while it is not connected."""
        if self.state is None:
            return None
        state = self.state
        return [('Tune', state['key'] or '—'), ('Chord', state['chord'] or '—'),
                ('Tempo', f"{state['bpm']} bpm" if state['bpm'] else '—')]


def sound_knobs(engine, player, who, accent):
    """The knobs of `player`'s sound in `engine`, by name: `who` it is names its instrument, and all are its colour."""
    knobs = {}
    for name, (*_, low, high, unit, what) in sound.SOUND.items():
        default = sound.defaults()[player][name]
        knobs[name] = Knob(name, engine.sound[player][name], low, high, default, unit, what, isinstance(default, int),
                           lambda knob: engine.set(player, knob.name, knob.value),
                           title=who if name == 'instrument' else None, accent=accent)
    return knobs


class Panel:
    """The controls beside the piano roll: tabs on top, their controls in sections below (scrolled when they do not
    fit), and at the bottom what the control under the mouse does."""

    def __init__(self, address, engine, mind_file=MIND_FILE):
        self.mind = Mind(address, mind_file)
        self.address = address
        self.sounds = {player: sound_knobs(engine, player, who, accent) for player, who, accent in VOICES}
        # title, colour, its sections (title, knobs or cards) once there are knobs
        self.tabs = [('Play', AI, self._play_sections), ('Sound', YOU, self._sound_sections)]
        self.tab = 0
        self.scroll = [0] * len(self.tabs)
        self.held = None          # (knob, x where the drag started, value then, pixels across its range)
        self.hover = None         # the target under the mouse: (kind, what, rect)
        self.last_click = (None, 0)
        self.targets = []         # (kind, what, rect) of what can be clicked, from the last draw
        self.content = pygame.Rect(0, HEADER_H, PANEL_W, 0)
        self.overflow = 0         # how far the content of the tab shown can scroll

    # what each tab holds

    def _play_sections(self):
        knobs = self.mind.knobs
        if not knobs:
            return None
        part, style = knobs['part'], knobs['style']
        return [('', [knobs['chart']]),
                ('Style', [(name, capital(what), style.value == value, lambda value=value: style.set(value))
                           for value, (name, what) in enumerate(STYLES)]),
                ('Part', [(name, what, part.value == value, lambda value=value: part.set(value))
                          for name, value, what in PARTS]),
                ('Character', self.mind.macros),
                ('Octaves', [knobs['lowest'], knobs['highest']])]

    def _sound_sections(self):
        """The band: a preset (sound.BANDS) or an instrument for each of its voices, and yours; then each one's
        sound."""
        band = [self.sounds[player]['instrument'] for player in (sound.AI, sound.CHORDS, sound.BASS)]

        def choose(programs):
            for knob, program in zip(band, programs):
                knob.set(program)

        cards = [(name, 'Its solo on {}, its chords on {}, its bass on {}'.format(*(sound.GM[p] for p in programs)),
                  tuple(round(knob.value) for knob in band) == programs, lambda programs=programs: choose(programs))
                 for name, programs in sound.BANDS]
        return [('Band', cards), ('Instruments', band + [self.sounds[sound.YOU]['instrument']])] + [
            (f"{who} sound" if player else 'Your sound', [knob for name, knob in self.sounds[player].items()
                                                           if name != 'instrument']) for player, who, _ in VOICES]

    def _rows(self, sections):
        """(kind, what, height) of each row of the tab: a section's title, cards, a slider or steppers side by
        side."""
        rows = []
        for title, knobs in sections:
            rows.append(('title', title, TITLE_H if title else 12))
            if knobs and isinstance(knobs[0], tuple):   # cards: (name, what, chosen, choose)
                rows.append(('cards', knobs, math.ceil(len(knobs) / across(knobs)) * (CARD_H + 8)))
                continue
            pair = []
            for knob in knobs:
                if stepped(knob) and knob.unit != 'program':
                    pair.append(knob)
                    if len(pair) == 2:
                        rows.append(('steppers', pair, STEPPER_H))
                        pair = []
                    continue
                if pair:
                    rows.append(('steppers', pair, STEPPER_H))
                    pair = []
                rows.append(('steppers', [knob], STEPPER_H) if stepped(knob) else
                            ('slider', knob, MACRO_H if knob.unit == 'macro' else SLIDER_H))
            if pair:
                rows.append(('steppers', pair, STEPPER_H))
        return rows

    # the mouse and the keys

    def handle(self, event, origin):
        """Mouse and key events, at `origin` the panel's top left corner on screen."""
        if event.type == pygame.MOUSEWHEEL:
            x, y = pygame.mouse.get_pos()
            target = self._at((x - origin[0], y - origin[1]))
            if pygame.key.get_mods() & pygame.KMOD_CTRL:   # only on purpose: scrolling past a control leaves it be
                if target and target[0] in ('slider', 'stepper'):
                    target[1].set(target[1].value + event.y * target[1].step())
            elif target or self.content.collidepoint(x - origin[0], y - origin[1]):
                self._scroll(-event.y * 48)
            return
        if event.type == pygame.KEYDOWN:
            if self.hover and self.hover[0] in ('slider', 'stepper') and event.key in STEP_KEYS:
                knob = self.hover[1]
                knob.set(knob.value + STEP_KEYS[event.key] * knob.step())
            elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                self._scroll((1 if event.key == pygame.K_PAGEDOWN else -1) * self.content.height * 0.8)
            return
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return
        pos = (event.pos[0] - origin[0], event.pos[1] - origin[1])
        if event.type == pygame.MOUSEMOTION:
            self.hover = self._at(pos)
            if self.held:
                knob, x, start, span = self.held
                fine = 5 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
                knob.set(start + (pos[0] - x) / (span * fine) * (knob.high - knob.low))
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.held = None
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._click(pos)

    def _click(self, pos):
        target = self._at(pos)
        if target is None:
            return
        kind, what, rect = target
        if kind == 'tab':
            self.tab, self.hover, self.held = what, None, None
            return
        if kind == 'card':
            what[3]()
            return
        knob = what
        if kind == 'stepper' and pos[0] < rect.x + ARROW_W:
            knob.set(knob.value - knob.step())
            return
        if kind == 'stepper' and pos[0] > rect.right - ARROW_W:
            knob.set(knob.value + knob.step())
            return
        name, at = self.last_click
        if name == knob.name and time.monotonic() - at < DOUBLE_CLICK_S:
            knob.set(knob.default)
            self.last_click = (None, 0)
            return
        self.last_click = (knob.name, time.monotonic())
        if kind == 'slider':
            track = rect.inflate(-2 * THUMB_R, 0)
            thumb = track.x + track.width * knob.share()
            if abs(pos[0] - thumb) > THUMB_R + 2:   # off the thumb: it jumps there first
                knob.set(knob.low + (pos[0] - track.x) / track.width * (knob.high - knob.low))
            self.held = (knob, pos[0], knob.value, track.width)
        else:
            self.held = (knob, pos[0], knob.value, DRAG)

    def _at(self, pos):
        return next((t for t in self.targets if t[2].collidepoint(pos)), None)

    def _scroll(self, by):
        self.scroll[self.tab] = round(min(max(self.scroll[self.tab] + by, 0), self.overflow))

    # drawing

    def draw(self, surface):
        width, height = surface.get_size()
        surface.fill(SURFACE)
        pygame.draw.line(surface, LINE, (0, 0), (0, height))
        self.targets = []
        self._player(surface, width)
        self._tabs(surface, width)
        self.content = pygame.Rect(0, PLAYER_H + HEADER_H, width, height - PLAYER_H - HEADER_H - FOOTER_H)
        _, accent, sections = self.tabs[self.tab]
        sections = sections()
        if sections is None:
            self.overflow = 0
            self._waiting(surface, accent)
        else:
            self._sections(surface, self._rows(sections), accent, width)
        self._footer(surface, width, height)

    def _player(self, surface, width):
        """The player, as in iReal Pro: Play or Stop, the Tune and where it is, the beat lights, the tempo and the
        metronome; it is always in sight, whichever the tab."""
        knobs, state = self.mind.knobs, self.mind.state or {}
        ready = all(name in knobs for name in ('playing', 'metronome', 'bpm'))
        playing = ready and bool(knobs['playing'].value)
        # Play or Stop, large
        button = pygame.Rect(PAD, 20, 64, 64)
        play = ('Stop' if playing else 'Play', 'Space bar: the tune from a bar of count-in, or stop it', playing,
                self.mind.transport)
        over = ready and self._over('Play') or self._over('Stop')
        colour = (mix(AI, TEXT, 0.25) if over else AI) if ready else RAISED
        pygame.gfxdraw.aacircle(surface, button.centerx, button.centery, 32, colour)
        pygame.gfxdraw.filled_circle(surface, button.centerx, button.centery, 32, colour)
        x, y = button.center
        icon = SURFACE if ready else FAINT
        if playing:
            pygame.draw.rect(surface, icon, (x - 10, y - 10, 20, 20), border_radius=3)
        else:
            triangle = [(x - 8, y - 13), (x + 14, y), (x - 8, y + 13)]
            pygame.gfxdraw.aapolygon(surface, triangle, icon)
            pygame.gfxdraw.filled_polygon(surface, triangle, icon)
        if ready:
            self.targets.append(('card', play, button))
        # the Tune and where it is
        left = button.right + 18
        chart = state.get('chart') or {}
        title = chart.get('title') or ('Connecting to the model' if not knobs else '—')
        text(surface, fit(title, 18, width - PAD - left, True), 18, TEXT, (left, 26), bold=True)
        beat = self.mind.beat()
        if not ready:
            where = f"AMT server at {self.address}"
        elif not playing:
            where = 'Stopped · Space to play'
        elif chart.get('count'):
            where = f"Count-in · {chart['count']}"
        elif chart.get('bar', -1) >= 0:
            where = f"Bar {chart['bar'] + 1} of {len(chart['bars'])} · {state.get('chord') or ''}"
        else:
            where = 'Starting…'
        text(surface, fit(where, 13, width - PAD - left), 13, MUTED, (left, 54))
        # a light on each beat of the bar: the one it is on lit, the first a little wider; amber through the count-in
        lights, gap = pygame.Rect(PAD, 102, width - 2 * PAD, LIGHTS_H), 8
        each = (lights.width - 3 * gap) / 5
        for k in range(4):
            w = each * (2 if k == 0 else 1)
            light = pygame.Rect(round(lights.x + (each * (k + 1 if k else 0)) + gap * k), lights.y, round(w), LIGHTS_H)
            lit = beat is not None and beat[0] == k
            pygame.draw.rect(surface, (YOU if beat[1] else AI) if lit else RAISED, light, border_radius=LIGHTS_H // 2)
        # the tempo, its arrows a bpm slower or faster (Shift: five), and the metronome
        y = 132
        text(surface, 'TEMPO', 11, MUTED, (PAD, y), bold=True)
        fixed = ready and knobs['bpm'].value > 0
        if ready:
            text(surface, 'fixed · click to reset' if fixed else "the tune's", 11, FAINT, (PAD + 52, y))
        text(surface, 'METRONOME', 11, MUTED, (width - PAD, y), bold=True, anchor='topright')
        tempo = pygame.Rect(PAD, y + 18, round((width - 2 * PAD) * 0.6), 40)
        bpm = state.get('bpm')
        pygame.draw.rect(surface, RAISED, tempo, border_radius=10)
        shift = lambda: 5 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
        for side, name, what, turn in ((-1, 'Slower', 'One bpm slower (Shift: five)', lambda: self.mind.tempo(-shift())),
                                       (1, 'Faster', 'One bpm faster (Shift: five)', lambda: self.mind.tempo(shift()))):
            arrow = pygame.Rect(tempo.x if side < 0 else tempo.right - 44, tempo.y, 44, tempo.height)
            if ready and self._over(name):
                pygame.draw.rect(surface, HOVER, arrow, border_radius=10)
            chevron(surface, arrow.center, side, (TEXT if self._over(name) else MUTED) if ready else FAINT)
            if ready:
                self.targets.append(('card', (name, what, False, turn), arrow))
        middle = tempo.inflate(-88, 0)
        if ready:
            self.targets.append(('card', ('Tempo', "Its tempo: the tune's, or one you fix with the arrows; a click "
                                          "here gives it back the tune's", False, self.mind.tempo), middle))
        number = text(surface, str(round(bpm)) if bpm else '—', 20, AI if fixed else TEXT,
                      (middle.centerx - 14, tempo.centery), bold=True, anchor='center')
        text(surface, 'bpm', 12, MUTED, (number.right + 5, tempo.centery + 2), anchor='midleft')
        clicks = ready and bool(knobs['metronome'].value)
        toggle = pygame.Rect(tempo.right + 10, tempo.y, width - PAD - tempo.right - 10, tempo.height)
        if clicks:
            pygame.draw.rect(surface, mix(AI, SURFACE, 0.8), toggle, border_radius=10)
            pygame.draw.rect(surface, AI, toggle, 1, border_radius=10)
        else:
            pygame.draw.rect(surface, HOVER if self._over('Metronome') else RAISED, toggle, border_radius=10)
        pygame.draw.circle(surface, AI if clicks else FAINT, (toggle.x + 18, toggle.centery), 5)
        text(surface, 'On' if clicks else 'Off', 15, TEXT if clicks else MUTED, (toggle.x + 32, toggle.centery),
             bold=clicks, anchor='midleft')
        if ready:
            self.targets.append(('card', ('Metronome', 'M: a click on each beat while the tune plays', clicks,
                                          self.mind.click), toggle))
        pygame.draw.line(surface, LINE, (1, PLAYER_H - 1), (width, PLAYER_H - 1))

    def _over(self, name):
        """Whether the mouse is over the card called `name`."""
        return self.hover is not None and self.hover[0] == 'card' and self.hover[1][0] == name

    def _tabs(self, surface, width):
        well = pygame.Rect(PAD, PLAYER_H + 12, width - 2 * PAD, 34)
        pygame.draw.rect(surface, RAISED, well, border_radius=10)
        each = well.width / len(self.tabs)
        for i, (title, accent, *_) in enumerate(self.tabs):
            rect = pygame.Rect(round(well.x + i * each) + 3, well.y + 3, round(each) - 6, well.height - 6)
            chosen = i == self.tab
            if chosen:
                pygame.draw.rect(surface, SELECTED, rect, border_radius=8)
                pygame.draw.line(surface, accent, (rect.centerx - 10, rect.bottom - 3), (rect.centerx + 10,
                                 rect.bottom - 3), 2)
            elif self.hover and self.hover[:2] == ('tab', i):
                pygame.draw.rect(surface, HOVER, rect, border_radius=8)
            text(surface, fit(title, 13, rect.width - 8), 13, TEXT if chosen else MUTED, rect.center,
                 anchor='center')
            self.targets.append(('tab', i, rect))

    def _waiting(self, surface, accent):
        """While the model is not connected: what we wait for, and how to start it."""
        x, y = self.content.centerx, self.content.y + self.content.height // 2 - 50
        turn = (time.monotonic() * 240) % 360
        arc(surface, (x, y), 15, 3, 360 - turn + 90, 360 - turn - 160, accent)
        text(surface, 'Connecting to the model', 16, TEXT, (x, y + 34), bold=True, anchor='midtop')
        text(surface, f"AMT server at {self.address}", 13, MUTED, (x, y + 60), anchor='midtop')
        text(surface, 'Start it with ./run_duet.sh: it reconnects on its own', 12, FAINT, (x, y + 82),
             anchor='midtop')

    def _sections(self, surface, rows, accent, width):
        placed = len(self.targets)   # the player's and the tabs': always in sight
        total = sum(h for *_, h in rows) + 16
        self.overflow = max(0, total - self.content.height)
        self.scroll[self.tab] = min(self.scroll[self.tab], self.overflow)
        surface.set_clip(self.content)
        y = self.content.y - self.scroll[self.tab]
        for kind, what, h in rows:
            if y + h > self.content.y and y < self.content.bottom:
                if kind == 'title':
                    text(surface, what.upper(), 11, MUTED, (PAD, y + h - 20), bold=True)
                elif kind == 'cards':
                    self._cards(surface, what, y, accent, width)
                elif kind == 'slider':
                    self._slider(surface, what, y, accent, width)
                else:
                    self._steppers(surface, what, y, accent, width)
            y += h
        surface.set_clip(None)
        if self.overflow:
            self._scrollbar(surface, total, width)
            if self.scroll[self.tab] < self.overflow:   # more below: the content fades into the footer
                for i in range(18):
                    shade = pygame.Surface((width - 1, 1), pygame.SRCALPHA)
                    shade.fill((*SURFACE, round(255 * i / 18)))
                    surface.blit(shade, (1, self.content.bottom - 18 + i))
        # only what shows can be clicked
        self.targets[placed:] = [(kind, what, rect.clip(self.content)) for kind, what, rect in self.targets[placed:]
                                 if rect.colliderect(self.content)]

    def _scrollbar(self, surface, total, width):
        track = self.content.inflate(0, -12)
        h = max(30, track.height * self.content.height / total)
        y = track.y + (track.height - h) * self.scroll[self.tab] / self.overflow
        pygame.draw.rect(surface, LINE, (width - 7, round(y), 3, round(h)), border_radius=2)

    def _cards(self, surface, cards, y, accent, width):
        columns = across(cards)
        each = (width - 2 * PAD - 8 * (columns - 1)) / columns
        for i, card in enumerate(cards):
            name, _, chosen, _ = card
            rect = pygame.Rect(round(PAD + (i % columns) * (each + 8)), y + (i // columns) * (CARD_H + 8), round(each),
                               CARD_H)
            over = self._over(name)
            if chosen:
                pygame.draw.rect(surface, mix(accent, SURFACE, 0.8), rect, border_radius=10)
                pygame.draw.rect(surface, accent, rect, 1, border_radius=10)
            else:
                pygame.draw.rect(surface, HOVER if over else RAISED, rect, border_radius=10)
            text(surface, fit(capital(name), 14, rect.width - 16, chosen), 14, TEXT if chosen or over else MUTED,
                 rect.center, bold=chosen, anchor='center')
            self.targets.append(('card', card, rect))

    def _slider(self, surface, knob, y, accent, width):
        accent = knob.accent or accent
        held = self.held is not None and self.held[0] is knob
        over = held or (self.hover is not None and self.hover[1] is knob)
        moved = abs(knob.value - knob.default) > (knob.high - knob.low) / 200
        text(surface, capital(knob.label()), 14, TEXT, (PAD, y + 6))
        value = capital(knob.shown()) if knob.unit == 'macro' else knob.shown()
        text(surface, value, 13, accent if moved else MUTED, (width - PAD, y + 7), anchor='topright')
        track = pygame.Rect(PAD + THUMB_R, y + 34, width - 2 * PAD - 2 * THUMB_R, 4)
        pygame.draw.rect(surface, LINE, track, border_radius=2)
        at = track.x + track.width * knob.share()
        home = track.x + track.width * (knob.default - knob.low) / (knob.high - knob.low)
        if abs(at - home) >= 1:   # from where it stands by default to where it is
            pygame.draw.rect(surface, accent, (round(min(at, home)), track.y, round(abs(at - home)), track.height),
                             border_radius=2)
        if 0 < knob.default - knob.low < knob.high - knob.low:
            pygame.draw.line(surface, MUTED, (round(home), track.y - 4), (round(home), track.bottom + 3), 1)
        r = THUMB_R + (1 if over else 0)
        pygame.gfxdraw.aacircle(surface, round(at), track.centery, r, TEXT)
        pygame.gfxdraw.filled_circle(surface, round(at), track.centery, r, TEXT)
        if over:
            pygame.gfxdraw.aacircle(surface, round(at), track.centery, r + 3, accent)
        if knob.unit == 'macro':
            text(surface, knob.ends[0], 11, FAINT, (PAD, track.bottom + 10))
            text(surface, knob.ends[1], 11, FAINT, (width - PAD, track.bottom + 10), anchor='topright')
        self.targets.append(('slider', knob, pygame.Rect(PAD, track.y - 12, width - 2 * PAD, track.height + 24)))

    def _steppers(self, surface, knobs, y, accent, width):
        each = (width - 2 * PAD - 12) / len(knobs)
        for i, knob in enumerate(knobs):
            accent = knob.accent or accent
            x = round(PAD + i * (each + 12))
            over = self.hover is not None and self.hover[1] is knob
            label = text(surface, capital(knob.label()), 12, MUTED, (x, y + 6))
            if knob.title and knob.accent:   # whose instrument: its colour in the piano roll
                pygame.draw.circle(surface, knob.accent, (label.right + 9, label.centery + 1), 3)
            box = pygame.Rect(x, y + 25, round(each), 34)
            pygame.draw.rect(surface, HOVER if over else RAISED, box, border_radius=8)
            moved = abs(knob.value - knob.default) > 0.01
            value = fit(capital(knob.shown()), 14, box.width - 2 * ARROW_W, moved)
            text(surface, value, 14, accent if moved else TEXT, box.center, bold=moved, anchor='center')
            for side in (-1, 1):
                end = knob.low if side < 0 else knob.high
                can = abs(knob.value - end) > 1e-6
                chevron(surface, (box.x + ARROW_W // 2 if side < 0 else box.right - ARROW_W // 2, box.centery), side,
                        (TEXT if over else MUTED) if can else FAINT)
            self.targets.append(('stepper', knob, box))

    def _footer(self, surface, width, height):
        y = height - FOOTER_H
        pygame.draw.line(surface, LINE, (1, y), (width, y))
        target = (self.held[0] if self.held else None) or (self.hover[1] if self.hover and self.hover[0] in
                                                           ('slider', 'stepper') else None)
        if target is not None:
            title, what = capital(target.label()), self._explain(target)
        elif self.hover and self.hover[0] == 'card':
            title, what = capital(self.hover[1][0]), capital(self.hover[1][1])
        else:
            text(surface, 'Space: play or stop · M: metronome · K: hide', 12, FAINT, (PAD, height - 30))
            return
        text(surface, title, 14, TEXT, (PAD, y + 14), bold=True)
        for i, line in enumerate(wrap(what, 12, width - 2 * PAD)[:3]):
            text(surface, line, 12, MUTED, (PAD, y + 38 + i * 17))

    def _explain(self, knob):
        if knob.unit != 'macro':
            return capital(knob.what)
        turns = ', '.join(name.replace('_', ' ') for name in self.mind.turns[knob.name])
        return f"From {knob.ends[0]} to {knob.ends[1]}: {knob.what} (turns {turns})"


def across(cards):
    """How many cards side by side: three in a row of three, two otherwise."""
    return 3 if len(cards) == 3 else 2


def stepped(knob):
    """Whether `knob` takes a few values, one by one: a stepper rather than a slider."""
    return knob.unit in ('program', 'notes', 'bpm', 'octave_from', 'octave_to') or \
        knob.unit.startswith('choice:')


def chevron(surface, centre, side, colour):
    x, y = centre
    points = [(x - 2 * side, y - 5), (x + 3 * side, y), (x - 2 * side, y + 5)]
    pygame.draw.lines(surface, colour, False, points, 2)


def arc(surface, centre, radius, width, start, end, colour):
    """A smooth thick arc from `start` to `end` degrees (counter-clockwise from 3 o'clock, `end` < `start`)."""
    steps = max(2, int(abs(start - end) / 3))
    points = lambda r: [(centre[0] + r * math.cos(math.radians(start + (end - start) * i / steps)),
                         centre[1] - r * math.sin(math.radians(start + (end - start) * i / steps)))
                        for i in range(steps + 1)]
    outline = points(radius + width / 2) + points(radius - width / 2)[::-1]
    pygame.gfxdraw.aapolygon(surface, outline, colour)
    pygame.gfxdraw.filled_polygon(surface, outline, colour)
