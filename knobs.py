"""
The controls beside the piano roll, in two tabs:
- Play: the Tune, Play or Stop with the metronome beside it, the Part it plays; the Character (density, harmony,
  role...), each control turning some of the model's Knobs (see music_transformer/server.py, "knobs"); the octaves
  it may play
- Sound: your piano's and its own in FluidSynth on this laptop (sound.py)
F2 opens Advanced: the model's Knobs, each on its own. Drag a
slider (Shift: finely), click a stepper's arrows, press the arrow keys or turn the wheel with Ctrl over either; a
double click puts it back. The wheel scrolls the tab.
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
from theme import AI, FAINT, HOVER, LINE, MUTED, RAISED, SELECTED, SURFACE, TEXT, YOU, capital, fit, mix, text, wrap

PANEL_W = 380
PAD = 20                  # the panel's margins
HEADER_H = 68             # the tabs
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
         ('solos', 2, "it solos as a jazz player does, for you to comp"),
         ('duet', 1, "it comps while you play and solos when you rest"),
         ('trade fours', 3, "four bars of your solo, then four of its own")]

# the sections of Advanced, and of each piano's sound
ADVANCED = [('Tune', ('chart', 'playing', 'part', 'licks', 'pickup', 'recall', 'bpm', 'metronome')),
            ('Timing', ('min_step', 'max_rest', 'run', 'answer', 'grid', 'phrase', 'breath', 'commit')),
            ('Harmony', ('out_of_key', 'chord_tone', 'tension', 'stay')),
            ('Texture', ('chord', 'no_repeat')),
            ('Role & dialogue', ('apart', 'lead', 'echo')),
            ('Imagination', ('temperature', 'top_p', 'self_window')),
            ('Range', ('lowest', 'highest'))]
SOUND_GROUPS = [('Mix', ('volume', 'pan', 'reverb')),
                ('Envelope', ('attack', 'decay', 'sustain', 'release')), ('Delay', ('echo', 'delay', 'feedback'))]


class Knob:
    """`turned(knob)` is called whenever what it shows changes (a musical control: whenever it moves)."""

    def __init__(self, name, value, low, high, default, unit, what, whole, turned, ends=None):
        self.name, self.unit, self.what, self.whole, self.turned, self.ends = name, unit, what, whole, turned, ends
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
        return 'tune' if self.name == 'chart' else self.name.replace('_', ' ')

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
                        self.state = json.loads(line[6:])
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


def sound_knobs(engine, player):
    """The knobs of `player`'s sound in `engine`."""
    knobs = []
    for name, (*defaults, low, high, unit, what) in sound.SOUND.items():
        default = defaults[player]
        knobs.append(Knob(name, engine.sound[player][name], low, high, default, unit, what, isinstance(default, int),
                          lambda knob: engine.set(player, knob.name, knob.value)))
    return knobs


class Panel:
    """The controls beside the piano roll: tabs on top, their controls in sections below (scrolled when they do not
    fit), and at the bottom what the control under the mouse does."""

    def __init__(self, address, engine, mind_file=MIND_FILE):
        self.mind = Mind(address, mind_file)
        self.address = address
        yours, its = sound_knobs(engine, sound.YOU), sound_knobs(engine, sound.AI)
        # title, colour, its sections (title, knobs or cards) once there are knobs; Advanced only while F2 shows it
        self.tabs = [('Play', AI, self._play_sections),
                     ('Sound', YOU, lambda: self._groups([('Your piano', ('instrument',)), *SOUND_GROUPS],
                                                         {k.name: k for k in yours}) +
                                            self._groups([('Its piano', ('instrument',)), *SOUND_GROUPS],
                                                         {k.name: k for k in its})),
                     ('Advanced', AI, lambda: self._groups(ADVANCED, self.mind.knobs))]
        self.tab, self.advanced = 0, False
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
        part, playing, click = knobs['part'], knobs['playing'].value, knobs['metronome'].value
        return [('', [knobs['chart']]),
                ('', [('Stop' if playing else 'Play', 'Space bar: the tune from a bar of count-in, or stop it',
                       bool(playing), self.mind.transport),
                      ('Metronome', 'A click on each beat while the tune plays', bool(click), self.mind.click)]),
                ('Part', [(name, what, part.value == value, lambda value=value: part.set(value))
                          for name, value, what in PARTS]),
                ('Character', self.mind.macros),
                ('Octaves', [knobs['lowest'], knobs['highest']])]

    def _groups(self, groups, knobs):
        if not knobs:
            return None
        placed = {n for _, names in groups for n in names}
        sections = [(title, [knobs[n] for n in names if n in knobs]) for title, names in groups]
        sections.append(('Other', [k for n, k in knobs.items() if n not in placed]))
        return [(title, ks) for title, ks in sections if ks]

    def _rows(self, sections):
        """(kind, what, height) of each row of the tab: a section's title, cards, a slider or steppers side by
        side."""
        rows = []
        for title, knobs in sections:
            rows.append(('title', title, TITLE_H if title else 12))
            if knobs and isinstance(knobs[0], tuple):   # cards: (name, what, chosen, choose)
                rows.append(('cards', knobs, math.ceil(len(knobs) / 2) * (CARD_H + 8)))
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
            if event.key == pygame.K_F2:   # no key of the laptop's piano
                self.advanced = not self.advanced
                self.tab, self.hover, self.held = 2 if self.advanced else 0, None, None
            elif self.hover and self.hover[0] in ('slider', 'stepper') and event.key in STEP_KEYS:
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
        self._tabs(surface, width)
        self.content = pygame.Rect(0, HEADER_H, width, height - HEADER_H - FOOTER_H)
        _, accent, sections = self.tabs[self.tab]
        sections = sections()
        if sections is None:
            self.overflow = 0
            self._waiting(surface, accent)
        else:
            self._sections(surface, self._rows(sections), accent, width)
        self._footer(surface, width, height)

    def _tabs(self, surface, width):
        well = pygame.Rect(PAD, 18, width - 2 * PAD, 34)
        pygame.draw.rect(surface, RAISED, well, border_radius=10)
        tabs = self.tabs if self.advanced else self.tabs[:2]
        each = well.width / len(tabs)
        for i, (title, accent, *_) in enumerate(tabs):
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
            if kind == 'title':   # the Sound tab's pianos, each in its player's colour
                accent = {'Your piano': YOU, 'Its piano': AI}.get(what, accent)
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
        self.targets = [(kind, what, rect if kind == 'tab' else rect.clip(self.content))
                        for kind, what, rect in self.targets if kind == 'tab' or rect.colliderect(self.content)]

    def _scrollbar(self, surface, total, width):
        track = self.content.inflate(0, -12)
        h = max(30, track.height * self.content.height / total)
        y = track.y + (track.height - h) * self.scroll[self.tab] / self.overflow
        pygame.draw.rect(surface, LINE, (width - 7, round(y), 3, round(h)), border_radius=2)

    def _cards(self, surface, cards, y, accent, width):
        each = (width - 2 * PAD - 8) / 2
        for i, card in enumerate(cards):
            name, _, chosen, _ = card
            rect = pygame.Rect(round(PAD + (i % 2) * (each + 8)), y + (i // 2) * (CARD_H + 8), round(each), CARD_H)
            over = self.hover is not None and self.hover[0] == 'card' and self.hover[1][0] == name
            if chosen:
                pygame.draw.rect(surface, mix(accent, SURFACE, 0.8), rect, border_radius=10)
                pygame.draw.rect(surface, accent, rect, 1, border_radius=10)
            else:
                pygame.draw.rect(surface, HOVER if over else RAISED, rect, border_radius=10)
            text(surface, fit(capital(name), 14, rect.width - 16, chosen), 14, TEXT if chosen or over else MUTED,
                 rect.center, bold=chosen, anchor='center')
            self.targets.append(('card', card, rect))

    def _slider(self, surface, knob, y, accent, width):
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
            x = round(PAD + i * (each + 12))
            over = self.hover is not None and self.hover[1] is knob
            text(surface, capital(knob.label()), 12, MUTED, (x, y + 6))
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
            text(surface, f"F2: {'close' if self.advanced else 'open'} Advanced", 12, FAINT, (PAD, height - 30))
            return
        text(surface, title, 14, TEXT, (PAD, y + 14), bold=True)
        for i, line in enumerate(wrap(what, 12, width - 2 * PAD)[:3]):
            text(surface, line, 12, MUTED, (PAD, y + 38 + i * 17))

    def _explain(self, knob):
        if knob.unit != 'macro':
            return capital(knob.what)
        turns = ', '.join(name.replace('_', ' ') for name in self.mind.turns[knob.name])
        return f"From {knob.ends[0]} to {knob.ends[1]}: {knob.what} (turns {turns})"


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
