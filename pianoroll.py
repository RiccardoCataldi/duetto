"""
A piano roll of the duo: your notes and the model's rise from the keyboard as you play.

    python pianoroll.py                                   # live, next to main.py (F: fullscreen, K: knobs, Esc: quit)
    python pianoroll.py video duo.mid --audio duo.ogg -o duo.mp4

Live, it listens to what main.py sends to FluidSynth on Midi Through: channel 0 is you, channel 1 the model;
the knobs beside it (knobs.py) turn the model's on its server while you play.
"""
import argparse
import math
import os
import subprocess
import sys
import time

import mido
import pygame

import theme
from theme import AI, FAINT, MUTED, TEXT, YOU, mix

SIZE = (1280, 720)
FPS = 60                  # live
VIDEO_FPS = 30
RANGE = (21, 108)         # live: the 88 keys of a piano, A0 to C8 (its bass walks from E1); a video fits the octaves its notes use
KEYBOARD_H = 120
SPEED = 140               # pixels a second the notes rise
TAIL_S = 5                # seconds the video goes on after the last note, while the reverb dies away

HEADER_H = 96            # the players, what the model hears, and the time, over the notes' fade

BACKGROUND = theme.BACKGROUND
WHITE_KEY = (232, 234, 240)
BLACK_KEY = (28, 30, 38)
FELT = (112, 30, 44)      # the strip of red felt over the keys
GRID = (255, 255, 255, 12)

PLAYERS = {0: ('You', 'piano', YOU), 1: ('AI', 'piano', AI)}
KEYS = 'duetto keys'      # main.py's MIDI port for the laptop's keyboard
# the laptop's keyboard as a piano, as in Ableton Live: a row of white keys with the black ones above it, from C
NOTE_KEYS = {key: step for step, key in enumerate('awsedftgyhujkolp')}


def is_black(pitch):
    return pitch % 12 in (1, 3, 6, 8, 10)


class Keyboard:
    """Where each key is, and the keyboard drawn once."""

    def __init__(self, width, top, low, high):
        self.top = top
        whites = [p for p in range(low, high + 1) if not is_black(p)]
        self.white_w = width / len(whites)
        self.black_w = self.white_w * 0.6
        self.x = {}
        for i, p in enumerate(whites):
            self.x[p] = (i * self.white_w, self.white_w)
        for p in range(low, high + 1):
            if is_black(p):
                left, _ = self.x[p - 1]
                self.x[p] = (left + self.white_w - self.black_w / 2, self.black_w)
        self.surface = pygame.Surface((width, KEYBOARD_H), pygame.SRCALPHA)
        self._draw(self.surface, {})
        self.shadow = pygame.Surface((width, 12), pygame.SRCALPHA)
        for y in range(12):
            self.shadow.fill((0, 0, 0, round(110 * (1 - y / 12) ** 2)), pygame.Rect(0, y, width, 1))

    def _draw(self, surface, lit):
        black_h = KEYBOARD_H * 0.62
        for p, (x, w) in self.x.items():
            if not is_black(p):
                rect = pygame.Rect(round(x) + 1, 0, round(w) - 1, KEYBOARD_H - 2)
                pygame.draw.rect(surface, lit.get(p, WHITE_KEY), rect, border_bottom_left_radius=4,
                                 border_bottom_right_radius=4)
                if p % 12 == 0 and w >= 14:
                    theme.text(surface, f"C{p // 12 - 1}", 10, (70, 74, 90) if p in lit else (150, 154, 168),
                               (rect.centerx, rect.bottom - 6), anchor='midbottom')
        for p, (x, w) in self.x.items():
            if is_black(p):
                rect = pygame.Rect(round(x), 0, round(w), black_h)
                pygame.draw.rect(surface, lit.get(p, BLACK_KEY), rect, border_bottom_left_radius=3,
                                 border_bottom_right_radius=3)

    def draw(self, screen, lit):
        """`lit` maps the pitches sounding now to the colour of who plays them."""
        if not lit:
            screen.blit(self.surface, (0, self.top))
            return
        keys = self.surface.copy()
        self._draw(keys, {p: mix(colour, WHITE_KEY if not is_black(p) else BLACK_KEY, 0.15)
                          for p, colour in lit.items()})
        screen.blit(keys, (0, self.top))

    def finish(self, screen):
        """The felt over the keys and the shadow it casts on them."""
        screen.blit(self.shadow, (0, self.top))
        pygame.draw.rect(screen, FELT, (0, self.top - 1, screen.get_width(), 3))


class Roll:
    """Draws the notes, `(start, end or None, pitch, channel)` in seconds, as they are at `now`."""

    def __init__(self, size=SIZE, keys=RANGE):
        self.size = size
        width, height = size
        self.keyboard = Keyboard(width, height - KEYBOARD_H, *keys)
        self.background = vertical_gradient(size, *BACKGROUND)
        self.grid = pygame.Surface((width, height - KEYBOARD_H), pygame.SRCALPHA)
        for p, (x, _) in self.keyboard.x.items():
            if p % 12 == 0:
                pygame.draw.line(self.grid, GRID, (round(x), 0), (round(x), height - KEYBOARD_H))
        # the notes fade out as they rise to the top
        self.fade = pygame.Surface((width, height // 3), pygame.SRCALPHA)
        for y in range(height // 3):
            alpha = round(255 * (1 - y / (height // 3)) ** 2)
            self.fade.fill((*BACKGROUND[0], alpha), pygame.Rect(0, y, width, 1))
        self.glow = {c: glow_sprite(colour) for c, (_, _, colour) in PLAYERS.items()}
        self.instruments = {}     # channel -> the instrument it plays, if not PLAYERS'
        self.readout = None       # what the model hears, shown at the top: (what, as it stands)
        self.chart = None         # the tune it plays over: {'title', 'bars': [[chord, ...], ...], 'bar', 'beat'}
        self.offline = None       # or why it cannot say
        self.hint = None          # a key to press, in the corner
        self.header = pygame.Surface((width, HEADER_H), pygame.SRCALPHA)
        for y in range(HEADER_H):
            self.header.fill((*BACKGROUND[0], round(235 * (1 - y / HEADER_H) ** 1.5)), pygame.Rect(0, y, width, 1))

    def draw(self, screen, notes, now, clock=None):
        screen.blit(self.background, (0, 0))
        screen.blit(self.grid, (0, 0))
        top = self.keyboard.top
        lit = {}
        for start, end, pitch, channel in notes:
            if pitch not in self.keyboard.x or channel not in PLAYERS:
                continue
            colour = PLAYERS[channel][2]
            ended = now if end is None or end > now else end
            if start > now:
                continue
            y_top = top - (now - start) * SPEED
            y_bottom = top - (now - ended) * SPEED
            if y_bottom < 0:
                continue
            x, w = self.keyboard.x[pitch]
            inset = 2 if not is_black(pitch) else 1
            rect = pygame.Rect(round(x) + inset, round(y_top), max(2, round(w) - 2 * inset),
                               max(3, round(y_bottom - y_top)))
            pygame.draw.rect(screen, colour, rect, border_radius=4)
            pygame.draw.rect(screen, mix(colour, (255, 255, 255), 0.45), (rect.x, rect.y, rect.w, min(rect.h, 3)),
                             border_top_left_radius=4, border_top_right_radius=4)
            if ended == now:
                lit[pitch] = colour
                sprite = self.glow[channel]
                screen.blit(sprite, (round(x + w / 2 - sprite.get_width() / 2), top - sprite.get_height() // 2),
                            special_flags=pygame.BLEND_ADD)
        screen.blit(self.fade, (0, 0))
        self.keyboard.draw(screen, lit)
        self.keyboard.finish(screen)
        self._header(screen, clock)
        if self.chart:
            self._chart(screen)

    def _header(self, screen, clock):
        screen.blit(self.header, (0, 0))
        width = self.size[0]
        x = 28
        for channel, (name, instrument, colour) in PLAYERS.items():
            instrument = self.instruments.get(channel, instrument)
            pygame.draw.circle(screen, colour, (x + 5, 31), 5)
            label = theme.text(screen, name, 15, TEXT, (x + 18, 21), bold=True)
            sub = theme.text(screen, theme.fit(instrument, 12, 170), 12, MUTED, (x + 18, label.bottom + 1))
            x = max(label.right, sub.right) + 30
        right = width - 28
        if clock is not None:
            stamp = theme.text(screen, f"{int(clock) // 60}:{int(clock) % 60:02d}", 15, TEXT, (right, 21), bold=True,
                               anchor='topright')
            hint = theme.text(screen, self.hint or 'duetto', 12, MUTED, (right, stamp.bottom + 1), anchor='topright')
            right = min(right, hint.left + 74)   # a long hint takes room from the readout
        if self.readout:
            self._readout(screen, x, right - 90)
        elif self.offline:
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4)
            words = theme.render(self.offline, 13, MUTED)
            pill = pygame.Rect(0, 20, words.get_width() + 40, 30)
            pill.centerx = width // 2
            pygame.draw.rect(screen, theme.RAISED, pill, border_radius=15)
            pygame.draw.circle(screen, mix(theme.FAINT, AI, pulse), (pill.x + 16, pill.centery), 4)
            screen.blit(words, words.get_rect(midleft=(pill.x + 28, pill.centery)))

    def _chart(self, screen):
        """The tune as a page of a Real Book: its bars in rows of eight (four in a narrow window), the chords in
        each, the bar being played lit with its beats under it, and a double line where the tune ends."""
        bars, now = self.chart['bars'], self.chart['bar']
        if self.chart.get('count'):   # the count-in: the beats to go, large
            theme.text(screen, str(self.chart['count']), 64, AI, (self.size[0] // 2, self.keyboard.top - 120),
                       bold=True, anchor='center')
        width = self.size[0]
        across = 8 if width >= 900 else 4
        cell_w, cell_h = min((width - 56) // across, 150), 44
        rows = -(-len(bars) // across)
        left, top = (width - cell_w * across) // 2, HEADER_H - 14
        page = pygame.Surface((cell_w * across + 24, rows * cell_h + 20), pygame.SRCALPHA)
        pygame.draw.rect(page, (*theme.SURFACE, 225), page.get_rect(), border_radius=12)
        screen.blit(page, (left - 12, top - 10))
        for i, chords in enumerate(bars):
            cell = pygame.Rect(left + i % across * cell_w, top + i // across * cell_h, cell_w, cell_h)
            if i == now:
                pygame.draw.rect(screen, mix(AI, theme.SURFACE, 0.72), cell.inflate(-4, -4), border_radius=7)
                for beat in range(4):
                    dot = (cell.centerx - 15 + 10 * beat, cell.bottom - 8)
                    pygame.draw.circle(screen, AI if beat <= self.chart['beat'] else FAINT, dot, 2)
            pygame.draw.line(screen, theme.LINE, cell.topleft, (cell.x, cell.bottom - 1), 1)
            same = i % across and chords == bars[i - 1]   # the bar before again: a Real Book's repeat sign
            names = ['%'] if same and i != now else chords
            for k, name in enumerate(names):
                colour = TEXT if i == now else FAINT if same else MUTED
                room = cell_w / len(names)
                theme.text(screen, theme.fit(name, 15, room - 8, i == now), 15, colour,
                           (round(cell.x + room * (k + 0.5)), cell.centery - 3), bold=i == now, anchor='center')
        if self.chart.get('quote'):   # whose phrase it is playing
            theme.text(screen, theme.fit(f"quoting {self.chart['quote']}", 13, cell_w * across), 13, AI,
                       (self.size[0] // 2, top + rows * cell_h + 22), anchor='center')
        end = left + (len(bars) - 1) % across * cell_w + cell_w
        y = top + (rows - 1) * cell_h
        for x in (end - 4, end - 1):
            pygame.draw.line(screen, MUTED, (x, y + 4), (x, y + cell_h - 5), 1 if x < end - 1 else 2)

    def _readout(self, screen, left, right):
        """Your key, chord and tempo, or the Tune and its bar, as cells across the middle, as many as fit."""
        cells = []
        for caption, value in self.readout:
            value = theme.capital(value)
            cells.append((caption.upper(), value, max(theme.width(value, 17, True), 64)))
        gap = 36
        while cells and sum(w for *_, w in cells) + gap * (len(cells) - 1) > right - left:
            cells.pop(0)
        x = (self.size[0] - sum(w for *_, w in cells) - gap * (len(cells) - 1)) // 2
        x = min(max(x, left), right - sum(w for *_, w in cells) - gap * (len(cells) - 1))
        for i, (caption, value, w) in enumerate(cells):
            if i:
                pygame.draw.line(screen, theme.LINE, (x - gap // 2, 22), (x - gap // 2, 52))
            theme.text(screen, caption, 10, MUTED, (x, 19), bold=True)
            theme.text(screen, value, 17, TEXT, (x, 33), bold=True)
            x += w + gap


def vertical_gradient(size, top, bottom):
    surface = pygame.Surface(size, pygame.SRCALPHA)
    for y in range(size[1]):
        surface.fill(mix(top, bottom, y / max(1, size[1] - 1)), pygame.Rect(0, y, size[0], 1))
    return surface


def glow_sprite(colour, radius=34):
    sprite = pygame.Surface((radius * 2, radius * 2))
    for r in range(radius, 0, -1):
        t = (1 - r / radius) ** 2 * 0.5
        pygame.draw.circle(sprite, tuple(round(c * t) for c in colour), (radius, radius), r)
    return sprite


def octaves(notes):
    """The keys from the C below the lowest note to the C above the highest."""
    pitches = [n[2] for n in notes]
    return min(pitches) // 12 * 12, max(pitches) // 12 * 12 + 12


def notes_from_midi(path):
    """The notes of a saved duo, `(start, end, pitch, channel)` in seconds."""
    notes, held, now = [], {}, 0.0
    for message in mido.MidiFile(path):
        now += message.time
        if message.type == 'note_on' and message.velocity > 0:
            held[(message.channel, message.note)] = now
        elif message.type in ('note_on', 'note_off'):
            start = held.pop((message.channel, message.note), None)
            if start is not None:
                notes.append((start, now, message.note, message.channel))
    return notes


class Typing:
    """The laptop's keyboard as a piano, as in Ableton Live: A W S E D F T G Y H U J K O L P are the keys from C,
    Z and X go an octave down and up, C and V play softer and louder. `send` takes a MIDI message."""

    def __init__(self, send):
        self.send = send
        self.octave, self.velocity = 5, 100   # A is middle C
        self.down = {}                         # key -> the pitch it plays, while it is held

    def handles(self, key):
        return key in NOTE_KEYS or key in ('z', 'x', 'c', 'v')

    def key(self, key, pressed):
        if pressed and key in NOTE_KEYS and key not in self.down:
            self.down[key] = 12 * self.octave + NOTE_KEYS[key]
            self.send([0x90, self.down[key], self.velocity])
        elif not pressed and key in self.down:
            self.send([0x80, self.down.pop(key), 0])
        elif pressed and key in ('z', 'x'):
            self.octave = min(max(self.octave + (1 if key == 'x' else -1), 0), 9)
        elif pressed and key in ('c', 'v'):
            self.velocity = min(max(self.velocity + (20 if key == 'v' else -20), 20), 120)

    def release(self):
        for key in list(self.down):
            self.key(key, False)

    def says(self):
        return f"A W S E D…: piano · Z X: C{self.octave - 1} · C V: velocity {self.velocity}"


class Live:
    """The notes heard on a MIDI port, kept while they are on screen."""

    def __init__(self):
        self.notes = []          # [start, end or None, pitch, channel]
        self.held = {}           # (channel, pitch) -> note

    def message(self, message, now):
        if len(message) < 3:
            return
        kind, channel, pitch, velocity = message[0] & 0xF0, message[0] & 0x0F, message[1], message[2]
        if kind == 0x90 and velocity > 0:
            self.off(channel, pitch, now)
            note = [now, None, pitch, channel]
            self.held[(channel, pitch)] = note
            self.notes.append(note)
        elif kind in (0x80, 0x90):
            self.off(channel, pitch, now)

    def off(self, channel, pitch, now):
        note = self.held.pop((channel, pitch), None)
        if note is not None:
            note[1] = now

    def visible(self, now, height):
        self.notes = [n for n in self.notes if n[1] is None or (now - n[1]) * SPEED < height]
        return self.notes


def live(port_name, server):
    import rtmidi
    import sound
    from knobs import PANEL_W, Panel
    pygame.init()
    pygame.display.set_caption('duetto')
    screen = pygame.display.set_mode((SIZE[0] + PANEL_W, SIZE[1]), pygame.RESIZABLE)
    midi = rtmidi.MidiIn()
    ports = midi.get_ports()
    port = next((i for i, p in enumerate(ports) if port_name in p), None)
    if port is None:
        sys.exit(f"No MIDI port named {port_name!r} in {ports}")
    midi.open_port(port)
    out = rtmidi.MidiOut()
    out.open_port(next(i for i, p in enumerate(out.get_ports()) if port_name in p))
    engine = sound.Engine(out.send_message)   # your sound and the model's, and their echoes, on the same port
    heard = Live()

    def hear(event, data=None):
        now = time.monotonic()
        heard.message(event[0], now)
        engine.heard(event[0], now)

    midi.set_callback(hear)
    panel, shown = Panel(server, engine), True
    keys = rtmidi.MidiOut()   # the laptop's keyboard plays into main.py, as a MIDI keyboard would
    typing = Typing(keys.send_message)

    def plug():
        port = next((i for i, p in enumerate(keys.get_ports()) if KEYS in p), None)
        if not keys.is_port_open() and port is not None:   # main.py is playing
            keys.open_port(port)
        return keys.is_port_open()

    typed = plug()   # on whenever main.py plays: F and K are keys of the piano then, F11 and Tab stand in for them

    def layout():
        width, height = screen.get_size()
        side = PANEL_W if shown and width > 2 * PANEL_W else 0
        area = pygame.Rect(0, 0, width - side, height)
        return Roll(area.size), screen.subsurface(area), pygame.Rect(area.right, 0, side, height)

    (roll, view, side), clock, started = layout(), pygame.time.Clock(), time.monotonic()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return
            if not typed and event.type == pygame.KEYDOWN:
                typed = plug()   # main.py started after us
            if typed and event.type in (pygame.KEYDOWN, pygame.KEYUP) and typing.handles(pygame.key.name(event.key)):
                typing.key(pygame.key.name(event.key), event.type == pygame.KEYDOWN)
                continue   # F and K are keys of the piano now
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_f, pygame.K_k, pygame.K_F11, pygame.K_TAB):
                if event.key in (pygame.K_f, pygame.K_F11):
                    pygame.display.toggle_fullscreen()
                else:
                    shown = not shown
                roll, view, side = layout()
            if event.type == pygame.VIDEORESIZE:
                roll, view, side = layout()
            if side.width:
                panel.handle(event, side.topleft)
        now = time.monotonic()
        roll.instruments = {player: sound.GM[engine.sound[player]['instrument']] for player in PLAYERS}
        roll.readout = panel.mind.readout()
        roll.chart = (panel.mind.state or {}).get('chart')
        roll.offline = None if panel.mind.knobs else 'Connecting to the model…'
        roll.hint = typing.says() if typed else None if shown else 'K: show the controls'
        roll.draw(view, [tuple(n) for n in heard.visible(now, view.get_height())], now, now - started)
        if side.width:
            panel.draw(screen.subsurface(side))
        pygame.display.flip()
        clock.tick(FPS)


def video(midi_path, audio, out):
    import imageio_ffmpeg
    pygame.init()
    notes = notes_from_midi(midi_path)
    length = max(n[1] for n in notes) + TAIL_S
    screen = pygame.Surface(SIZE)
    roll = Roll(SIZE, octaves(notes))
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE[0]}x{SIZE[1]}', '-r', str(VIDEO_FPS), '-i', '-']
    if audio:
        command += ['-i', audio, '-c:a', 'aac', '-b:a', '160k', '-shortest']
    command += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'slow', '-movflags', '+faststart',
                out]
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in range(int(length * VIDEO_FPS)):
        now = frame / VIDEO_FPS
        roll.draw(screen, notes, now, now)
        ffmpeg.stdin.write(pygame.image.tobytes(screen, 'RGB'))
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"Video saved as {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', nargs='?', choices=('live', 'video'), default='live')
    parser.add_argument('midi', nargs='?', help='video: the saved duo (generated/<time>.mid)')
    parser.add_argument('--audio', help='video: its rendering, to put under the picture')
    parser.add_argument('-o', '--out', default='duo.mp4')
    parser.add_argument('--port', default='Midi Through', help='live: the MIDI port to listen to')
    parser.add_argument('--server', default=os.environ.get('AMT_SERVER', 'localhost:8765'),
                        help='live: the AMT server whose knobs to turn')
    args = parser.parse_args()
    if args.mode == 'video':
        video(args.midi, args.audio, args.out)
    else:
        live(args.port, args.server)
