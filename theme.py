"""
The look of the duo's window, shared by the piano roll (pianoroll.py) and its controls (knobs.py): colours and type.
"""
import pygame

BACKGROUND = ((10, 12, 19), (19, 23, 36))   # the piano roll's, top and bottom
SURFACE = (16, 19, 29)      # the panel
RAISED = (26, 30, 44)       # cards, steppers, the tabs' well
HOVER = (36, 41, 60)
SELECTED = (48, 54, 78)
LINE = (38, 43, 61)
TEXT = (232, 235, 243)
MUTED = (146, 152, 172)
FAINT = (86, 92, 114)
YOU = (255, 176, 84)        # amber
AI = (94, 214, 230)         # teal: its solo
CHORDS = (167, 139, 250)    # violet: its chords
BASS = (110, 205, 140)      # green: its bass

FONTS = ('notosans', 'dejavusans', 'freesans')   # the first one installed; pygame's own if none
_fonts, _texts = {}, {}


def font(size, bold=False):
    if (size, bold) not in _fonts:
        path = next((p for p in (pygame.font.match_font(name, bold=bold) for name in FONTS) if p), None)
        _fonts[(size, bold)] = pygame.font.Font(path, size)
    return _fonts[(size, bold)]


def render(string, size, colour, bold=False):
    """`string` drawn once and kept: the window redraws the same words sixty times a second."""
    key = (string, size, colour, bold)
    if key not in _texts:
        if len(_texts) > 3000:
            _texts.clear()
        _texts[key] = font(size, bold).render(string, True, colour)
    return _texts[key]


def text(surface, string, size, colour, pos, bold=False, anchor='topleft'):
    """Draws `string` with its `anchor` (a pygame.Rect attribute) at `pos`; returns where it went."""
    image = render(string, size, colour, bold)
    rect = image.get_rect(**{anchor: pos})
    surface.blit(image, rect)
    return rect


def width(string, size, bold=False):
    return font(size, bold).size(string)[0]


def fit(string, size, room, bold=False):
    """`string`, cut short with an ellipsis if it is wider than `room`."""
    if width(string, size, bold) <= room:
        return string
    while len(string) > 1 and width(string + '…', size, bold) > room:
        string = string[:-1]
    return string.rstrip() + '…'


def wrap(string, size, room, bold=False):
    """The lines of `string` within `room` pixels."""
    lines, line = [], ''
    for word in string.split():
        if line and width(f"{line} {word}", size, bold) > room:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def capital(string):
    return string[:1].upper() + string[1:]
