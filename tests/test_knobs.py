import os
import tempfile
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import knobs


class MindTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'mind.json')
        self.mind = knobs.Mind(None, self.path)   # no server: we give it the knobs ourselves
        self.mind.sock = DummySocket()
        self.sent = []

        def send(knob):
            self.sent.append(knob)
            self.mind.send(knob)

        for name, (low, high, default) in {'min_step': (5, 60, 20), 'max_rest': (50, 800, 200),
                                           'apart': (0.0, 6.0, 2.0), 'lead': (0.0, 6.0, 3.0),
                                           'echo': (0.0, 6.0, 3.0)}.items():
            self.mind.knobs[name] = knobs.Knob(name, default, low, high, default, '', '', isinstance(low, int),
                                               send)

    def tearDown(self):
        self.tmp.cleanup()

    def macro(self, label):
        return next(m for m in self.mind.macros if m.name == label)

    def test_a_musical_control_turns_its_knobs_along_their_line_and_its_middle_is_the_default(self):
        self.assertEqual(knobs.along((60, 20, 8), 0.5), 20)
        self.assertEqual(knobs.along((60, 20, 8), 0.0), 60)
        self.assertEqual(knobs.along((60, 20, 8), 0.75), 14)
        self.macro('density').set(1.0)
        self.assertEqual((self.mind.knobs['min_step'].value, self.mind.knobs['max_rest'].value), (8, 100))
        self.assertEqual(self.macro('density').shown(), 'dense')
        self.assertEqual({k.name for k in self.sent}, {'min_step', 'max_rest'})

    def test_what_you_turn_is_kept_for_next_time(self):
        self.macro('role').set(0.0)
        again = knobs.Mind(None, self.path)
        self.assertEqual((again.saved['knobs']['apart'], again.saved['knobs']['lead']), (4.0, 0.0))
        self.assertEqual(next(m for m in again.macros if m.name == 'role').value, 0.0)

    def test_a_knob_you_never_turned_starts_where_its_control_stands(self):
        self.macro('role').set(1.0)                 # leads
        again = knobs.Mind(None, self.path)
        again.sock = DummySocket()
        again.saved['knobs'] = {'min_step': 30}     # as if only min_step had ever been turned
        served = {name: knobs.Knob(name, default, 0, 150, default, '', '', False, again.send)
                  for name, default in (('min_step', 20.0), ('apart', 2.0), ('lead', 3.0))}
        again._restore(served)
        self.assertEqual({name: knob.value for name, knob in served.items()}, {'min_step': 30, 'apart': 0.0, 'lead': 6.0})
        self.assertEqual(again.saved['knobs'], {'min_step': 30})   # it goes on following its control

    def test_a_choice_shows_its_name(self):
        part = knobs.Knob('part', 3, 0, 3, 1, 'choice:comps/duet/solos/trades', '', True, print)
        self.assertEqual(part.shown(), 'trades')
        self.assertEqual(part.step(), 1)
        tune = knobs.Knob('chart', 2, 1, 3, 1, 'choice:autumn_leaves/blues_in_f/misty', '', True, print)
        self.assertEqual(tune.shown(), 'blues in f')   # the tunes count from 1: there is always one
        bpm = knobs.Knob('bpm', 0, 0, 240, 0, 'bpm', '', True, print)
        self.assertEqual(bpm.shown(), "the tune's")
        bpm.value = 96
        self.assertEqual(bpm.shown(), '96 bpm')


class PanelTest(unittest.TestCase):
    def setUp(self):
        import pygame
        import sound
        self.pygame = pygame
        pygame.init()
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = DummyEngine(sound.defaults())
        self.panel = knobs.Panel(None, self.engine, os.path.join(self.tmp.name, 'mind.json'))
        self.panel.tab = 1                           # your sound: it needs no server
        self.surface = pygame.Surface((knobs.PANEL_W, 720))
        self.panel.draw(self.surface)

    def tearDown(self):
        self.tmp.cleanup()

    def target(self, kind, name):
        return next(rect for k, what, rect in self.panel.targets if k == kind and what.name == name)

    def click(self, pos):
        for kind in (self.pygame.MOUSEBUTTONDOWN, self.pygame.MOUSEBUTTONUP):
            self.panel.handle(self.pygame.event.Event(kind, pos=pos, button=1), (0, 0))

    def test_where_the_tune_is_reads_as_cells(self):
        mind = knobs.Mind(None, os.path.join(self.tmp.name, 'other.json'))
        self.assertIsNone(mind.readout())
        mind.state = {'doing': 'alone', 'key': 'Misty', 'chord': None, 'bpm': None}
        self.assertEqual(mind.readout(), [('Tune', 'Misty'), ('Chord', '—'), ('Tempo', '—')])

    def test_the_metronome_is_beside_play_and_a_click_turns_it_on_and_off(self):
        mind = self.panel.mind
        for name, low, high, unit in (('chart', 1, 3, 'choice:autumn_leaves/blues_in_f/misty'),
                                      ('playing', 0, 1, 'choice:stopped/playing'), ('metronome', 0, 1, 'choice:off/on'),
                                      ('part', 0, 3, 'choice:comps/duet/solos/trades'),
                                      ('lowest', 0, 8, 'octave_from'), ('highest', 0, 8, 'octave_to')):
            mind.knobs[name] = knobs.Knob(name, low, low, high, low, unit, '', True, mind.send)
        self.panel.tab = 0                           # Play
        card = lambda name: next(rect for kind, what, rect in self.panel.targets if kind == 'card' and what[0] == name)
        self.panel.draw(self.surface)
        play, click = card('Play'), card('Metronome')
        self.assertEqual(click.y, play.y)            # on Play's row, to its right
        self.assertGreater(click.x, play.right)
        self.click(click.center)
        self.assertEqual(mind.knobs['metronome'].value, 1)
        self.panel.draw(self.surface)
        self.click(card('Metronome').center)
        self.assertEqual(mind.knobs['metronome'].value, 0)
        self.assertEqual(mind.knobs['playing'].value, 0)   # and the tune still waits for Play

    def test_a_steppers_arrows_step_it(self):
        box = self.target('stepper', 'instrument')
        self.click((box.right - 5, box.centery))
        self.click((box.right - 5, box.centery))
        self.assertEqual(self.engine.calls[-1], (0, 'instrument', 2))
        self.click((box.x + 5, box.centery))
        self.assertEqual(self.engine.calls[-1], (0, 'instrument', 1))

    def test_the_wheel_scrolls_over_a_control_and_ctrl_wheel_turns_it(self):
        from unittest import mock
        track = self.target('slider', 'volume')
        wheel = lambda y: self.panel.handle(self.pygame.event.Event(self.pygame.MOUSEWHEEL, x=0, y=y), (0, 0))
        calls = len(self.engine.calls)
        with mock.patch('pygame.mouse.get_pos', return_value=track.center):
            wheel(-1)
            self.assertEqual(len(self.engine.calls), calls)   # scrolling down over it leaves it be
            self.assertGreater(self.panel.scroll[1], 0)
            scrolled = self.panel.scroll[1]
            with mock.patch('pygame.key.get_mods', return_value=self.pygame.KMOD_LCTRL):
                wheel(1)
        self.assertEqual(self.engine.calls[-1][1], 'volume')
        self.assertEqual(self.panel.scroll[1], scrolled)


class DummyEngine:
    def __init__(self, values):
        self.sound, self.calls = values, []

    def set(self, player, name, value):
        self.calls.append((player, name, round(value)))


class DummySocket:
    def sendall(self, data):
        pass


if __name__ == '__main__':
    unittest.main()
