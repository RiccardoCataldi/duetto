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

        for name, (low, high, default) in {'min_step': (5, 60, 20), 'space': (10, 150, 50), 'left_step': (10, 200, 40),
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
        self.assertEqual((self.mind.knobs['min_step'].value, self.mind.knobs['space'].value), (8, 15))
        self.assertEqual(self.macro('density').shown(), 'dense')
        self.assertEqual(self.mind.knobs['left_step'].value, 15)
        self.assertEqual({k.name for k in self.sent}, {'min_step', 'space', 'left_step'})

    def test_a_knob_is_dimmed_while_what_it_shapes_is_not_happening(self):
        self.mind.state = {'mode': 'accompanying', 'key': 'C major', 'chord': 'G7', 'bpm': 92,
                           'active': ['out_of_key', 'space', 'apart']}
        self.assertTrue(self.mind.active(self.macro('role')))
        self.assertFalse(self.mind.active(self.macro('dialogue')))
        self.assertTrue(self.mind.active(self.mind.knobs['space']))
        self.assertFalse(self.mind.active(self.mind.knobs['echo']))
        self.assertEqual(self.mind.status(), 'C major · G7 · 92 bpm · accompanying you')

    def test_what_you_turn_is_kept_for_next_time(self):
        self.macro('role').set(0.0)
        again = knobs.Mind(None, self.path)
        self.assertEqual((again.saved['knobs']['apart'], again.saved['knobs']['lead']), (4.0, 0.0))
        self.assertEqual(next(m for m in again.macros if m.name == 'role').value, 0.0)

    def test_a_knob_you_never_turned_starts_where_its_control_stands(self):
        self.macro('role').set(1.0)                 # leads
        again = knobs.Mind(None, self.path)
        again.sock = DummySocket()
        again.saved['knobs'] = {'space': 30}        # as if only space had ever been turned
        served = {name: knobs.Knob(name, default, 0, 150, default, '', '', False, again.send)
                  for name, default in (('space', 50.0), ('apart', 2.0), ('lead', 3.0))}
        again._restore(served)
        self.assertEqual({name: knob.value for name, knob in served.items()}, {'space': 30, 'apart': 0.0, 'lead': 6.0})
        self.assertEqual(again.saved['knobs'], {'space': 30})   # it goes on following its control

    def test_a_preset_sets_its_controls_and_knobs_and_the_next_one_puts_back_what_it_leaves_out(self):
        self.mind.knobs['space'].value = 90
        presets = {preset[0]: preset for preset in self.mind.presets()}
        self.mind.play(presets['call & response'])
        self.assertEqual((self.macro('dialogue').value, self.mind.knobs['space'].value), (1.0, 150))
        self.assertTrue(self.mind.playing(presets['call & response']))
        self.mind.play(presets['it solos'])
        self.assertAlmostEqual(self.mind.knobs['space'].value, knobs.along((150, 50, 15), 0.8))   # density's, again
        self.assertEqual(self.macro('dialogue').value, 0.3)
        self.assertFalse(self.mind.playing(presets['call & response']))
        self.mind.play(presets['duet'])
        self.assertEqual(self.mind.knobs['space'].value, 50)
        self.assertTrue(all(m.value == 0.5 for m in self.mind.macros))

    def test_a_choice_shows_its_name(self):
        scale = knobs.Knob('scale', 3, 0, 3, 0, 'choice:yours/major/natural_minor/harmonic_minor', '', True, print)
        self.assertEqual(scale.shown(), 'harmonic minor')
        self.assertEqual(scale.step(), 1)
        bpm = knobs.Knob('bpm', 0, 0, 240, 0, 'bpm', '', True, print)
        self.assertEqual(bpm.shown(), 'yours')
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

    def test_what_the_model_hears_and_does_reads_as_cells(self):
        mind = knobs.Mind(None, os.path.join(self.tmp.name, 'other.json'))
        self.assertIsNone(mind.readout())
        mind.state = {'mode': 'alone', 'key': 'D minor', 'chord': None, 'bpm': None, 'active': []}
        self.assertEqual(mind.readout(), [('Key', 'D minor'), ('Chord', '—'), ('Tempo', '—'),
                                          ('AI', 'playing on its own')])

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
