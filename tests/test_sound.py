import os
import tempfile
import unittest

import mido

import sound


def duo_file(path):
    mid = mido.MidiFile()
    for channel, pitch in ((0, 60), (1, 67)):
        mid.tracks.append(mido.MidiTrack([
            mido.Message('program_change', channel=channel, program=0),
            mido.Message('note_on', channel=channel, note=pitch, velocity=100, time=0),
            mido.Message('note_off', channel=channel, note=pitch, velocity=0, time=mid.ticks_per_beat)]))
    mid.save(path)


class SoundTest(unittest.TestCase):
    def test_each_setting_reaches_the_players_channel_and_its_echoes(self):
        self.assertEqual(sound.messages(sound.AI, 'instrument', 48), [[0xC1, 48], [0xC3, 48]])
        self.assertEqual(sound.messages(sound.YOU, 'reverb', 90), [[0xB0, 91, 90], [0xB2, 91, 90]])
        self.assertEqual(sound.messages(sound.YOU, 'delay', 300), [])

    def test_the_envelope_is_a_soundfont_nrpn_offset_in_fluidsynths_units(self):
        attack = sound.messages(sound.YOU, 'attack', 2400)[:4]   # two times longer, twice: 2400 timecents
        data = 8192 + 2400 // 2
        self.assertEqual(attack, [[0xB0, 99, 120], [0xB0, 98, 34], [0xB0, 6, data >> 7], [0xB0, 38, data & 127]])

    def test_a_note_echoes_quieter_each_time_and_lets_go_after_each_echo(self):
        values = sound.defaults()
        values[sound.YOU].update(delay=250, echo=0.5, feedback=0.5)
        echoes = sound.Echoes()
        on = echoes.plan([0x90, 60, 100], 10.0, values)
        self.assertEqual(on, [(10.25, [0x92, 60, 50]), (10.5, [0x92, 60, 25]), (10.75, [0x92, 60, 12]),
                              (11.0, [0x92, 60, 6])])
        off = echoes.plan([0x80, 60, 0], 10.4, values)
        self.assertEqual([when for when, _ in off], [10.65, 10.9, 11.15, 11.4])
        self.assertTrue(all(m == [0x82, 60, 0] for _, m in off))

    def test_without_echo_level_there_is_no_delay(self):
        self.assertEqual(sound.Echoes().plan([0x90, 60, 100], 0, sound.defaults()), [])

    def test_an_echo_is_let_go_only_when_no_other_echo_holds_its_key(self):
        gate = sound.Echoes()
        gate.gate([0x92, 60, 50])
        gate.gate([0x92, 60, 25])
        self.assertIsNone(gate.gate([0x82, 60, 0]))
        self.assertEqual(gate.gate([0x82, 60, 0]), [0x82, 60, 0])

    def test_a_saved_duo_is_baked_with_its_sound_and_echoes(self):
        values = sound.defaults()
        values[sound.AI].update(instrument=48, echo=0.5, feedback=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            duo_file(os.path.join(tmp, 'duo.mid'))
            sound.bake(os.path.join(tmp, 'duo.mid'), os.path.join(tmp, 'baked.mid'), values)
            baked = [m for m in mido.MidiFile(os.path.join(tmp, 'baked.mid')) if not m.is_meta]
        programs = {(m.channel, m.program) for m in baked if m.type == 'program_change'}
        self.assertEqual(programs, {(0, 0), (2, 0), (1, 48), (3, 48), (4, 0), (6, 0), (5, 32), (7, 32)})   # you; its
        # solo, its chords and its bass, each with its echoes' channel
        notes = [(m.channel, m.note, m.velocity) for m in baked if m.type == 'note_on' and m.velocity]
        self.assertEqual(sorted(notes), [(0, 60, 100), (1, 67, 100), (3, 67, 50)])

    def test_the_sound_is_kept_between_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'sound.json')
            values = sound.defaults()
            values[sound.YOU]['instrument'] = 19
            sound.save(values, path)
            self.assertEqual(sound.load(path)[sound.YOU]['instrument'], 19)
            self.assertEqual(sound.load(os.path.join(tmp, 'none.json')), sound.defaults())


if __name__ == '__main__':
    unittest.main()
