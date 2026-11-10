"""The server with Aria's session and a model that knows nothing: needs torch and ariautils, not the model itself."""
import os
import socket
import sys
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from music_transformer import server
try:
    import torch
    from music_transformer import aria_duo
except ImportError:   # ariautils is not installed here
    aria_duo = None


@unittest.skipIf(aria_duo is None, "ariautils is not installed")
class ServerTest(unittest.TestCase):
    def test_a_duo_hears_your_note_and_says_its_tempo_and_its_notes_with_how_hard(self):
        model = lambda tokens, start: torch.zeros(aria_duo.TOKENIZER.vocab_size)
        player = SimpleNamespace(engine=aria_duo, session=lambda **knobs: aria_duo.AriaSession(model, **knobs))
        with server.Server(('127.0.0.1', 0), server.make_handler(player)) as duo:
            threading.Thread(target=duo.serve_forever, daemon=True).start()
            with socket.create_connection(duo.server_address, timeout=10) as sock:
                sock.sendall(b"on 0.000 0.300 60 80\n")
                lines = sock.makefile('r')
                self.assertEqual(lines.readline(), "tempo 0 0\n")
                kind, onset, duration, pitch, hand, velocity = lines.readline().split()
                self.assertEqual(kind, 'note')
                self.assertIn(hand, ('left', 'right'))
                self.assertIn(int(velocity), aria_duo.VELOCITIES)
            duo.shutdown()


if __name__ == '__main__':
    unittest.main()
