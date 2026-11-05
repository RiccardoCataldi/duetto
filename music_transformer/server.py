"""
Serve the duo over TCP so the model can run on another machine (e.g. a Jetson).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    ssh -N -L 8765:localhost:8765 jetson          # on the laptop, then AMT_SERVER=localhost:8765

One line per message, times in seconds since the connection opened:
    client -> server   "on <onset> <duration> <pitch> <velocity>"   you pressed a key (the duration is a guess)
    client -> server   "off <onset> <pitch> <duration>"  the key you pressed at <onset> stopped sounding (pedal and all)
    server -> client   "note <onset> <duration> <pitch> <hand>"   the model's next note, of its left or right hand
    server -> client   "cut <onset>"                     forget the model's notes from <onset> on: they are resampled
                                                         (or the strategy knob changed: it starts again its new way)
    server -> client   "tempo <bpm> <metronome>"         the tempo it keeps to (0: yours), its beats on k * 60 / bpm,
                                                         and whether to click them (1); at once and whenever turned
One duo at a time; a second connection waits for the first to close.

A connection that starts with "knobs" turns the duo's knobs (duo.KNOBS) instead, while a duo plays or before:
    server -> client   "knob <name> <value> <lowest> <highest> <default> <unit> <what it does>"   one per knob, then "ready"
    client -> server   "set <name> <value>"              from the model's next note on, and for the next duos
    server -> client   "state <json>"                    four times a second: DuoSession.state(), what it hears and does
"""

import argparse
import itertools
import json
import math
import os
import queue
import socketserver
import statistics
import threading
import time

from anticipation.config import TIME_RESOLUTION

from music_transformer.amt import AMTGenerator
from music_transformer.duo import ALWAYS, KNOBS, LOOKAHEAD, MARGIN, DuoSession, knob
from music_transformer.jam import STRATEGIES, JamSession
from music_transformer.memory import midi_files

STATE_S = 0.25    # how often a knobs connection hears what the duo is doing


def ms(seconds):
    """'40 ms (95% within 60 ms)': the median and the 95th percentile of `seconds`."""
    if not seconds:
        return '-'
    ordered = sorted(seconds)
    worst = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return f"{statistics.median(ordered) * 1000:.0f} ms (95% within {worst * 1000:.0f} ms)"


def make_handler(generator, left=None, corpus=()):
    lock = threading.Lock()
    knobs = {name: default for name, (default, *_) in KNOBS.items()}   # as last turned, for this duo and the next
    playing = []                                                         # (session, inbox) of the duo now playing

    class Handler(socketserver.StreamRequestHandler):
        disable_nagle_algorithm = True  # every note goes out at once

        def handle(self):
            opened = time.monotonic()   # times on the wire count from here
            first = self.rfile.readline()
            if first.split()[:1] == [b'knobs']:
                self.turn()
            else:
                self.duo(first, opened)

        def turn(self):
            for name, (_, low, high, unit, what) in KNOBS.items():
                self.wfile.write(f"knob {name} {knobs[name]} {low} {high} {KNOBS[name][0]} {unit or '-'} {what}\n".encode())
            self.wfile.write(b"ready\n")
            threading.Thread(target=self.report, daemon=True).start()
            try:
                for line in self.rfile:
                    kind, *values = line.decode().split()
                    if kind == 'set' and len(values) == 2 and values[0] in KNOBS:
                        name = values[0]
                        knobs[name] = knob(name, float(values[1]))
                        for session, inbox in playing:
                            session.turn(**{name: knobs[name]})
                            if name in ('bpm', 'metronome'):
                                inbox.put(('tempo',))
                            if name == 'strategy':
                                inbox.put(('swap',))
                        print(f"duo: {name} = {knobs[name]}")
            except (ConnectionResetError, ValueError) as e:
                print(f"duo: bad knob ({e})")

        def report(self):
            idle = {'mode': 'listening', 'key': None, 'chord': None, 'bpm': None, 'active': sorted(ALWAYS)}
            while True:
                try:
                    state = playing[0][0].state() if playing else idle
                except Exception:          # the duo moved on while we read it; the next report will do
                    state = None
                try:
                    if state:
                        self.wfile.write(f"state {json.dumps(state)}\n".encode())
                except (OSError, ValueError):   # the knobs connection closed
                    return
                time.sleep(STATE_S)

        def duo(self, first, opened):
            with lock:
                print("duo: started")
                inbox = queue.Queue()
                threading.Thread(target=self.read, args=(inbox, first), daemon=True).start()
                hands = {} if left is None else {'left': left}

                def begin():   # the duo's way, or one of the jam_bot's
                    if knobs['strategy']:
                        return JamSession(generator.model, generator.device, started=opened, **knobs)
                    return DuoSession(generator.model, generator.device, started=opened, corpus=corpus, **hands, **knobs)

                session = begin()
                heard, played, shown = 0, 0, None
                sampling, answers, since = [], [], None   # seconds to sample each note; from your note to its next
                playing.append((session, inbox))
                inbox.put(('tempo',))
                try:
                    while True:
                        # your keys come first; otherwise sample on, or wait while far enough ahead
                        wait = None if session.base is None else max(session.ahead() - LOOKAHEAD, 0) / TIME_RESOLUTION
                        try:
                            note = inbox.get(timeout=wait) if wait != 0 else inbox.get_nowait()
                        except queue.Empty:
                            note = False
                        if note is None:
                            break
                        if note and note[0] == 'tempo':
                            self.wfile.write(f"tempo {session.bpm} {session.metronome}\n".encode())
                        elif note and note[0] == 'swap':
                            playing.remove((session, inbox))
                            session = begin()
                            playing.append((session, inbox))
                            self.wfile.write(f"cut {time.monotonic() - opened:.3f}\n".encode())
                            print(f"duo: its way is now {STRATEGIES[knobs['strategy']]}")
                        elif note and note[0] == 'off':
                            session.release(*note[1:])
                        elif note:
                            heard += 1
                            since = since or time.perf_counter()
                            cut = session.hear(*note[1:])
                            key = session.listener.said_key()
                            if key != shown:
                                shown = key
                                sure = session.listener.belief[key]
                                print(f"duo: {note[1]:.1f}s you are in {session.listener.name(key)} ({sure:.0%})")
                            if cut is not None:
                                self.wfile.write(f"cut {cut:.3f}\n".encode())
                        elif session.ahead() < LOOKAHEAD:
                            begun = time.perf_counter()
                            sampled = session.play()
                            cut = session.recut()
                            if cut is not None:
                                self.wfile.write(f"cut {cut:.3f}\n".encode())
                            if sampled:
                                onset, duration, pitch, hand = sampled
                                self.wfile.write(f"note {onset:.3f} {duration:.3f} {pitch} {hand}\n".encode())
                                sampling.append(time.perf_counter() - begun)
                                if since:
                                    answers.append(time.perf_counter() - since)
                                    since = None
                                played += 1
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    playing.remove((session, inbox))
                print(f"duo: ended, heard {heard} notes, played {played}")
                if sampling:
                    print(f"duo: a note took {ms(sampling)} to sample; its first after yours went out {ms(answers)} "
                          f"after it (it sounds no sooner than {MARGIN * 10} ms after)")

        def read(self, inbox, first):
            try:
                for line in itertools.chain([first], self.rfile):
                    kind, *values = line.split()
                    if kind == b'on':
                        onset, duration, pitch, *velocity = values
                        inbox.put(('on', float(onset), float(duration), int(pitch), int(velocity[0]) if velocity else 80))
                    elif kind == b'off':
                        onset, pitch, duration = values
                        inbox.put(('off', float(onset), int(pitch), float(duration)))
            except (ConnectionResetError, ValueError) as e:
                print(f"duo: bad input ({e})")
            finally:
                inbox.put(None)

    return Handler


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', default=None, help='Hugging Face name or local path of the AMT model')
    parser.add_argument('--left', type=int, default=None,
                        help='the General MIDI program its left hand is in the context (default: duo.LEFT, a bass)')
    parser.add_argument('--corpus', default=None, help='a folder of MIDI files it remembers: your saved duos')
    args = parser.parse_args()
    generator = AMTGenerator(model_name=args.model)
    print(f"AMT model loaded on {generator.device}, listening on {args.host}:{args.port}")
    corpus = midi_files(args.corpus) if args.corpus and os.path.isdir(args.corpus) else []
    print(f"{len(corpus)} MIDI files in its memory")
    Server((args.host, args.port), make_handler(generator, args.left, corpus)).serve_forever()
