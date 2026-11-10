"""
Serve the duo over TCP so the model can run on another machine (e.g. a Jetson).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    python -m music_transformer.server --engine aria --checkpoint aria-medium-gen.safetensors   # Aria, on CUDA
    ssh -N -L 8765:localhost:8765 jetson          # on the laptop, then AMT_SERVER=localhost:8765

One line per message, times in seconds since the connection opened:
    client -> server   "on <onset> <duration> <pitch> <velocity>"   you pressed a key (the duration is a guess)
    client -> server   "off <onset> <pitch> <duration>"  the key you pressed at <onset> stopped sounding (pedal and all)
    server -> client   "note <onset> <duration> <pitch> <hand> [<velocity>]"   the model's next note, of its left or
                                                      right hand; Aria says how hard, AMT leaves it to the client
    server -> client   "cut <onset>"                     forget the model's notes from <onset> on: they are resampled
                                                         (or the strategy knob changed: it starts again its new way)
    server -> client   "tempo <bpm> <metronome>"         the tempo it keeps to (0: yours), its beats on k * 60 / bpm,
                                                         and whether to click them (1); at once and whenever turned
One duo at a time; a second connection waits for the first to close.

A connection that starts with "knobs" turns the duo's knobs (the engine's KNOBS) instead, while a duo plays or before:
    server -> client   "engine <amt or aria>", then
                       "knob <name> <value> <lowest> <highest> <default> <unit> <what it does>"   one per knob, then "ready"
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

from types import SimpleNamespace

STATE_S = 0.25    # how often a knobs connection hears what the duo is doing


def ms(seconds):
    """'40 ms (95% within 60 ms)': the median and the 95th percentile of `seconds`."""
    if not seconds:
        return '-'
    ordered = sorted(seconds)
    worst = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return f"{statistics.median(ordered) * 1000:.0f} ms (95% within {worst * 1000:.0f} ms)"


def amt(model=None, left=None, corpus=None):
    """The duo with the Anticipatory Music Transformer (duo.py), or one of the jam_bot's ways (jam.py)."""
    from music_transformer import duo
    from music_transformer.amt import AMTGenerator
    from music_transformer.jam import JamSession
    from music_transformer.memory import midi_files
    generator = AMTGenerator(model_name=model)
    hands = {} if left is None else {'left': left}
    corpus = midi_files(corpus) if corpus and os.path.isdir(corpus) else []
    print(f"AMT model loaded on {generator.device}, {len(corpus)} MIDI files in its memory")

    def session(**knobs):
        if knobs.get('strategy'):
            return JamSession(generator.model, generator.device, **knobs)
        return duo.DuoSession(generator.model, generator.device, corpus=corpus, **hands, **knobs)

    return SimpleNamespace(engine=duo, session=session)


def aria(checkpoint, genre='jazz'):
    """The duo with Aria (aria_duo.py)."""
    from music_transformer import aria_duo
    model = aria_duo.AriaModel(checkpoint)
    warm = aria_duo.AriaSession(model, genre=genre)   # its first note on the GPU takes ~0.3 s: not in your duo
    warm.hear(0.0, 0.3, 60)
    warm.play()
    print(f"Aria loaded on {model.device} from {checkpoint}")
    return SimpleNamespace(engine=aria_duo, session=lambda **kw: aria_duo.AriaSession(model, genre=genre, **kw))


def make_handler(player):
    """`player`: its `engine` module (KNOBS, ALWAYS, knob, LOOKAHEAD, MARGIN) and `session(started, **knobs)`."""
    engine = player.engine
    KNOBS, ALWAYS, LOOKAHEAD, knob = engine.KNOBS, engine.ALWAYS, engine.LOOKAHEAD, engine.knob
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
            self.wfile.write(f"engine {engine.NAME}\n".encode())
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
                session = player.session(started=opened, **knobs)
                heard, played, shown = 0, 0, None
                sampling, answers, since = [], [], None   # seconds to sample each note; from your note to its next
                playing.append((session, inbox))
                inbox.put(('tempo',))
                try:
                    while True:
                        # your keys come first; otherwise sample on, or wait while far enough ahead
                        # both engines count in ticks of 10 ms
                        wait = None if session.base is None else max(session.ahead() - LOOKAHEAD, 0) / 100
                        try:
                            note = inbox.get(timeout=wait) if wait != 0 else inbox.get_nowait()
                        except queue.Empty:
                            note = False
                        if note is None:
                            break
                        if note and note[0] == 'tempo':
                            self.wfile.write(f"tempo {session.bpm} {session.metronome}\n".encode())
                        elif note and note[0] == 'swap':   # the strategy knob: the duo's way, or one of the jam_bot's
                            playing.remove((session, inbox))
                            session = player.session(started=opened, **knobs)
                            playing.append((session, inbox))
                            self.wfile.write(f"cut {time.monotonic() - opened:.3f}\n".encode())
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
                                onset, duration, pitch, *how = sampled
                                self.wfile.write(f"note {onset:.3f} {duration:.3f} {pitch} {' '.join(map(str, how))}\n".encode())
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
                          f"after it (it sounds no sooner than {engine.MARGIN * 10} ms after)")

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
    parser.add_argument('--engine', choices=('amt', 'aria'), default='amt')
    parser.add_argument('--model', default=None, help='amt: Hugging Face name or local path of the AMT model')
    parser.add_argument('--left', type=int, default=None,
                        help='amt: the General MIDI program its left hand is in the context (default: duo.LEFT, a bass)')
    parser.add_argument('--corpus', default=None, help='amt: a folder of MIDI files it remembers: your saved duos')
    parser.add_argument('--checkpoint', help='aria: the model.safetensors of aria-medium-gen (or one fine-tuned)')
    parser.add_argument('--genre', default='jazz', help="aria: the genre it is told the music is ('' for none)")
    args = parser.parse_args()
    if args.engine == 'aria':
        if not args.checkpoint:
            parser.error('--engine aria needs --checkpoint')
        player = aria(args.checkpoint, args.genre or None)
    else:
        player = amt(args.model, args.left, args.corpus)
    print(f"listening on {args.host}:{args.port}")
    Server((args.host, args.port), make_handler(player)).serve_forever()
