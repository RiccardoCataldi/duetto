"""
Serve the duo over TCP so the model can run on another machine (e.g. one with a GPU).

    python -m music_transformer.server            # listens on 127.0.0.1:8765
    ssh -N -L 8765:localhost:8765 gpu             # on the laptop, then AMT_SERVER=localhost:8765

One line per message, times in seconds since the connection opened:
    client -> server   "hello"                           the duo begins (any first line does; without it, at your
                                                         first key)
    client -> server   "on <onset> <duration> <pitch> <velocity>"   you pressed a key (the duration is a guess)
    client -> server   "off <onset> <pitch> <duration>"  the key you pressed at <onset> stopped sounding (pedal and all)
    server -> client   "note <onset> <duration> <pitch> <hand>"   the model's next note, of its left or right hand;
                                                      how hard is left to the client
    server -> client   "cut <onset>"                     forget the model's notes from <onset> on: they are resampled
    server -> client   "tempo <bpm> <metronome> <bar>"   the tempo it keeps to (0: yours), a bar's first beat at <bar>
                                                         and a beat every 60 / bpm from it, and whether to click
                                                         them (1); at once and whenever turned
One duo at a time; a second connection waits for the first to close.

A connection that starts with "knobs" turns the duo's knobs (duo.KNOBS) instead, while a duo plays or before:
    server -> client   "knob <name> <value> <lowest> <highest> <default> <unit> <modes> <what it does>"   one per knob
                                                         (<modes>: free, tune or free/tune, where it acts), then "ready"
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
    """The duo with the Anticipatory Music Transformer (duo.py)."""
    from music_transformer import duo
    from music_transformer.amt import AMTGenerator
    from music_transformer.memory import midi_files
    generator = AMTGenerator(model_name=model)
    hands = {} if left is None else {'left': left}
    corpus = midi_files(corpus) if corpus and os.path.isdir(corpus) else []
    print(f"AMT model loaded on {generator.device}, {len(corpus)} MIDI files in its memory")

    def session(**knobs):
        return duo.DuoSession(generator.model, generator.device, corpus=corpus, **hands, **knobs)

    return SimpleNamespace(engine=duo, session=session)


def make_handler(player):
    """`player`: its `engine` module (KNOBS, MODES, acting, knob, LOOKAHEAD, MARGIN) and `session(started, **knobs)`."""
    engine = player.engine
    KNOBS, LOOKAHEAD, knob = engine.KNOBS, engine.LOOKAHEAD, engine.knob
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
                self.wfile.write(f"knob {name} {knobs[name]} {low} {high} {KNOBS[name][0]} {unit or '-'} "
                                 f"{'/'.join(engine.MODES[name])} {what}\n".encode())
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
                            if name in ('bpm', 'metronome', 'chart', 'playing'):
                                inbox.put(('tempo',))
                        print(f"duo: {name} = {knobs[name]}")
            except (ConnectionResetError, ValueError) as e:
                print(f"duo: bad knob ({e})")

        def report(self):
            while True:
                try:
                    # with no duo yet, the knobs as last turned say its Mode
                    state = playing[0][0].state() if playing else {
                        **engine.acting(knobs), 'doing': 'listening', 'key': None, 'chord': None, 'bpm': None}
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
                knobs['playing'] = 0   # a tune waits for Play, whatever the duo before left
                session = player.session(started=opened, **knobs)
                heard, played, shown = 0, 0, None
                sampling, answers, since = [], [], None   # seconds to sample each note; from your note to its next
                playing.append((session, inbox))
                inbox.put(('tempo',))
                try:
                    while True:
                        # your keys come first; otherwise sample on, or wait while far enough ahead
                        # it counts in ticks of 10 ms
                        if session.base is None:
                            session.begin()   # over a tune it may start by itself
                        wait = 0.2 if session.base is None else max(session.ahead() - LOOKAHEAD, 0) / 100
                        try:
                            note = inbox.get(timeout=wait) if wait != 0 else inbox.get_nowait()
                        except queue.Empty:
                            note = False
                        if note is None:
                            break
                        if note and note[0] == 'tempo':
                            # it clicks while a tune plays, or, free, once the duo has begun
                            clicks = session.metronome and (session.playing if session.chart else session.base is not None)
                            self.wfile.write(f"tempo {session.pace} {int(clicks)} {session.began:.3f}\n".encode())
                            if session.chart and not session.playing:   # stopped: what it has not sounded goes
                                self.wfile.write(f"cut {time.monotonic() - opened:.3f}\n".encode())
                        elif note and note[0] == 'off':
                            session.release(*note[1:])
                        elif note:
                            heard += 1
                            if heard == 1:
                                inbox.put(('tempo',))   # free, its click starts with your first note
                            since = since or time.perf_counter()
                            cut = session.hear(*note[1:])
                            key = session.listener.said_key()
                            if key != shown and key is not None:
                                shown = key
                                sure = session.listener.belief[key]
                                print(f"duo: {note[1]:.1f}s you are in {session.listener.name(key)} ({sure:.0%})")
                            if cut is not None:
                                self.wfile.write(f"cut {cut:.3f}\n".encode())
                        elif session.base is not None and session.ahead() < LOOKAHEAD:
                            begun = time.perf_counter()
                            sampled = session.play()
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
    parser.add_argument('--model', default=None, help='Hugging Face name or local path of the AMT model')
    parser.add_argument('--left', type=int, default=None,
                        help='the General MIDI program its left hand is in the context (default: duo.LEFT, a bass)')
    parser.add_argument('--corpus', default=None, help='a folder of MIDI files it remembers: your saved duos')
    args = parser.parse_args()
    player = amt(args.model, args.left, args.corpus)
    print(f"listening on {args.host}:{args.port}")
    Server((args.host, args.port), make_handler(player)).serve_forever()
