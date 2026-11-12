# duetto

A second piano that plays with you in real time: the Anticipatory Music Transformer, steered by rules that make it
listen like a jazz partner, either freely or over a tune.

## Language

### Playing together

**Mode**:
How the duo plays: **Free** (it follows your key, chord and pulse) or **Tune** (it plays over a Tune's chords).
_Avoid_: engine, strategy, real book mode

**Tune**:
A chart in `charts/`: a title, key, tempo and its chords bar by bar, played round and round.
_Avoid_: chart (in the UI), song, real book, play along

**Transport**:
Whether a Tune is playing: Play starts its Count-in, Stop silences it. Never saved: the duo never starts by itself.
_Avoid_: running, started

**Count-in**:
The bar of clicks before a Tune's first bar.

**Part**:
What the model plays over a Tune: **Comps**, **Solos**, **Duet** (comps while you play, solos when you rest) or
**Trade fours**.
_Avoid_: role (that is a Character control), preset

### Shaping it

**Character**:
The nine musical controls (density, harmony, role, dialogue, groove, risk, focus, memory, key follow): one position
for both Modes, each meaning the same musical intent in either.
_Avoid_: macro (in the UI), musical controls

**Preset**:
A named position of the Character plus the Free Mode's texture (chord size, hands, octaves, timeout).
_Avoid_: playing style, way

**Knob**:
One parameter of the model's rules, with a range and a default; each declares the Modes it acts in. Character
controls turn Knobs.
_Avoid_: setting, param

### What it plays

**Lick**:
A phrase of one to four bars filed under the chords it was played over, free of any key: a great soloist's (from the
Weimar Jazz Database) or **yours**, from what you played over a Tune.
_Avoid_: quote (that is the act of playing a Lick)

**Recall**:
How closely it follows its memory of you: in Free Mode the Somax-style corpus of your playing, over a Tune your Licks.
_Avoid_: memory (as a Knob name)

**Self window**:
How much of its own recent playing stays in the model's context.
_Avoid_: memory
