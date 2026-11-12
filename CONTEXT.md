# duetto

A second piano that plays with you in real time: the Anticipatory Music Transformer, steered by rules that make it
listen like a jazz partner over a tune.

## Language

### Playing together

**Tune**:
A chart in `charts/`: a title, key, tempo and its chords bar by bar, played round and round. The duo always plays
over one: the first, until you choose another.
_Avoid_: chart (in the UI), song, real book, play along, mode (there was a Free Mode with no Tune: it is gone)

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
The nine musical controls (density, harmony, role, dialogue, groove, risk, focus, memory, key follow), each a
musical intent.
_Avoid_: macro (in the UI), musical controls

**Knob**:
One parameter of the model's rules, with a range and a default. Character controls turn Knobs.
_Avoid_: setting, param

### What it plays

**Lick**:
A phrase of one to four bars filed under the chords it was played over, free of any key: a great soloist's (from the
Weimar Jazz Database) or **yours**, from what you played over a Tune.
_Avoid_: quote (that is the act of playing a Lick)

**Pickup**:
The one to three notes that lead from one Lick into the next when there is room between them: a walk up or down the
scale, a chromatic approach off the beat, or an enclosure of the next Lick's first note.
_Avoid_: bridge (that is a Tune's B section), interpolation

**Recall**:
How many of the Licks it quotes are yours, from what you played over the Tune, rather than the great soloists'.
_Avoid_: memory (as a Knob name)

**Self window**:
How much of its own recent playing stays in the model's context.
_Avoid_: memory
