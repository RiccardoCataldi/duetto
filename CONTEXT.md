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
What the model plays over a Tune: **Comps**, **Solos** or **Duet** (comps while you play, solos when you rest).
_Avoid_: role (that is a Character control), preset

**Style**:
How any Tune is played: **Swing** (swung eighths, a walking bass), **Bossa** (even eighths, a bossa's bass),
**Even 8ths** (even eighths, a walking bass) or **Ballad** (a bass in two, long chords, and a solo planned phrase by
phrase: sung legato, laid back, its last note held, silence after it, nothing faster than triplets). Yours to choose,
as in iReal Pro: a Tune never says its own.
_Avoid_: genre, feel, groove (that is a Character control)

**Voicing**:
The notes it comps a chord with: rootless, of four notes, in the inversion nearest the last one played; its
**shell** (third and seventh) while you play.
_Avoid_: chord shape, inversion (a Voicing has two)

**Way**:
How it plays the changes for two bars when it quotes no Lick: a bebop line, a bebop scale, an arpeggio up to the
ninth, the chord's extensions, its scale, a pentatonic, the blues, the altered scale over a dominant that resolves, or
Coltrane's cell (1 2 3 5). Over a Ballad: no pentatonic, no blues.
_Avoid_: mode, device, pattern

**Guide tone**:
The third or the seventh of a chord: where its solo lands on the beat the chord begins.

**Theme**:
The melody of a Tune, written in its chart bar by bar (optional): what its solo paraphrases, phrase by phrase, as
often as the `theme` Knob says.
_Avoid_: head, melody (in the UI), lead sheet

**Own phrase**:
Over a Ballad, a phrase of its solo that is no Lick and no Theme: its rhythm borrowed from a soloist's, its notes its
own, on their way to its Arrival.
_Avoid_: line (that is its solo's notes in a row, whatever plays them), plan

**Arrival**:
The note an Own phrase ends on, chosen before it begins: a colour of the chord there (third, seventh, a tension).
_Avoid_: target (that is a chord tone on the beat), goal

**Voice**:
One of the three things the model plays, each on its own MIDI channel and instrument: its **solo** (its line, or a
Lick it quotes), its **chords** (what it comps, under you or under its own solo) and its **bass**.
_Avoid_: hand (that is how the model's rules see them), track, part (that is what it plays over a Tune)

**Band**:
The instruments of the three Voices: a preset (piano trio, vibes, two pianos) or one chosen for each. It changes the
sound only, never the notes.
_Avoid_: ensemble, mode

**Player**:
The top of the panel, always in sight: Play/Stop, where the Tune is, a light on each beat, the tempo and the
metronome.
_Avoid_: transport bar

### Shaping it

**Character**:
The eleven musical controls (density, breath, harmony, role, dialogue, groove, risk, focus, memory, key follow, vocabulary),
each a musical intent.
_Avoid_: macro (in the UI), musical controls

**Knob**:
One parameter of the model's rules, with a range and a default. Character controls turn Knobs.
_Avoid_: setting, param

### What it plays

**Lick**:
A phrase of one to four bars, from where its player began it to where they ended it, filed under the chords it was
played over, free of any key: a great soloist's (from the Weimar Jazz Database) or **yours**, from what you played
over a Tune.
_Avoid_: quote (that is the act of playing a Lick)

**Upbeat**:
The notes of a Lick its soloist played before its first bar line, in the last two beats of the bar before.
_Avoid_: pickup (that is what the duo adds between two Licks), anacrusis

**Pickup**:
The one to three notes that lead from one Lick into the next when there is room between them: a walk up or down the
scale, a chromatic approach off the beat, or an enclosure of the next Lick's first note.
_Avoid_: bridge (that is a Tune's B section), interpolation

**Variation**:
A Lick played again to develop it, right after it was quoted or after you played it, whatever the chords: as it was
(a sequence), a beat later, its first half only, or upside down.
_Avoid_: mutation, transformation

**Arc**:
How its solo builds over a chorus (three of a blues): from eighths, short phrases and the low of its register up to
its height three quarters through, and down again.
_Avoid_: curve, intensity

**Recall**:
How many of the Licks it quotes are yours, from what you played over the Tune, rather than the great soloists'.
_Avoid_: memory (as a Knob name)

**Self window**:
How much of its own recent playing stays in the model's context.
_Avoid_: memory
