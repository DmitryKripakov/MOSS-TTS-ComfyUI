"""Director prompts and the timed-script builder for the voice-acting fine-tunes.

Two layers, as LAION runs them (model card of
laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3, CC-BY-4.0, and
docs/DIRECTOR.md of LAION-AI/Humaneness-Voice-Demo-Server):

  1. A language model (the "director") turns plain text into a performance:
     a delivery cue before every sentence, vocal bursts between sentences,
     beats of silence — and NEVER a number. `system_prompt()` is what it is told.
  2. `build()` does the arithmetic the sft3 format needs: `[N.N seconds
     duration]` before every speech segment, `[N.N seconds pause]` for gaps,
     `(label, N.N seconds)` for bursts, and Tokens = the sum at 12.5 frames/s.
     Leaving these numbers to the language model is a bad bet — the format only
     works when they add up, and length control is what sft3 honours best.

The sft3 bracket rule: square bracket = seconds; round bracket WITH a number = a
burst; round bracket WITHOUT one = a burst if it names a known sound, otherwise a
delivery direction. A direction that picks up a number turns into a sound.

The LAION 8B voice-acting model (delay) takes a plain `instruction` and `text`
instead; its director output is two fields and needs no arithmetic.

Pure Python — no torch, no model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

FRAME_RATE = 12.5               # MOSS audio frames per second
SECONDS_PER_WORD = 4.0 / FRAME_RATE   # LAION's TIMED_FRAMES_PER_WORD = 4 frames per word
MIN_SEGMENT = 0.4               # shortest speech segment
SEGMENT_MAX = 12.0              # training split anything longer
BURST_DEFAULT, BURST_SHORT, BURST_LONG = 0.28, 0.14, 0.48   # median / p10 / p90 of the corpus
BURST_MIN, BURST_MAX = 0.14, 1.2
PAUSE_DEFAULT, LONG_PAUSE, PAUSE_MIN = 0.3, 0.8, 0.2
LEAD_PAUSE = 0.3
STRETCH_MIN, STRETCH_MAX = 0.6, 1.5   # past ~1.5x the model fills time with invented words

# The 22 labels that occur in sft3's training data (model card) plus the
# better-realised classes LAION offers its own director (DIRECTOR.md §5.2).
# These are the sounds the director is told to use.
OFFERED_BURSTS = [
    "chuckle", "breathy giggle", "childlike giggle", "nervous giggle", "cackle", "guffaw",
    "contented sigh", "wistful sigh", "exasperated sigh", "relief sigh",
    "surprised gasp", "sharp inhale", "deep breath", "yawn", "ahem", "clears throat",
    "coughing", "exhausted groan", "low mumble", "soft hum", "resonant hum",
    "mournful wail", "scream", "shriek", "growl", "purr",
]
# Every class LAION has a burst adapter for (laion/vocal-burst-lora-adapters):
# the builder must recognise these as sounds even though the director is not
# offered the mouth / whistle ones (they never realise on this model).
KNOWN_BURSTS = set(OFFERED_BURSTS) | {
    "affirmative grunt", "convulsive sob", "cough", "deep breathing", "displeased grunt",
    "drinking noises", "effort grunt", "fast breathing", "fearful gasp", "frustrated groan",
    "gulps", "gurgling", "heavy breathing", "hiccup", "hiccups", "hiss", "humming",
    "kissing noises", "lip smack", "normal breathing", "pain moan", "panting",
    "person whistling playfully", "person whistling to get attention", "pleasure moan",
    "quiet sob", "sharp whistle", "slow breathing", "slurping noises", "smack one s lips",
    "snicker", "sniff", "snort", "snorting giggle", "sobs", "soft whistle", "spitting",
    "swallows", "trembling whimper", "wolf whistle",
}
# A short fragment ending on one of these words is a burst too ("a soft chuckle").
# Kept to the training-data core: matched against every class, ordinary
# directions such as "(panting after the stairs)" would turn into sounds.
_CORE_BURST_WORDS = {"mumble", "ahem", "sigh", "gasp", "chuckle", "giggle", "groan", "inhale",
                     "hum", "scream", "yawn", "breath", "cackle", "shriek", "coughing", "wail",
                     "growl", "purr"}
_LENGTH_WORDS = {"short": BURST_SHORT, "brief": BURST_SHORT, "quick": BURST_SHORT,
                 "tiny": BURST_SHORT, "long": BURST_LONG, "drawn-out": BURST_LONG,
                 "drawn out": BURST_LONG, "extended": BURST_LONG}

LANGUAGE_CODES = {"English": "EN", "German": "DE"}

TARGET_SFT3 = "LAION sft3 — timed script"
TARGET_INSTRUCTION = "LAION 8B / MOSS — instruction + text"
TARGETS = [TARGET_SFT3, TARGET_INSTRUCTION]
VOICE_DESCRIBE = "describe the voice"
VOICE_REFERENCE = "voice is fixed (reference clip / voice LoRA)"
VOICE_MODES = [VOICE_DESCRIBE, VOICE_REFERENCE]
WORDING_EXACT = "keep the words exactly"
WORDING_ADAPT = "may adapt the words for speech"
WORDING_MODES = [WORDING_EXACT, WORDING_ADAPT]
BURSTS_OFTEN = "often (natural speech)"
BURSTS_SPARING = "sparingly"
BURSTS_NONE = "none"
BURST_MODES = [BURSTS_OFTEN, BURSTS_SPARING, BURSTS_NONE]


# ---------------------------------------------------------------------------
# system prompts
# ---------------------------------------------------------------------------

_SFT3_PROMPT = """You are a voice director. You turn the text the user gives you into a performance script for an expressive text-to-speech model. You never produce audio; you write the script the model performs.

Reply with exactly this, and nothing else — no preamble, no explanation, no markdown, no code fences:

GENERAL: <one line>
SCRIPT:
<the script>

GENERAL — one line about the clip as a whole.
{voice_rule}
- Then the delivery: energy, pace, tension, warmth, breath, how close to the microphone.
- End with "reads as" and the emotions a listener should hear, e.g. "reads as amusement, relief".
- No numbers, no clip length, no language name — they are added for you.

SCRIPT — the words, performed.
{wording_rule}
- Before EVERY sentence put a delivery cue in round brackets saying how to speak it, in plain English prose about what the voice does: (clearly amused, letting it show). Short sentences of only a few words may share the cue of the sentence they belong with.
- Every cue names its strength with one adverb from these bands:
    held down: barely, faintly, only slightly, just a little
    plainly audible, controlled: clearly, plainly, noticeably, unmistakably
    running hard: strongly, intensely, very, deeply
    at the limit: overwhelmingly, extremely, utterly, completely
- Let the cues change through the text as the feeling changes; do not repeat one cue for everything.
{burst_rule}
- Beats of silence: [pause] for a short beat, [long pause] for a long one. Square brackets are ONLY for these two.
- Cues are always in English, even when the spoken words are not.

NEVER:
- a number anywhere inside any bracket — a round bracket with a number in it is read as a sound, not as a cue;
- seconds, durations, lengths or any other arithmetic — the timing is computed for you;
- capital-letter shouting (WHAT, AAAGH) — the model spells capitals out letter by letter; write the intensity in the cue instead;
- emoji, markdown, asterisks, quotation marks around the whole text, stage directions outside brackets — everything outside brackets is spoken aloud;
- underscores or identifiers inside a cue.

Example of a reply:

GENERAL: warm and conversational, quick and bright, the laugh sitting just under the words; reads as amusement
SCRIPT:
(clearly amused, letting it show) you are not going to believe what he did at the wedding. (chuckle) he actually wore the costume, the whole thing, with the hat. [pause] (quieter, warmer, only slightly embarrassed for him) and honestly, it suited him better than the suit would have.
"""

_SFT3_VOICE_DESCRIBE = ("- Start with the voice that fits the text: age, gender, timbre, pitch range, "
                        "clarity — e.g. \"a young adult feminine voice, bright, fairly smooth\".")
_SFT3_VOICE_REFERENCE = ("- Never describe the voice itself (age, gender, timbre, accent, who it is) — it "
                         "is fixed by a reference recording or a voice adapter, and describing it "
                         "fights them.")

_SFT3_WORDING_EXACT = ("- Keep the user's words exactly, in their order and language. You only add cues, "
                       "sounds and beats around them. Write normal sentence case.")
_SFT3_WORDING_ADAPT = ("- Keep the meaning and language of the user's text, but you may smooth the words "
                       "for natural speech: contractions, broken-up long sentences, natural fillers. Do not "
                       "add new content. Write normal sentence case.")

_SFT3_BURSTS = {
    BURSTS_OFTEN: (
        "- Vocal bursts — real non-speech sounds — each in its OWN round bracket, standing between "
        "sentences, never inside one, always with words after it: (chuckle). Use them often: real "
        "speech is full of them, and a reply without a breath, a laugh or a sigh sounds read rather "
        "than spoken. One in most texts, two or three when the moment is emotional.\n"
        "- A sound named inside a cue produces no sound: (amused, with a chuckle) gives no chuckle; "
        "(amused) ... (chuckle) ... gives one.\n"
        "- Write the sound, never the action: (chuckle), not (he chuckles). Prefix short or long if "
        "it matters: (long exasperated sigh). Never open or close the script on a sound, and never "
        "put [pause] right after one.\n"
        "- Use only these sounds, spelled exactly: {labels}."),
    BURSTS_SPARING: (
        "- Vocal bursts — real non-speech sounds — only where the text clearly calls for one, each "
        "in its OWN round bracket between sentences, with words after it: (wistful sigh). A sound "
        "named inside a cue produces nothing. Write the sound, never the action. Never open or "
        "close on a sound, never put [pause] right after one.\n"
        "- Use only these sounds, spelled exactly: {labels}."),
    BURSTS_NONE: "- No vocal bursts (laughs, sighs, breaths): only cues and beats.",
}

_INSTRUCTION_PROMPT = """You are a voice director. You turn the text the user gives you into an instruction for an expressive text-to-speech model, which then speaks the text. You never produce audio.

Reply with exactly this, and nothing else — no preamble, no explanation, no markdown, no code fences:

INSTRUCTION: <one or two sentences>
TEXT:
<the words to speak>

INSTRUCTION — how to perform the text, in plain English, whatever language the text is in.
{voice_rule}
- The emotion and how strongly it shows (barely / clearly / intensely / overwhelmingly), the character, the pace, the energy, whispering or shouting, breathiness — and how the delivery changes along the text if it does.
- e.g. "Excited and breathless, like sharing amazing news with a best friend, speeding up towards the end."

TEXT — {wording_rule}
- No brackets, no tags, no stage directions, no emoji, no markdown: everything here is spoken aloud.
- No capital-letter shouting (WHAT, AAAGH) — put the intensity into the instruction.
"""

_INSTR_VOICE_DESCRIBE = ("- Include the voice that fits: age, gender, timbre — e.g. \"a weary old sailor, "
                         "low and gravelly\".")
_INSTR_VOICE_REFERENCE = ("- Never describe the voice itself (age, gender, timbre) — it is fixed by a "
                          "reference recording or a voice adapter. Describe only the performance.")
_INSTR_WORDING_EXACT = "the user's words exactly, in their order and language."
_INSTR_WORDING_ADAPT = ("the user's text in its language and meaning, smoothed for natural speech if "
                        "needed; no new content.")


def system_prompt(target: str = TARGET_SFT3, voice: str = VOICE_DESCRIBE,
                  wording: str = WORDING_EXACT, bursts: str = BURSTS_OFTEN,
                  extra: str = "") -> str:
    """The system prompt for the director LLM."""
    if target == TARGET_INSTRUCTION:
        text = _INSTRUCTION_PROMPT.format(
            voice_rule=_INSTR_VOICE_REFERENCE if voice == VOICE_REFERENCE else _INSTR_VOICE_DESCRIBE,
            wording_rule=_INSTR_WORDING_ADAPT if wording == WORDING_ADAPT else _INSTR_WORDING_EXACT,
        )
    else:
        burst_rule = _SFT3_BURSTS.get(bursts, _SFT3_BURSTS[BURSTS_OFTEN])
        text = _SFT3_PROMPT.format(
            voice_rule=_SFT3_VOICE_REFERENCE if voice == VOICE_REFERENCE else _SFT3_VOICE_DESCRIBE,
            wording_rule=_SFT3_WORDING_ADAPT if wording == WORDING_ADAPT else _SFT3_WORDING_EXACT,
            burst_rule=burst_rule.replace("{labels}", ", ".join(OFFERED_BURSTS)),
        )
        if bursts == BURSTS_NONE:  # the example would teach the opposite
            text = text.replace(" (chuckle) he actually", " he actually")
    extra = (extra or "").strip()
    if extra:
        text += f"\nAdditional notes from the user, which take precedence over style choices above:\n{extra}\n"
    return text


# ---------------------------------------------------------------------------
# reading the director's reply
# ---------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^\s*```[\w-]*\s*$", re.M)
_FIELD = re.compile(r"^\s*\**\s*(GENERAL|SCRIPT|INSTRUCTION|TEXT)\s*\**\s*:\s*", re.M | re.I)


def split_fields(reply: str) -> dict:
    """{'general'|'script'|'instruction'|'text': value} from a director reply;
    thinking and code fences are dropped. No labelled field -> {'body': reply}."""
    s = _THINK.sub("", str(reply or ""))
    s = _FENCE.sub("", s).strip()
    found = list(_FIELD.finditer(s))
    if not found:
        return {"body": s}
    out = {}
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(s)
        out[m.group(1).lower()] = s[m.end():end].strip()
    return out


# ---------------------------------------------------------------------------
# the timed script
# ---------------------------------------------------------------------------

_NUM = r"([0-9]*\.?[0-9]+)\s*(?:s|sec|secs|seconds?)?"
_TAG = re.compile(r"\[([^\]]*)\]|\(([^)]*)\)")
_DURATION_IN = re.compile(rf"^\s*{_NUM}\s+duration\s*$", re.I)
_PAUSE_IN = re.compile(rf"^\s*(?:{_NUM}\s+)?(long\s+)?(?:pause|beat|silence)\s*$", re.I)
_BURST_WITH_NUM = re.compile(rf"^(.*?),\s*{_NUM}\s*$", re.I)
_CAPS = re.compile(r"\b[A-ZÄÖÜ]{3,}\b")


def _burst_label(body: str):
    """(label, seconds) when a number-less round bracket names a sound, else None."""
    t = " ".join(body.lower().strip().rstrip(".!").split())
    secs = BURST_DEFAULT
    for word in sorted(_LENGTH_WORDS, key=len, reverse=True):
        if t.startswith(word + " "):
            secs, t = _LENGTH_WORDS[word], t[len(word) + 1:]
            break
    plain = t.replace("_", " ")
    if plain in KNOWN_BURSTS:
        return plain, secs
    words = plain.split()
    if 0 < len(words) <= 4 and words[-1] in _CORE_BURST_WORDS:
        return plain, secs
    return None


def parse(script: str) -> list:
    """Director script -> [('speech', text) | ('direction', text) |
    ('burst', label, secs) | ('pause', secs, explicit) | ('duration', secs)]."""
    s = _CAPS.sub(lambda m: m.group(0).lower(), str(script or ""))
    items, pos = [], 0
    for m in _TAG.finditer(s):
        if s[pos:m.start()].strip():
            items.append(("speech", s[pos:m.start()]))
        pos = m.end()
        square, round_ = m.group(1), m.group(2)
        body = " ".join((square if square is not None else round_).split())
        if not body:
            continue
        dur = _DURATION_IN.match(body)
        if dur:                                    # a duration the user wrote, in either bracket
            items.append(("duration", float(dur.group(1))))
            continue
        pause = _PAUSE_IN.match(body)
        if pause:
            secs = float(pause.group(1)) if pause.group(1) else (LONG_PAUSE if pause.group(2) else PAUSE_DEFAULT)
            items.append(("pause", secs, bool(pause.group(1))))
            continue
        with_num = _BURST_WITH_NUM.match(body) if round_ is not None else None
        if with_num:
            items.append(("burst", with_num.group(1).strip(),
                          min(max(float(with_num.group(2)), BURST_MIN), BURST_MAX)))
            continue
        # a sound named in square brackets is still a sound; anything else in
        # square brackets is read the round-bracket way, never dropped
        burst = _burst_label(body)
        items.append(("burst", *burst) if burst else ("direction", body))
    if s[pos:].strip():
        items.append(("speech", s[pos:]))
    return items


def _sentences(text: str) -> list:
    text = " ".join(text.split())
    return [p for p in re.split(r"(?<=[.!?…])\s+", text) if p]


def _speech_seconds(words: int, speed: float) -> float:
    return max(MIN_SEGMENT, words * SECONDS_PER_WORD / max(speed, 0.1))


def _split_long(sentence: str, secs: float) -> list:
    """A segment over SEGMENT_MAX -> evenly sized word chunks, preferring commas."""
    n = math.ceil(secs / SEGMENT_MAX)
    if n <= 1:
        return [sentence]
    words = sentence.split()
    size = math.ceil(len(words) / n)
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= size or (len(cur) >= size * 0.6 and w.endswith((",", ";", ":"))):
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class TimedScript:
    script: str          # goes into SCRIPT: and Text:, byte-identical
    frames: int          # Tokens:
    seconds: float
    spoken: str          # the words alone
    notes: list          # what was adjusted


def render(script: str, speed: float = 1.0, target_seconds: float = 0.0) -> TimedScript:
    """The sft3 timed script for a director script (numbers optional)."""
    notes = []
    seq = []                 # [kind, payload, seconds]
    cue, want = None, None
    for item in parse(script):
        kind = item[0]
        if kind == "duration":
            want = item[1]
        elif kind == "direction":
            cue = item[1] if cue is None else f"{cue}; {item[1]}"
        elif kind == "pause":
            # LAION's director rule: no beat straight after a sound. A pause
            # with its own number was written on purpose and stays.
            if seq and seq[-1][0] == "burst" and not item[2]:
                notes.append("dropped a [pause] right after a sound")
                continue
            if seq and seq[-1][0] == "pause":
                seq[-1][2] += item[1]
            else:
                seq.append(["pause", None, item[1]])
        elif kind == "burst":
            if cue is not None:
                seq.append(["direction", cue, 0.0])
                cue = None
            seq.append(["burst", item[1], item[2]])
        else:
            for sent in _sentences(item[1]):
                words = len(sent.split())
                secs = _speech_seconds(words, speed)
                if want is not None:   # honoured within what the model can carry
                    got = min(max(want, secs * STRETCH_MIN), secs * STRETCH_MAX)
                    if abs(got - want) > 0.05:
                        notes.append(f"{want:.1f}s asked for {words} words — using {got:.1f}s")
                    secs, want = got, None
                if cue is not None:
                    seq.append(["direction", cue, 0.0])
                    cue = None
                chunks = _split_long(sent, secs)
                for chunk in chunks:
                    seq.append(["speech", chunk, secs * len(chunk.split()) / words])
    if cue is not None:
        notes.append(f"dropped a cue with no words after it: ({cue})")
    if not any(k == "speech" for k, _, _ in seq):
        return TimedScript("", 0, 0.0, "", notes + ["no words to speak"])
    if seq[0][0] != "pause":
        seq.insert(0, ["pause", None, LEAD_PAUSE])
    while seq[-1][0] == "pause":                  # never end on silence the model must fill
        seq.pop()
    for s in seq:
        if s[0] == "pause":
            s[2] = max(s[2], PAUSE_MIN)

    if target_seconds and target_seconds > 0:
        fixed = sum(s[2] for s in seq if s[0] != "speech")
        spoken = sum(s[2] for s in seq if s[0] == "speech")
        factor = (target_seconds - fixed) / spoken if spoken > 0 else 1.0
        clamped = min(max(factor, STRETCH_MIN), STRETCH_MAX)
        if abs(clamped - factor) > 1e-6:
            notes.append(f"target {target_seconds:.1f}s needs speech x{factor:.2f}; "
                         f"limited to x{clamped:.2f} (beyond that the model invents words)")
        for s in seq:
            if s[0] == "speech":
                s[2] *= clamped

    out, plain, total = [], [], 0.0
    for kind, payload, secs in seq:
        shown = round(secs, 1)
        if kind == "pause":
            out.append(f"[{shown:.1f} seconds pause]")
        elif kind == "burst":
            out.append(f"({payload}, {shown:.1f} seconds)")
        elif kind == "direction":
            out.append(f"({payload})")
            shown = 0.0
        else:
            shown = max(shown, 0.1)
            out.append(f"[{shown:.1f} seconds duration] {payload}")
            plain.append(payload)
        total += shown
    total = round(total, 1)
    return TimedScript(" ".join(out), int(round(total * FRAME_RATE)), total, " ".join(plain), notes)


def general_line(general: str, seconds: float, language: str) -> str:
    """GENERAL folded to the card's one-line shape: '…; reads as …; 6.0s, EN.'"""
    g = " ".join(str(general or "").split()).strip(" ;.,")
    code = LANGUAGE_CODES.get(language, (language or "EN")[:2].upper())
    tail = f"{seconds:.1f}s, {code}."
    return f"{g}; {tail}" if g else tail


@dataclass
class Built:
    instruction: str
    text: str
    target_tokens: int
    seconds: float
    report: str


def build(reply: str, fmt: str = "auto", language: str = "English", speed: float = 1.0,
          target_seconds: float = 0.0) -> Built:
    """Director reply -> (instruction, text, target_tokens) for the MOSS nodes.

    fmt: "auto" (by the reply's fields), "sft3 timed script" or "instruction + text".
    """
    fields = split_fields(reply)
    if fmt == "auto":
        fmt = "instruction + text" if ("instruction" in fields or "text" in fields) and "script" not in fields \
            else "sft3 timed script"
    if fmt == "instruction + text":
        instruction = fields.get("instruction", "")
        text = " ".join(fields.get("text", fields.get("body", "")).split())
        if not text:
            raise ValueError("Script Builder: no TEXT in the director reply.")
        return Built(instruction, text, 0, 0.0, "instruction + text: no timing (the model decides the length)")
    script = fields.get("script", fields.get("body", ""))
    timed = render(script, speed=speed, target_seconds=target_seconds)
    if not timed.script:
        raise ValueError("Script Builder: the script has no words to speak "
                         f"(got: {script[:200]!r}).")
    general = general_line(fields.get("general", ""), timed.seconds, language)
    instruction = f"GENERAL: {general}\nSCRIPT:\n{timed.script}"
    lines = [f"sft3 timed script: {timed.seconds:.1f}s = {timed.frames} frames"] + timed.notes
    return Built(instruction, timed.script, timed.frames, timed.seconds, "\n".join(lines))
