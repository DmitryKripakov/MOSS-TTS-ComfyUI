"""Director prompt + timed-script builder (director.py): checked against the
two worked examples LAION publishes for sft3 — the model card's and the
DIRECTOR.md one, which is real `timed_script.render()` output."""

from __future__ import annotations

import re

import pytest

from moss_tts import director as d


def _adds_up(timed: str, frames: int) -> bool:
    secs = sum(float(x) for x in re.findall(r"\[([0-9.]+) seconds (?:duration|pause)\]", timed))
    secs += sum(float(x) for x in re.findall(r", ([0-9.]+) seconds\)", timed))
    return abs(secs * d.FRAME_RATE - frames) <= 0.5


def test_laion_director_worked_example():
    reply = """GENERAL: intensely amused, the laugh sitting just under the words; reads as amusement
SCRIPT:
(clearly amused, letting it show) i genuinely did not think you were going to
say that out loud. (chuckle) but you did, and now we both have to live with it.
[pause] (quieter, warmer, taking his time) give me a second to think about it
properly."""
    b = d.build(reply)
    assert b.text == (
        "[0.3 seconds pause] (clearly amused, letting it show) [4.2 seconds duration] i genuinely did "
        "not think you were going to say that out loud. (chuckle, 0.3 seconds) [3.8 seconds duration] "
        "but you did, and now we both have to live with it. [0.3 seconds pause] (quieter, warmer, "
        "taking his time) [2.9 seconds duration] give me a second to think about it properly.")
    assert b.target_tokens == 148 and b.seconds == 11.8
    assert b.instruction == ("GENERAL: intensely amused, the laugh sitting just under the words; reads as "
                             "amusement; 11.8s, EN.\nSCRIPT:\n" + b.text)


def test_model_card_example_round_trips():
    card = """[0.4 seconds pause] (intensely amused: letting it out / not hiding it, warm and open,
unguarded; bright, relaxed) [2.4 seconds duration] You are not going to believe this.
[0.3 seconds pause] (breathy giggle, 0.4 seconds) [0.2 seconds pause] (still intensely amused)
[2.3 seconds duration] He actually wore it to the wedding."""
    t = d.render(card)
    assert t.script == " ".join(card.split())
    assert t.frames == 75 and t.notes == []


@pytest.mark.parametrize("cue, burst", [
    ("chuckle", ("chuckle", 0.28)),
    ("a soft chuckle", ("a soft chuckle", 0.28)),
    ("long exasperated sigh", ("exasperated sigh", 0.48)),
    ("short sniff", ("sniff", 0.14)),
    ("clears_throat", ("clears throat", 0.28)),
    ("clearly amused, with a small chuckle", None),     # a burst inside a cue is a cue
    ("panting after the stairs", None),
    ("quietly, warmly", None),
])
def test_burst_or_direction(cue, burst):
    assert d._burst_label(cue) == burst


def test_rules_from_the_director_manual():
    t = d.render("(chuckle) [pause] HELLO there, my friend, how are you. (quietly, 2 seconds) yes. (sigh)")
    assert t.script.startswith("[0.3 seconds pause] (chuckle, 0.3 seconds) [")   # no beat after a sound
    assert "hello there" in t.script                                              # caps spelled out otherwise
    assert "(quietly, 1.2 seconds)" in t.script   # a number turns a cue into a sound, clamped to 1.2 s
    assert t.script.endswith("(sigh, 0.3 seconds)")      # closing sound kept, trailing silence never
    assert _adds_up(t.script, t.frames)
    t = d.render("[a soft sigh] fine, let us go then, all of us together. (with a grin)")
    assert t.script.startswith("[0.3 seconds pause] (a soft sigh, 0.3 seconds)")  # square -> still a sound
    assert any("no words after it" in n for n in t.notes)


def test_long_sentences_split_and_budget():
    long = " ".join(["word"] * 60) + "."                 # 60 words = 19.2 s
    t = d.render("(calmly) " + long)
    segs = [float(x) for x in re.findall(r"\[([0-9.]+) seconds duration\]", t.script)]
    assert len(segs) == 2 and max(segs) <= d.SEGMENT_MAX and _adds_up(t.script, t.frames)
    base = d.render("(calmly) one two three four five six seven eight nine ten.")
    slow = d.render("(calmly) one two three four five six seven eight nine ten.", target_seconds=4.5)
    assert slow.seconds == pytest.approx(4.5, abs=0.11) and _adds_up(slow.script, slow.frames)
    capped = d.render("(calmly) one two three four five six seven eight nine ten.", target_seconds=30)
    assert capped.seconds < 6 and any("limited" in n for n in capped.notes)   # x1.5 at most
    assert base.seconds == pytest.approx(0.3 + 3.2, abs=0.05)
    assert d.render("(calmly) one two three four five.", speed=2.0).seconds == pytest.approx(0.3 + 0.8)


def test_reply_parsing_and_instruction_format():
    reply = "<think>let me see</think>\n```\nGENERAL: tired\nSCRIPT:\n(barely awake) good morning to you all.\n```"
    b = d.build(reply, language="German")
    assert b.instruction.startswith("GENERAL: tired; ") and ", DE.\nSCRIPT:\n[0.3 seconds pause]" in b.instruction
    b = d.build("INSTRUCTION: Excited and breathless.\nTEXT:\nWE won\nthe cup!")
    assert (b.instruction, b.text, b.target_tokens) == ("Excited and breathless.", "WE won the cup!", 0)
    assert d.build("(gently) just a script, no labels at all here.").target_tokens > 0
    with pytest.raises(ValueError, match="no words"):
        d.build("GENERAL: x\nSCRIPT:\n(chuckle) [pause]")


def test_system_prompts():
    p = d.system_prompt()
    assert "GENERAL:" in p and "SCRIPT:" in p and "breathy giggle" in p and "(chuckle)" in p
    assert "a young adult feminine voice" in p and "never describe" not in p.lower()
    p = d.system_prompt(voice=d.VOICE_REFERENCE, bursts=d.BURSTS_NONE, extra="pirate captain")
    assert "Never describe the voice itself" in p and "(chuckle)" not in p and p.rstrip().endswith("pirate captain")
    p = d.system_prompt(target=d.TARGET_INSTRUCTION, wording=d.WORDING_ADAPT)
    assert "INSTRUCTION:" in p and "SCRIPT:" not in p and "smoothed" in p
    assert all(b in d.KNOWN_BURSTS for b in d.OFFERED_BURSTS)


def test_director_nodes():
    import moss_tts.nodes as nodes

    (prompt,) = nodes.MossTTSV15DirectorPrompt().build(d.TARGET_SFT3, d.VOICE_DESCRIBE,
                                                       d.WORDING_EXACT, d.BURSTS_OFTEN)
    assert prompt == d.system_prompt()
    out = nodes.MossTTSV15ScriptBuilder().build(
        "GENERAL: calm\nSCRIPT:\n(calmly) this is a test of the script builder node.", "auto",
        "English", 1.0, 0.0)
    instruction, text, tokens, seconds, report = out
    assert instruction.endswith(text) and tokens == round(seconds * 12.5) and "frames" in report
