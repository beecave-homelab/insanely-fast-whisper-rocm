# pysubs2 — Technical Report

## 1) What it is (and why it fits)

**pysubs2** is a mature, MIT-licensed Python library for **editing, retiming, and converting** subtitle files. It exposes a clean object model (`SSAFile`, `SSAEvent`, `SSAStyle`) and includes a small CLI for quick conversions/retimes. As a toolbox, it’s ideal when *you* want to implement house-style rules (CPL/CPS, line breaks, min/max durations) on top. ([pysubs2.readthedocs.io][1])

### Supported formats (IO)

ASS/SSA, SRT, MicroDVD, MPL2, TMP, **WebVTT**, **TTML**, **SAMI**, plus **OpenAI Whisper** inputs (incl. `load_from_whisper`). This covers typical ASR→SRT/VTT flows and gives you a path for richer styling via ASS if needed. ([pysubs2.readthedocs.io][2])

---

## 2) Capabilities relevant to Whisper-based ASR

### a) Direct Whisper ingestion

* `pysubs2.load_from_whisper(result_or_segments)` converts Whisper API results (or segment lists) to an editable `SSAFile`. Great for starting from your current pipeline and then applying your own segmentation/wrapping logic. ([pysubs2.readthedocs.io][3])

### b) Clean, Pythonic API surface

* **Core classes:** `SSAFile` (the subtitle doc), `SSAEvent` (a cue), `SSAStyle` (formatting). Times are **int milliseconds**; helpers like `make_time()` save you manual conversions. ([pysubs2.readthedocs.io][4])
* **Retiming:** `SSAFile.shift(...)`, `SSAFile.transform_framerate(in_fps, out_fps)`. Per-event shift via `SSAEvent.shift(...)`. ([pysubs2.readthedocs.io][3])
* **Sorting & housekeeping:** `SSAFile.sort()`, `SSAFile.remove_miscellaneous_events()` (what `--clean` does in the CLI). ([pysubs2.readthedocs.io][3])
* **Styles:** `SSAStyle` with alignment, colors, margins; SRT export can keep `<i>` etc., and there are options to keep/convert tags when moving between formats. ([pysubs2.readthedocs.io][4])

### c) CLI for quick ops (nice to have)

Batch convert/retime, keep or strip tags, transform framerate; useful when smoke-testing formatter outputs outside your app:
`pysubs2 --to srt *.ass`, `pysubs2 --shift 0.3s *.srt`, `--clean`, `--srt-keep-html-tags`, `--srt-keep-ssa-tags`. ([pysubs2.readthedocs.io][5])

### d) Encoding ergonomics

Defaults to UTF-8; you can specify encodings or pass non-UTF-8 through with `errors="surrogateescape"` (the CLI does this by default). This avoids messy decode failures with legacy files. ([pysubs2.readthedocs.io][4])

---

## 3) Known limits / gotchas (so you plan properly)

* **No built-in CPL/CPS/“linguistic” segmentation.** pysubs2 doesn’t enforce industry-style formatting by itself—you implement splits/merges and wrapping logic using `SSAEvent.text`, `start`, `end`. (That’s by design; it’s a substrate.) ([pysubs2.readthedocs.io][4])
* **WebVTT features are basic.** VTT is handled “as a flavour of SRT”; **no VTT-specific styling/alignment** support. If you need positioning/alignment, generate **ASS** for authoring, then down-convert as needed. ([pysubs2.readthedocs.io][2])
* **TTML/SAMI are basic.** TTML write support is limited; SAMI parser is rudimentary—fine for simple conversions, not for complex styling. ([pysubs2.readthedocs.io][2])

---

## 4) Minimal integration blueprint (chunk- or word-level)

> Goal: take Whisper (or stable-ts/WhisperX) output and shape it into **best-practice** cues with pysubs2.

1. **Ingest**

   * From Whisper Python result: `subs = pysubs2.load_from_whisper(result)`; or build events from your own (segment/word) objects. ([pysubs2.readthedocs.io][3])

2. **Resegment** (your logic)

   * Iterate segment/word stream; split on **punctuation/pauses**, clamp **duration** to ~0.8–7.0 s, ensure **≤2 lines**, wrap to **~35–42 CPL**, and enforce **~12–17 CPS** by adjusting duration or splitting. (Compute CPS = chars/duration_s; CPL = max line length.)
   * Use `SSAEvent.plaintext` to work without SSA tags; write back to `text` or `plaintext`. ([pysubs2.readthedocs.io][4])

3. **Polish timing**

   * Snap boundaries to silence gaps if you have them (from stable-ts/WhisperX) and call `event.shift(...)` or tweak `start/end`. Sort at the end: `subs.sort()`. ([pysubs2.readthedocs.io][3])

4. **Export**

   * `subs.save("out.srt")` or `subs.save("out.vtt", format_="vtt")`. Remember VTT styling limits. ([pysubs2.readthedocs.io][3])

### Tiny starter skeleton (drop-in)

```python
import pysubs2
from math import ceil

MAX_CPL = 40
MIN_DUR = 800       # ms
MAX_DUR = 7000      # ms
TARGET_CPS = 13

def wrap_lines(text, max_cpl=MAX_CPL):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + (1 if line else 0) + len(w) <= max_cpl:
            line = w if not line else f"{line} {w}"
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    return lines[:2], len(lines) > 2  # hard cap to 2 lines

def cps(chars, dur_ms): return chars / max(dur_ms, 1) * 1000

def build_event(start_ms, end_ms, text):
    dur = max(MIN_DUR, min(MAX_DUR, end_ms - start_ms))
    lines, truncated = wrap_lines(text)
    txt = "\\N".join(lines)  # SubStation newline; SRT writer converts to <br>/<i> etc. as needed
    ev = pysubs2.SSAEvent(start=start_ms, end=start_ms + dur, text=txt)
    # if CPS too high, split here (left to the reader: split by words and recurse)
    if cps(len(" ".join(lines)), dur) > TARGET_CPS:
        # split logic: find a mid point, create two events
        pass
    return [ev]

subs = pysubs2.SSAFile()
for seg in whisper_segments:  # or your stable-ts / WhisperX segments with words
    subs.events += build_event(int(seg.start * 1000), int(seg.end * 1000), seg.text)

subs.sort()
subs.save("out.srt")
```

> Notes
>
> * Use **word timestamps** (stable-ts or WhisperX) to split on word boundaries for better CPL/CPS control and natural breaks; pysubs2 will happily accept the events you create.
> * For HTML/SSA mixed tags in SRT, see the CLI/API options to *keep* tags when converting. ([pysubs2.readthedocs.io][5])

---

## 5) Practical “got-it” checklist

* **Install:** `pip install pysubs2` (Python ≥3.9). ([PyPI][6])
* **Whisper import:** `pysubs2.load_from_whisper(...)`. ([pysubs2.readthedocs.io][3])
* **Shift / retime:** `subs.shift(s=±0.5)`; **framerate fix:** `subs.transform_framerate(25, 23.976)`. ([pysubs2.readthedocs.io][3])
* **Clean extras:** `subs.remove_miscellaneous_events()` or CLI `--clean`. ([pysubs2.readthedocs.io][3])
* **VTT caveat:** VTT export has **no advanced styling/alignment** in pysubs2—treat it like SRT. ([pysubs2.readthedocs.io][2])
* **Encoding quirks:** specify `encoding=` or use `errors="surrogateescape"` pass-through. ([pysubs2.readthedocs.io][4])

---

## 6) When you might add another tool

* Need **stable word times**? Pair with **stable-ts** or **WhisperX** and then feed words into your pysubs2 formatter. (pysubs2 itself does not compute word timings.)
* Need **ASS effects/advanced positioning**? pysubs2 can author **ASS**; use Aegisub for visual QA, then down-convert if distribution requires SRT/VTT. ([pysubs2.readthedocs.io][4])

---

## 7) Bottom line

pysubs2 gives you **solid IO + timing primitives** and a **clean Python API** to build your **own** best-practice subtitle formatter/segmenter for Whisper outputs. You’ll implement CPL/CPS and linguistic splits yourself, but everything else—loading, retiming, sorting, converting, preserving tags, and exporting—is already there and well-documented. ([pysubs2.readthedocs.io][1])

---

### Key references

* Docs homepage & tutorial (API, timing, styles, examples). ([pysubs2.readthedocs.io][1])
* Supported formats (incl. VTT/TTML/SAMI/Whisper, and VTT limitations). ([pysubs2.readthedocs.io][2])
* CLI usage & options (`--clean`, tag-keep flags, transform-framerate). ([pysubs2.readthedocs.io][5])
* API reference (`load_from_whisper`, `shift`, `transform_framerate`, `remove_miscellaneous_events`). ([pysubs2.readthedocs.io][3])
* PyPI (version, Python≥3.9, install). ([PyPI][6])

If you want, I can turn this into a **drop-in formatter module** that accepts Whisper/WhisperX/stable-ts JSON and emits SRT/VTT with your exact rules.

[1]: https://pysubs2.readthedocs.io/?utm_source=chatgpt.com "pysubs2 — pysubs2 1.8.0 documentation"
[2]: https://pysubs2.readthedocs.io/en/latest/supported-formats.html "Supported File Formats — pysubs2 1.8.0 documentation"
[3]: https://pysubs2.readthedocs.io/en/latest/api-reference.html "API Reference — pysubs2 1.8.0 documentation"
[4]: https://pysubs2.readthedocs.io/en/latest/tutorial.html "API tutorial: Let’s import pysubs2 — pysubs2 1.8.0 documentation"
[5]: https://pysubs2.readthedocs.io/en/latest/cli.html "Using pysubs2 from the Command Line — pysubs2 1.8.0 documentation"
[6]: https://pypi.org/project/pysubs2/ "pysubs2 · PyPI"
