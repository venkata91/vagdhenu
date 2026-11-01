# Vāgdhenu — Sanskrit Chant TTS

*"The wish-cow of speech."* A production-grade, single-speaker **Sanskrit chant (pārāyaṇa) text-to-speech** system — it *chants* classical ślokas with metrically-aware durations and tradition-faithful melodic contour, not flat read-aloud.

> **MOS ~4.6** (expert listener). Conjuncts — including retroflex aspirates (ṣṭ, ḍḍh, …) — render 100% correctly, the class earlier architectures could not crack. Used to produce **MBTN** (32 YouTube videos, 17h 34m) and the **Śrīmad Bhāgavatam** (16,017 verses, audio app + 31 karaoke videos).

[ **[Project page + live demo](https://prathosh.in/vagdhenu/)** · [Model weights → HF](https://huggingface.co/prathoshap/vagdhenu) · [Demo → HF Space](https://huggingface.co/spaces/prathoshap/vagdhenu-demo) · Tech report → `docs/TECH_REPORT.md` ]

## Demos (rendered with this system)
- **Mahābhārata Tātparya Nirṇaya (MBTN)** — full chant series: [YouTube playlist](https://www.youtube.com/playlist?list=PLL1s8qiaGy0IP0G_PhlwaGA5EOfzoKrV_)
- **Śrīmad Bhāgavatam** — karaoke-video series: [YouTube playlist](https://www.youtube.com/playlist?list=PLDiYyVdyo2Sc)

Developed and maintained by **Prof. Prathosh, Indian Institute of Science, Bengaluru.**

## How it works
- **Backbone:** IndicF5 / F5-TTS — a flow-matching **DiT** (OT-CFM mel-infilling, ~337M params, *no* native duration or pitch head). Sanskrit is routed through **Kannada script** (Devanagari triggers Hindi schwa-deletion).
- **Vocoder:** NVIDIA **BigVGAN-v2**, fine-tuned on F5 vocos-mel (mandatory — vocos shivers on long vowels).
- **Prosody:** F5's content fidelity is bulletproof but its prosody is *text-driven, not designable*. The working levers are **the reference clip** (voice + swara + pace, via the *half-reference rule*) and a **voice-steering fine-tune**. (See `docs/TECH_REPORT.md` §14 for the full account — this is the central architectural finding.)
- **Text frontend (`src/prep_text.py`)** — the most reusable piece: Deva→SLP1→Kannada routing, internal visarga sandhi (utva/rutva/lopa/satva), homorganic anusvāra, vocalic-ṝ handling, daṇḍa-final rules, meter/gaṇa (L/G) detection.

## Layout
```
src/         text frontend, meter detection, inference, post-gate, reference bank
             (tamil_*.py + reference_bank/bank_ta.json: experimental Tamil path)
pipeline/    data-prep (cut→pair→train) + build/assemble/QC
demo/        Gradio app (HF ZeroGPU)
docs/        scrubbed technical report + frontend/pipeline references
examples/    sample inputs + rendered outputs (examples/tamil/: Tiruppallāṇḍu fixture + santhai plan/shard)
tests/       CPU-only pytest suite (models stubbed) + Sanskrit golden snapshot
scripts/     env setup + weight download
```

## Install & quickstart
Requires **Python 3.10** and a **CUDA 12.1 GPU**.
```bash
bash scripts/setup.sh    # torch+cu121, deps, BigVGAN, and downloads weights -> models/
# render a Devanagari verse (+ meter) to a chanted wav:
python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out
# -> out/sample_anushtubh.wav
```
The batch renderer takes a shard JSON: `[{"id","meter","padas":[devanagari…],"seed","out"}]`. For one-off single-verse renders see `src/render_production.py`. `CHAMP_ROOT` env overrides the weights dir (default `models/`).

## Case studies
- **MBTN** (Mahābhārata Tātparya Nirṇaya) — 32-adhyāya *video* deliverable (Devanagari + Kannada karaoke, tanpura), shipped.
- **Śrīmad Bhāgavatam** — 12 skandhas, ~18k verses, *audio* app + a 31-video 3-script (Devanāgarī · Kannada · IAST) karaoke series. Sanskrit text gratefully acknowledged to **Poornaprajna Samshodhana Mandiram, Bengaluru**.

## Attribution & licenses
- Code: **Apache-2.0** (`LICENSE`).
- Built on **AI4Bharat IndicF5** (MIT), **NVIDIA BigVGAN-v2**, and **F5-TTS** — see their licenses; weights redistributed per those terms.
- Model weights + intended-use/ethics note: see the HF model card.

## Tamil (experimental)
The renderer can also chant Tamil Śrīvaiṣṇava pāsurams, starting with Periyāḻvār's Tiruppallāṇḍu (`examples/tamil/tiruppallandu.txt`). It's off unless you ask for it. Only shard entries carrying `"lang": "ta"` take the Tamil path, and those come from `src/tamil_plan.py`. Nothing auto-detects Tamil, and Sanskrit renders are byte-for-byte what they were.

Be clear about where this stands. No Tamil reference audio ships with the repo, and nobody has listened to Tamil output from this model yet, so the quality is unknown. Until the two reference clips below exist, every Tamil clip fails with an error naming the missing entry and wav.

**Fixture format.** UTF-8 text in the source edition's spelling (Grantha letters with sub/superscript marks are allowed in display text).
- `#` starts a comment line. `@key: value` header lines (`title`, `work_id`, `source`, …) come before the first section.
- Sections: `[invocation]`, `[taniyan N]`, `[pasuram N]`, `[adivaravu]`, `[colophon]`. Only pāsurams are rendered by default; the `sections` config key can add taniyans.
- Inside a pāsuram, each line is one recitation segment, the `⋆`-segment of the edition. The repeat mark for pāsurams recited twice may open the first line. It is `‡` by default; set the header `@repeat_mark: <symbol>` for editions that use another symbol. The mark is stripped from the text. A trailing `॥ N ॥` must match the section number.
- `⋆` and `॥ N ॥` stay in the display text; the repeat mark is recorded as `marked` in the plan and manifest. The model only sees Tamil letters: markup and digits are stripped, `ஶ` becomes `ஸ`, and anything outside the model's Tamil vocabulary is an error naming the segment.

**Modes** (`--mode`):
- `single`: one voice, one unit per segment.
- `santhai`: the ācārya says a segment once, then the śiṣyas repeat it `santhai_repeats` times (default 2).
- `goshti`: everyone together in one voice. With `alternate: true` units are labelled group A or B, switching per segment or per pāsuram (`alternate_unit`), and each group can use its own reference via `refs`.
- `repeat_marked` (default `true`) recites the marked pāsurams a second time; set it to `false` to recite each pāsuram once.

Gaps between units come from a `--config` JSON and default to: segment 0.55 s, end of pāsuram 0.95 s, ācārya→śiṣya 0.8 s, śiṣya→śiṣya 0.6 s, group switch 0.4 s. `python src/tamil_plan.py --help` lists every config key.

**Adding the two reference clips** (no code change). Record a santhai clip and a goṣṭhi clip of your own voice, each 15 s or shorter (the half-reference rule). Drop them into `src/reference_bank/` as `ta_santhai.wav` and `ta_goshti.wav`, then fill `ref_text` (exactly the Tamil words you hear in the clip) and `sec_per_syll` in `src/reference_bank/bank_ta.json`. The `_template` entry in that file shows every field. To measure `sec_per_syll`:
```bash
python src/tamil_bank.py sps --wav src/reference_bank/ta_santhai.wav --ref_text 'பல்லாண்டு பல்லாண்டு …'
# {"dur_s": …, "nsyll": …, "sps": …, "warning": null}   (duration / Tamil syllables x 1.10; warns over 15 s)
```

**Running it.** Planning and assembly run on CPU; only `render.py` needs the GPU.
```bash
# 1. CPU: fixture -> plan + shard (examples/tamil/ has the santhai output committed)
python src/tamil_plan.py --fixture examples/tamil/tiruppallandu.txt --mode santhai \
    --unit_dir out/tamil_units --plan_out out/tp.plan.json --shard_out out/tp.shard.json
# 2. GPU: one wav per unit (Tamil refs come from --bank_ta, default src/reference_bank/bank_ta.json)
python src/render.py --shard out/tp.shard.json --results out/tp.results.json
# 3. CPU: units -> one wav per pāsuram + manifest (display text, per-unit timings)
python src/tamil_assemble.py --plan out/tp.plan.json --units_dir out/tamil_units --outdir out/tamil
```

**Tests.** The suite is CPU only. torch, F5 and BigVGAN are stubbed and never imported, and a golden snapshot pins the Sanskrit outputs.
```bash
python3 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Ethics / intended use
Single-speaker synthesis of sacred Sanskrit recitation, for pārāyaṇa/study/accessibility. The voice is the author's own. Please use responsibly; do not impersonate.

The same applies to Tamil. Tamil references must be the author's own consented recordings. Don't use anyone else's voice, and don't present the output as a real ācārya, goṣṭhi or temple recitation.

## Citation
*(BibTeX added with the arXiv report.)*
