# Switchboard Phase 1 — Trainability Report

**Date:** 2026-09-05 · **Target student:** `Qwen/Qwen3.8-27B` (hybrid Gated-DeltaNet + attention)
**Source:** `data/switchboard/phase1_sft.jsonl` (202 rows) · **Canonical:** `data/switchboard/phase1_sft_v2.jsonl`

---

## 1. Verdict

> ### TRAINABLE AFTER FIX
> The data trains end-to-end and the loss falls on both a CPU micro-run and a real GPU Axolotl run.
> Two format defects were found; both are fixed in **`phase1_sft_v2.jsonl`**, which is now the canonical
> training file. Neither defect was a hard blocker for Axolotl specifically, but one is a hard blocker for
> every other consumer (vLLM, eval replay, bare `apply_chat_template`).

Three independent checks agreed on the diagnosis. The template audit and the CPU micro-train each produced a
fixed copy independently; the two files are **byte-identical (md5 `93d6ed90…`)**, which is strong corroboration.

### The two defects and the single canonical fix

| # | Defect | Found by | Blast radius | Fix in v2 |
|---|---|---|---|---|
| 1 | `tool_calls[].function.arguments` is a **JSON string** (OpenAI wire format). `q38.jinja` does `tool_call.arguments\|items`, which requires a Mapping → `TypeError: Can only get item pairs from a mapping.` | Template audit + CPU micro-train | **202/202 rows fail** any bare `apply_chat_template`. Axolotl alone survives — it `json.loads` string arguments itself (`ax/src/axolotl/prompt_strategies/chat_template.py:1170-1183`). That rescue is version-specific and must not be relied on. | Parse to `dict` — **680 argument objects converted**, 0 unparseable, semantically identical |
| 2 | Every row's **last message has role `tool`** — a dangling `{"status": "ended"}` observation answering the final `end_call` that no assistant turn ever consumes | GPU Axolotl smoke | Axolotl logs `"Last turn is not trainable … This is likely a dataset design issue"` on **202/202 rows**. The tokens are permanently masked dead weight (~15 tok/row) and the run risks an incorrectly unmasked final EOT/EOS | Drop trailing `tool` messages — **202 dropped**, last turn is now the final assistant turn (trainable) |

**Checked and clean — no fix needed:** no `null`/empty content (all 4,465 messages are `str`); no missing
`tool_call.id`; no missing `tool_call_id` on any tool message; every tool message answers a preceding call id;
`system` first on all rows; `tools` schema and `meta` untouched.

**Preserved verbatim in v2:** all message text, the system prompt, tool schemas, `meta`, tool-call ids and ordering.
2,033 assistant turns in, 2,033 out. `phase1_sft.jsonl` was **not modified** (md5 `8d6b78d6…` unchanged).

**Trace content untouched.** The fix only changes SFT rendering, so `phase1_final.jsonl` was not regenerated.
Re-ran the validator as a sanity check anyway: `validate_trace.py phase1_final.jsonl scenarios_p1.jsonl`
→ **`validated 202 traces: 202 accepted, 0 rejected`**. `end_call` is still present in every last assistant
message (v2 keeps that turn; only the observation *after* it is dropped).

---

## 2. The exact render the model sees

Rendered with the real `Qwen/Qwen3.8-27B` tokenizer (its chat template is byte-identical to local `q38.jinja`),
`add_generation_prompt=False, enable_thinking=False`. Full text: `render_example_v2.txt`.

```text
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "verify_identity", "description": "Verify the caller. …",
 "parameters": {"type": "object", "properties": {"document_number": {…}, "date_of_birth": {…}},
 "required": ["document_number", "date_of_birth"]}}}
{"type": "function", "function": {"name": "search_customer", …}}
…  ← one JSON object per tool, newline-separated  (~1,180 tokens = 57% of the sample)
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
…  ← template's own XML instructions
[authored system prompt]<|im_end|>
<|im_start|>user
Buenas tardes, cambio de compañía de móvil y quiero poner el número nuevo en la cuenta…<|im_end|>
<|im_start|>assistant
<think>

</think>

Buenas tardes, ahora mismo lo vemos. ¿Me dice su DNI y su fecha de nacimiento, por favor?<|im_end|>
<|im_start|>user
Sí, apunte: siete dos, nueve dos, cuatro ocho, seis cinco, efe de Francia…<|im_end|>
<|im_start|>assistant
<think>

</think>

Un momento, lo compruebo.

<tool_call>
<function=verify_identity>
<parameter=document_number>
72924865F
</parameter>
<parameter=date_of_birth>
12/10/1958
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"status": "ok", "customer_id": "cus_22337"}
</tool_response><|im_end|>
…
<|im_start|>assistant
<think>

</think>

Gracias a usted, Carmen. Que pase buena tarde.

<tool_call>
<function=end_call>
</function>
</tool_call><|im_end|>          ← v2 ends HERE, on a trainable assistant turn
```

Four things to internalise about this format:

1. **Tool calls are XML, not JSON** — `<tool_call><function=NAME><parameter=K>`. At serving time vLLM needs
   `--tool-call-parser qwen3_xml` (alias `qwen3_coder`). The `hermes` parser expects JSON inside `<tool_call>`
   and would fail on **every** call.
2. **Tool results are rendered under the `user` role**, wrapped in `<tool_response>` — not a distinct role.
3. **Every assistant turn carries an empty think block**, the exact string `<think>\n\n</think>`, identical
   across all 2,033 turns. This comes from `preserve_thinking` (undefined → true), *not* from `enable_thinking`.
   Do **not** set `preserve_thinking: false` — it changes the render.
4. **`end_call` renders with no `<parameter=>` block** (`<function=end_call>\n</function>`) because its arguments
   are empty — 202 of 680 calls. Worth a parser smoke test before the eval.

### Train / inference agreement
`add_generation_prompt=True, enable_thinking=False` → `<|im_start|>assistant\n<think>\n\n</think>\n\n`,
**byte-identical** to the training header. No drift.

> **Serving trap.** With `enable_thinking` *unset* (vLLM's default) the template instead prepends
> `"Reasoning effort is set to xhigh…"` to the system block (+36 tokens) and emits an **open** `<think>\n` —
> neither of which appears in training. Pass `chat_template_kwargs={"enable_thinking": false}` on **every**
> request, for both serving and eval.

---

## 3. Token statistics

Real `Qwen/Qwen3.8-27B` tokenizer, `enable_thinking=False`, tools attached.

| | rows | render failures | min | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| **v1 raw** | 202 | **202 (100%)** — `TypeError` | — | — | — | — | — | — |
| **v2, with tools** | 202 | **0** | 1,746 | 2,080 | 2,066 | **2,392** | 2,478 | **2,563** |
| **v2, without tools** | 202 | 0 | 632 | 900 | 879 | 1,180 | 1,266 | 1,352 |

- **Tool-schema block costs ~1,180 tokens/row — 57% of every sample.** The single largest lever on step time.
- Rows over 4,096 tokens: **0**. Rows over 2,560: **1**.
- Nothing truncates at any sane `sequence_len`. The configured `16384` is **6.4× larger than the longest row**.

---

## 4. Mask statistics

Produced by replicating Axolotl 0.19.0.dev0's *real* algorithm — the sentinel char-diff locator
(`_find_turn_from_text` + `_build_dummy_turn` + the `abs(eos_idx - turn_end) <= 3` EOS rule) — not a naive
prefix diff. The fast-tokenizer locator path is confirmed active (**0 locator mismatches**, so no fallback).
Config: `roles_to_train: [assistant]`, `train_on_eos: turn`, `enable_thinking: false`.

| Metric | v1 raw | **v2** |
|---|---|---|
| Total tokens | 423,280 | **420,250** (−3,030 dead tokens) |
| Trainable tokens | 69,386 | **69,386** (unchanged — no signal lost) |
| **Trainable %** | 16.39% | **16.51%** |
| Per-row trainable % (min / mean / p50 / p95 / max) | — | 10.99 / **16.39** / 16.35 / 19.73 / 22.21 |

**Correctness checks on v2 — all zero:**

| Check | Count |
|---|---|
| Assistant turns with an empty/lost span | **0** / 2,033 |
| Turns where `<\|im_end\|>` was *not* trained | **0** |
| Tool-call XML not fully inside the trained span | **0** |
| Foreign tokens (`<\|im_start\|>`, `<tool_response>`) inside a span | **0** |
| Assistant header (`<think>`) leaking into a span | **0** |
| Locator mismatches (would force the slow fallback) | **0** |

The assistant header is correctly *excluded* and `<|im_end|>` correctly *included*. Because
`_build_dummy_turn` drops `tool_calls`, the diff span widens to cover content **and** the tool-call block —
including the **235 silent turns** (empty content, tool calls only), which were checked specifically.

Sample trained spans (exactly what the loss is computed on):

```text
with tool call: 'Un momento, lo compruebo.\n\n<tool_call>\n<function=verify_identity>\n
                 <parameter=document_number>\n72924865F\n</parameter>\n
                 <parameter=date_of_birth>\n12/10/1958\n</parameter>\n</function>\n</tool_call>'
plain turn:     'Buenas tardes, ahora mismo lo vemos. ¿Me dice su DNI y su fecha de nacimiento, por favor?'
```

---

## 5. Loss curves

### CPU micro-train (`Qwen/Qwen3-0.6B` fallback, LoRA r=8, fp32, 8 rows)

`Qwen3.8-0.8B` and `Qwen3.8-2B` **do not exist** — both 404 against the authenticated HF API. The published
Qwen3.8 family is `27B`, `27B-FP8`, `Flash-Next(-FP8)`, `2.4T-A95B(-FP8)`; the smallest is 27B, not CPU-trainable.
Qwen3-0.6B is the same Qwen/ChatML family with `tools` + `enable_thinking` support.

| step | loss | grad_norm | wall |
|---|---|---|---|
| 1 | 2.782 | 3.78 | 65.1 s |
| 2 | 2.388 | 3.36 | 62.3 s |
| 3 | 2.504 | 4.12 | 64.0 s |
| 4 | 2.207 | 3.45 | 65.3 s |

All finite, no NaN. Because each step sees different examples, a **fixed probe** (mean loss over the same 8
examples) is the real signal: **2.708 → 1.935 (−0.773)**. Total 256.7 s.

Greedy decode from a training prompt: `Buenas tardes, el documento y la fecha coinciden. ¿Quieres confirmar el
nuevo número?<|im_end|>` — in-domain, correct language and register, correct stop token.

### GPU Axolotl smoke (`Qwen/Qwen3-1.7B`, RTX 4090, real Axolotl 0.19.0.dev0, 194 train / 8 held-out)

Train loss over 30 steps (bs 1 × grad-accum 4):

```
3.29 3.21 3.29 2.77 2.11 2.18 1.96 1.72 1.58 1.42   (steps 1-10)
1.83 1.52 1.49 1.61 1.31 1.29 1.22 1.42 1.42 1.25   (steps 11-20)
1.39 1.32 1.11 1.33 1.14 1.17 1.31 1.21 1.17 1.26   (steps 21-30)
```

**Held-out eval loss: 3.771 (step 0) → 1.575 (10) → 1.318 (20) → 1.281 (30).** No divergence, no NaN.
Generalisation is real — the eval curve tracks the train curve on rows the model never saw.

Preprocessing: 0 rows dropped, 0 all-masked rows, 16.98% trainable (matches the 16.5% simulation).
Throughput 2.201 samples/s → ~4,275 tok/s total, ~726 tok/s trainable. **Peak memory 3.42 GiB.**

---

## 6. Cost

| Item | |
|---|---|
| GPU smoke total | **≈ $0.30** (budget $1.00) |
| Breakdown | 26 min @ $0.393/h = $0.17 · 12 min @ $0.39/h = $0.08 · 8 min @ $0.381/h = $0.05 |
| Wall clock | 47 min (budget 60) |
| Rentals | 3 — two were duds |

Two boxes failed for driver reasons, not data reasons: **49959770** (driver 570.211.01, broken CUDA init → bogus
`--tf32 requires Ampere`) and **49961534** (driver 560.35.03/cuda 12.6 → `Error 803: unsupported display driver /
cuda driver combination`). **49962416** (driver 590.48.01/cuda 13.1) completed. All three destroyed.
The eval box **49957392 was never touched**.

> Lesson for the 27B run: filter Vast offers on `cuda_max_good >= 13` / recent drivers. Two of three 4090 boxes
> were unusable, which is a ~2× multiplier on expected rental attempts.

---

## 7. Recommended Axolotl settings for the 27B run

Deltas against `pipeline/gpu/axolotl_lora_q38_27b.yaml`:

```yaml
# --- data ---
datasets:
  - path: /workspace/data/phase1_sft_v2.jsonl   # was train_turns.jsonl (the old 10.8k-token data_v1)
    type: chat_template
    field_messages: messages
    field_tools: tools                # ← tools live under "tools"; schemas are OpenAI-style
                                      #    {"type":"function","function":{name,description,parameters}}
    roles_to_train: ["assistant"]     # every assistant turn is a verified-good target
    train_on_eos: turn                # confirmed: <|im_end|> trained on all 2,033 turns
    message_property_mappings:
      role: role
      content: content
      tool_calls: tool_calls
      tool_call_id: tool_call_id

chat_template: tokenizer_default       # unchanged — hub template == local q38.jinja, byte-identical
chat_template_kwargs:
  enable_thinking: false               # unchanged. Do NOT add preserve_thinking: false.

# --- sequencing ---
sequence_len: 4096                     # was 16384. p95 = 2,392, max = 2,563 → 4096 is 1.6x the longest row.
                                       # 16384 wastes allocator headroom on a 27B for zero benefit.
sample_packing: false                  # keep OFF: only 202 rows, and packing fragments the
                                       # shared ~1,180-token tool prefix across boundaries
pad_to_sequence_len: false
```

Keep as-is: `bf16`, `gradient_checkpointing: true` (+ `use_reentrant: false`), `attn_implementation:
flash_attention_2`, the CutCrossEntropy plugin, `adapter: lora`, `lora_r: 64 / alpha: 128 / dropout: 0.05`.

**Still to be settled by the 27B smoke, not by this report:**

- **`linear_attn.*` LoRA targets** (`in_proj_qkv`, `in_proj_z`, `out_proj`). The GPU smoke ran on dense
  Qwen3-1.7B, so these were dropped and remain **unvalidated**. Verify the module names resolve on the real
  hybrid 27B before a long run — a silent no-match means you train attention/MLP only.
- **`tf32: true`** — left on for the 27B, but it is exactly what broke smoke box #1. If `TrainingArguments`
  raises an Ampere error, the box's CUDA init is broken; move on rather than debugging tf32.

**Two sizing observations worth acting on:**

1. **Step count is very low.** 202 rows × 2 epochs ÷ effective batch 8 (`micro_batch_size: 1` ×
   `gradient_accumulation_steps: 8`) = **~50 optimizer steps** for a rank-64 LoRA. Consider
   `gradient_accumulation_steps: 4` (~100 steps) or 3 epochs. The smoke reached eval 1.281 in 30 steps at
   effective batch 4, so signal arrives fast, but 50 steps is thin for r=64.
2. **Total trainable signal is ~69k tokens/epoch.** Small. Watch the held-out curve for overfitting after
   epoch 1 — `evals_per_epoch: 2` is already set, so keep the eval split and actually read it.

Also carve a held-out split: the smoke used **194 train / 8 eval**, and `test_datasets` in the config expects a
separate dev file. Split `phase1_sft_v2.jsonl` before uploading, stratified over `meta.language` × `meta.style`.

### Serving / eval (vLLM)

```bash
--tool-call-parser qwen3_xml     # NOT hermes — the template emits XML, hermes expects JSON
--enable-auto-tool-choice
# every request:
chat_template_kwargs={"enable_thinking": false}
```

Smoke-test the parser against an **empty-argument** call (`<function=end_call>\n</function>`, 202/680 of the
calls in this set) before trusting eval numbers.

---

## 8. Files

| Path (under `data/switchboard/`) | |
|---|---|
| `phase1_sft_v2.jsonl` | **canonical training data** — 202 rows, both fixes applied |
| `phase1_sft.jsonl` | original, **unmodified** (md5 `8d6b78d6…`) |
| `trainability/make_v2.py` | the fix, with assertions and post-conditions |
| `trainability/render_example_v2.txt` | full render of row 0 |
| `trainability/token_stats_v2.json` | v1-vs-v2 render + token stats |
| `trainability/axolotl_mask_sim_v2.{py,json}` | mask simulation on v2 |
| `trainability/validate_trace_recheck.json` | 202/202 traces still accepted |
| `trainability/template_audit.json`, `cpu_microtrain.json`, `gpu_smoke.json` | the three source checks |

Superseded by `phase1_sft_v2.jsonl` (fix 1 only, no trailing-tool trim; byte-identical to each other):
`trainability/phase1_sft_render_fixed.jsonl`, `trainability/phase1_sft_dictargs.jsonl`.

On HF `Enriqueag26/agentic-distill-work`: `switchboard/phase1/phase1_sft_v2.jsonl` and
`switchboard/phase1/TRAINABILITY_REPORT.md`.
