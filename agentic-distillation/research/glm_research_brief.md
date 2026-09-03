# What GLM's research (4.5 → 5 → 5.2 → 5.3) teaches us, and exactly what we can reproduce to distill Qwen3.8-27B for tau3-Banking and GDPval-AA v2 with ~$300 of API credit and tens of H100-hours

*Technical brief, 2026-09-03. Everything tagged **[UNVERIFIED]** comes from secondary sources, search snippets, or memory and must be re-checked before it drives a decision. Everything else is traced to a primary Z.ai/THUDM document linked inline.*

---

## 1. GLM's post-training arc in one page

| Version (date) | Architecture / base | What changed in post-training | Numbers worth remembering |
|---|---|---|---|
| **GLM-4.5** (Jul 2025, [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)) | 355B-A32B MoE (+ Air 106B-A12B); hybrid thinking/non-thinking | Expert models (reasoning / agent / general) trained separately then unified by self-distillation. Agentic SFT built from **MCP servers + real tool APIs → LLM-generated queries → LLM user-simulator → multi-judge success filter, keep only successes** (Sec 3.1). RL via GRPO in the open-source **slime** framework. Thinking data "meticulously balanced" between reasoning and no-reasoning samples. | TAU-bench retail 79.7 / airline 60.4; BFCL v3 77.8. In-house CC-Bench = 52 coding tasks; manual eval set 660 prompts. SWE-bench Verified 64.2, Terminal-Bench 37.5, BrowseComp 26.4 **[from memory, verify]**. |
| **GLM-5** (Feb 2026, [arXiv 2602.15763](https://arxiv.org/abs/2602.15763)) | ~744B-A40B, DSA sparse attention, ~28.5T pretrain tokens **[from memory, verify]** | Infra: server-based **multi-task rollout orchestrator** (each task = microservice with own rollout+reward), **TITO gateway** (train on the exact token IDs the inference engine emitted), asymmetric double-sided IS clipping (ε_l=0.2, ε_h=0.28), staleness filter, >1k concurrent rollouts, **10k+ SWE environments** across thousands of repos in 9 languages. SFT: **erroneous segments retained but loss-masked**; interleaved + preserved thinking. **On-policy cross-stage distillation**: per-token advantage = sg[log π_teacher(y_t) − log π_student(y_t)] on student-sampled tokens, teacher = previous-stage checkpoint, group size 1, batch 1024. CC-Bench-V2 scored with unit tests + Agent-as-a-Judge. | tau2-bench 89.7; Terminal-Bench 2 56.2; SWE-bench Verified 77.8. No decontamination statement anywhere in the report. |
| **GLM-5.1** (~Apr 2026) | (not separately documented) | Only visible as the baseline in the 5.2 blog. | SWE-bench Pro 58.4; Terminal-Bench 2.1 63.5; FrontierSWE 30.5; PostTrainBench 20.1. |
| **GLM-5.2** (Jun 2026, [HF blog](https://huggingface.co/blog/zai-org/glm-52-blog)) | **IndexShare**: one sparse-attention indexer shared per 4 layers (2.9× fewer per-token FLOPs at 1M ctx); 1M context; MIT weights | Long-horizon RL (what docs.z.ai later calls "SAO with compaction"): (a) binary verifiable rewards; (b) **switch from group-relative (GRPO) to critic-based PPO on individual rollouts** because long traces are few and expensive; (c) **compaction inside training** – each compacted sub-trace is a trainable trajectory, **token-level loss** to remove length imbalance; (d) online two-stage anti-reward-hacking guard (rule filter → LLM intent judge; rollout continues after a caught hack, penalised not truncated); (e) slime gains four rollout modes: white-box, black-box, compact trajectory, sub-agent. | SWE-bench Pro 62.1; TB 2.1 81.0; FrontierSWE 74.4; PostTrainBench 34.3; SWE-Marathon 13.0. From the 5.3 card: **GDPval-AA v2 1508**, Toolathlon 59.9, AutomationBench 26.2, HLE-w/-Tools 54.7. No data volumes disclosed. |
| **GLM-5.3** (Aug 14 2026, [docs.z.ai](https://docs.z.ai/guides/llm/glm-5.3), [HF card](https://huggingface.co/zai-org/GLM-5.3)) | **Same base model as 5.2** – "every gain comes from post-training" | **End-to-end environment synthesis**: research agents turn real-work task patterns into runnable long-horizon envs with multi-step dependencies and hidden state; a **judge agent must solve each task** before admission; **verifiers written without access to the reference solution**, then hardened by auditing solver trajectories for reward shortcuts; reward is binary and "reliable enough to train on directly". Z.ai concedes meaningful human-in-the-loop work and that rewards were machine-generated only for a subset. | **GDPval-AA v2 1769** (vs 1508; Fable 5 1743, Qwen3.8-Max 1739, GPT-5.6 Sol 1730); Toolathlon 73.0; AutomationBench 48.2; TB 3.0 28.3 (from 4.6); TB 2.1 88.2; Z.ai Code Bench 34.5% (from 23.4%); SWE-Marathon 42.5; CyberGym 84.5; 753B params; custom license. **No tau2/tau3/BFCL numbers published.** |
| **GLM-5.3-Flash** (Aug 26 2026, [HF card](https://huggingface.co/zai-org/GLM-5.3-Flash), [docs](https://docs.z.ai/guides/vlm/glm-5.3-flash)) | 320B-A18B MoE, hybrid sparse+linear attention + mHC, natively multimodal, 1M ctx, **plain MIT** | Same 5.3 pipeline family; explicit **render-and-inspect** loop for office deliverables (PPTX/PDF/DOCX/XLSX; detects overflow, misalignment, overlap by rendering and looking). reasoning_effort ∈ {low, high, max}. | TB 2.1 84.3; DeepSWE 63.4; AutomationBench 48.8; Code Bench 29.0 (Opus 4.8 29.5); AA Intelligence Index 57. **GDPval-AA v2 ~1770–1773 [UNVERIFIED, secondary]**; **tau3-Banking 47.2, rank 6/164 [UNVERIFIED, blocked aggregator snippet]**. API $0.15 in / $0.50 out per 1M tokens, 83% cache discount. |

**The one-line lesson:** the biggest jump on our target metric (GDPval-AA +261 Elo, 5.2→5.3) came with **zero architecture change** – it was environment synthesis + solvability filtering + shortcut-hardened binary verifiers + long-horizon RL. The data pipeline is the product; RL is the amplifier. We can afford the pipeline; we cannot afford the amplifier at GLM's scale, but the pipeline alone (used as an expert-iteration filter) is where GLM-4.5 already got 79.7 on TAU retail.

---

## 2. Environment & task synthesis: GLM recipes mapped onto our two targets

### 2.1 The GLM recipe, decomposed
1. **Tool surface first** (4.5 §3.1): tools as MCP-style JSON schemas; an LLM reads the schemas and generates queries.
2. **Research-agent env generation** (5.3): from *real task patterns* to runnable environments with **hidden state** and **3–6 dependent steps**, framed as "substantial work end to end", not decomposed exercises.
3. **Judge-agent solvability gate** (5.3): a strong model must actually solve the task once; unsolvable tasks are dropped.
4. **Reference-free verifier** (5.3): the verifier author never sees the gold trajectory; then **run cheap solvers and patch any verifier that passes a wrong trajectory**.
5. **Binary reward** – no rubric scalar for training gates.
6. **Two-stage guard** (5.2): rule filter (high recall) → LLM judge on flagged actions only (high precision).
7. **User simulator + multi-judge committee** (4.5) for conversational tasks; keep only successes.

### 2.2 Target A — Rho-Bank (tau3-Banking analog) over the 698-doc KB

What we already have maps cleanly: our **replay-verified gold actions** are GLM's step 3 (solvability gate) done deterministically – cheaper and stricter than a judge agent. What to add:

**Research-agent task synthesis prompt (copy this shape):**
```
You are designing customer-service tasks for a retail bank. Inputs: (a) 3–5 policy
documents from the KB (pasted), (b) the tool schemas (pasted), (c) a seed customer
record. Produce ONE task as JSON with fields:
  customer_goal (natural language, as the customer would say it, incomplete),
  hidden_facts (things the customer knows but only reveals if asked: e.g. second
    account, travel dates, joint owner),
  policy_traps (≥1 clause in the pasted docs that makes the naive action wrong),
  required_action_sequence (3–6 tool calls with args; later calls depend on values
    returned by earlier ones),
  forbidden_actions (calls that violate the pasted policy),
  success_state_predicate (a statement about the final DB state only).
Do not reuse a goal you've produced before; vary the policy clause that governs the case.
```
Run it once per KB-doc cluster; a few hundred tasks costs <$5 on Flash. Then **replay `required_action_sequence` against the simulator** – drop any task whose replay fails or whose predicate is not satisfied afterwards (our existing gate = GLM's judge agent).

**Reference-free verifier design (GLM 5.3 step 4):** the verifier author gets `customer_goal + hidden_facts + policy docs + tool schemas` but **not** `required_action_sequence`. It writes:
- a **state-diff assertion** (`final_db == expected transform of seed_db`),
- a **forbidden-call check** over the transcript (any call in `forbidden_actions` → fail),
- a **disclosure check**: a hidden fact was elicited before the action that depends on it (catches guessing).
Then **shortcut audit**: run 2–3 cheap solver rollouts (student at temperature 1, or Flash at `reasoning_effort=low`); inspect every *pass*. Typical shortcuts we should expect: verifier only checks the final balance so a wrong-account transfer plus reversal passes; policy check only looks for exact strings. Patch and re-run. This is cheap (rollouts on our own GPUs, no API).

**User simulator:** Flash at `reasoning_effort=low`, system prompt = customer persona + hidden_facts + "reveal only when asked directly; be terse; never mention tool names". tau3-Banking is a user-sim benchmark, so the simulator's stinginess is a knob to match evaluation difficulty **[tau3 protocol details UNVERIFIED – confirm on Artificial Analysis]**.

### 2.3 Target B — GDPval-analog occupational deliverables

GDPval-AA v2 is a **pairwise Elo** over deliverables (docx/xlsx/pptx/pdf) across occupations **[v2 protocol details UNVERIFIED]**. Do **not** use the 220 public OpenAI GDPval gold prompts as training prompts (decontamination, §3.4); use them only as the *pattern source* for GLM-style "research agent" synthesis.

**Task synthesis prompt:**
```
Occupation: {occupation}. Here are 3 public GDPval task descriptions for this
occupation (do NOT copy them). Write a NEW task a senior {occupation} would receive by
email: a 4–8 sentence brief, 1–3 attached input files (generate their contents: CSV,
short memo, table), the requested deliverable type (docx|xlsx|pptx|pdf), and 5 rubric
items a demanding manager would check (specific, verifiable by opening the file).
```
**Sandbox for the teacher** (this is where GLM-5.3-Flash's documented behaviour pays off): tools = `write_file`, `run_python` (python-docx, openpyxl, python-pptx, reportlab), `render` (soffice → PDF → pdf2image PNG), `view_image`, `read_file`. Flash is multimodal, so **it will look at the rendered page and fix overflow/misalignment** – record those tool calls; that loop is the behaviour we want the student to inherit (the student is text-only, so keep `render` returning structured layout diagnostics – bounding boxes, overflow flags, slide count – alongside the PNG, so the behaviour survives without vision).

**Verifier (reference-free, two layers):**
- **Deterministic layer**: file opens; correct type; page/slide count in range; every rubric-named entity (table, chart, section header) exists (`python-docx`/`openpyxl` introspection); no text overflow flags from the layout diagnostics; numbers in the deliverable reconcile with numbers in the input files (cheap regex/arith check).
- **Pairwise judge layer** (mirrors Elo): judge model sees *two* rendered deliverables (teacher vs. student baseline, or teacher vs. teacher) for the same brief plus rubric, outputs a winner. **Keep a teacher trajectory only if it beats the student's own deliverable** – this ensures every SFT sample is strictly above the student's current level, which is the only thing that matters for an Elo metric. Position-swap both orderings to kill position bias.

Both verifiers are written before any trajectory is inspected, then hardened by auditing passes from cheap solvers (GLM 5.3 step 4).

---

## 3. Data hygiene: what to keep, what to mask, what format

### 3.1 Judge-filtered successes (4.5 §3.1, 5.3)
Keep a trajectory only if **all** hold: binary verifier passes; no forbidden action; two-stage guard clean (rule filter over the transcript → LLM judge on flagged steps *only*, so judge cost is ~5–10% of trajectories); for GDPval, pairwise win vs. student baseline. Expect 30–60% yield on Banking, lower on office tasks.

### 3.2 Loss masking of erroneous segments (GLM-5 §3.1 – verbatim: "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors")
This is the single highest-leverage cheap trick. Implementation:
- Standard masks: `labels=-100` on system, user, user-simulator, and `<tool_response>` tokens.
- **Error-span mask**: an assistant span is "erroneous" if (a) the following tool response is an error / validation failure, (b) the rule filter or judge flagged it, or (c) a policy-forbidden call. Mask the *whole assistant message* containing the bad call (think + call), **train normally on the recovery message**. Do not delete the segment – the recovery only makes sense in context.
- GLM-5's slide pipeline masks "defective pages" found by rendering (§4.2.5). Analog: mask the assistant turn that produced a page the layout diagnostics flagged, keep the fix turn.
- **Token-level averaging** (5.2): average loss over all unmasked tokens in the global batch, not per sample – otherwise 60k-token GDPval trajectories are down-weighted 20× relative to Banking ones. In TRL SFTTrainer this is the default per-token mean across devices; just do not switch to per-sequence loss.
- **Compaction segments as separate samples** (5.2): if a teacher trajectory exceeds the student's training context, split at a compaction point (summary of prior work inserted as a user message) and train each segment as its own sample.

### 3.3 Thinking format and the exact Qwen3.8 chat-template implications
Collect from Z.ai with `thinking` on, `clear_thinking=false`, `reasoning_effort=high` (Banking) or `max` (GDPval), temperature 1, top_p 0.95, `tool_stream=true`, and **echo `reasoning_content` back verbatim on every assistant message with tool_calls** ([Thinking Mode docs](https://docs.z.ai/guides/capabilities/thinking-mode); harnesses that skip this see infinite loops – goose #7363). Store OpenAI-style messages with `reasoning_content`.

GLM template ([chat_template.jinja](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/chat_template.jinja)) vs. Qwen:
| GLM-5.3-Flash | Qwen3.x student |
|---|---|
| `<|assistant|><think>…</think><tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` | `<|im_start|>assistant\n<think>\n…\n</think>\n\n<tool_call>\n{"name":…,"arguments":{…}}\n</tool_call><|im_end|>` |
| `<|observation|><tool_response>…` | `<|im_start|>user\n<tool_response>\n…\n</tool_response><|im_end|>` |
| keep think iff `not clear_thinking or loop.index0 > last_user_index` | keep think iff index > `last_query_index` (last real user msg, not a tool response) **[verify Qwen3.8's template still uses this rule]** |
| `reasoning_effort` tag Low/High/Max | none – **strip it**, never leak GLM control tokens |

Consequences:
1. **Never hand-render GLM markup.** Convert to OpenAI messages and call `tokenizer.apply_chat_template(messages, tools=…)` on the Qwen3.8 tokenizer – it emits the JSON `<tool_call>` form and routes tool results into the user role (automatically masked).
2. **One SFT sample per user-turn boundary.** Qwen's template (like GLM's with `clear_thinking=true`) drops `<think>` for assistant messages before the last user message at inference. Train identically: for a 4-user-turn Banking episode, build 4 samples; in each, only the current think→tool_call→observation→think chain carries reasoning (interleaved thinking preserved *within* the chain, stale thinking dropped). Training with preserved reasoning across user turns and inferring without it is a train/test mismatch.
3. **Compute masks from the rendered token stream**, not from string offsets: render prefixes incrementally and diff token counts, or use the template's `{% generation %}` markers with `return_assistant_tokens_mask=True` if Qwen3.8's template has them **[verify]**. Round-trip-test: assert that unmasked tokens decode to exactly the assistant spans.
4. **Empty-think convention**: if the student will run in thinking mode at eval, every assistant message in training must carry a non-empty `<think>` block; drop any teacher message where `reasoning_content` is missing rather than inserting an empty block.
5. **Cross-family tokenizers** (GLM vs Qwen) rule out any token-level distillation objective from GLM; GLM contributes text-level trajectories only (see §4).

### 3.4 Decontamination
Neither GLM report discloses any decontamination method or benchmark exclusion list, and Z.ai markets GDPval-AA v2 directly – **assume the teacher is optimised on the GDPval distribution.** For us: 13-gram overlap filter of every synthesized prompt, hidden fact and rubric against the public tau-bench/tau2 task texts and the 220 GDPval gold tasks; log removals. Minutes of CPU; document it in the eval writeup.

---

## 4. RL / OPD: what slime, "SAO", and cross-stage distillation do — and what runs on 1–8 H100

### 4.1 What they are
- **slime** ([THUDM/slime](https://github.com/THUDM/slime), Apache-2.0): Megatron-LM training + SGLang rollout servers + Data Buffer. Agentic RL = swap in `--custom-generate-function-path` (multi-turn loop over the router `/generate`, `loss_mask=0` on tool/user tokens) + `--custom-rm-path`. Off-policy correction via TIS (`--use-tis`; GLM-5 used ε_l=0.2/ε_h=0.28 + staleness cap). `fully_async` mode keeps a pool of in-flight episodes so slow user-sim episodes don't stall the step ([example](https://github.com/THUDM/slime/blob/main/examples/fully_async/README.md)). **No LoRA path** in the docs; full-parameter only. Documented single-node (8×H100) ceiling: ≤9B dense or 30B-A3B MoE. **Qwen3.8 is not in the supported list** (Qwen3.5/3.6/3-Next are).
- **"SAO"**: the acronym does not appear on any primary Z.ai page; docs.z.ai says "SAO with compaction"; third parties expand it as "Scalable Agentic Optimization" **[UNVERIFIED]**. Substantively it is the 5.2 recipe: critic-based PPO on individual long rollouts, compaction sub-traces as trajectories, token-level loss, online anti-hacking guard.
- **On-policy cross-stage distillation** (GLM-5): reverse-KL-on-sampled-tokens, same family as [Thinking Machines' OPD](https://thinkingmachines.ai/blog/on-policy-distillation/) (Qwen3-8B reached 70% AIME'24 with ~1,800 GPU-h vs 17,920 for RL; a *short* OPD pass restored IF-eval 45%→83% after domain mid-training). Requires a teacher that can **score a caller-supplied completion** with the **same tokenizer**. [Revisiting OPD (2603.25562)](https://arxiv.org/abs/2603.25562) adds top-K=32 local support matching, top-p rollouts, special-token masking (41.7 vs 36.4 avg math on 7B).

### 4.2 Honest verdict
| GLM component | 1–4 H100 | 8 H100 | Cheapest faithful substitute |
|---|---|---|---|
| Env synthesis + judge gate + reference-free verifiers + shortcut audit | **Yes** (API + CPU) | Yes | None needed – copy directly (§2). |
| Judge-filtered SFT with error masking, token-level loss, compaction splits | **Yes** (QLoRA/LoRA 27B on 1–2 H100, ~10–20 H100-h for ~5k samples × 2 epochs) | Yes | This *is* GLM-4.5's agentic recipe minus RL. TRL `SFTTrainer` + PEFT. |
| Critic-based PPO on long rollouts (5.2 "SAO") | No | Borderline-no (27B dense full-param ≈ 430 GB opt+weights vs 640 GB; TP=8 + CPU-Adam offload, ~10 s/step) | **RFT / expert iteration**: sample N=8 student rollouts per task, keep verifier-passes (+ error-masked recoveries), SFT. Same binary reward, no critic, LoRA on 2 H100. |
| GRPO agentic RL in slime | No | Only if Qwen3.8→Megatron conversion works (undocumented; smoke-test first) | **TRL `GRPOTrainer` with LoRA + colocated vLLM**, or **verl** with LoRA, group 4–8, ≤300 tasks, on 4–8 H100 for a few hours – small but genuine on-policy signal for Banking where rewards are binary and episodes short. |
| Online two-stage anti-hacking guard | Yes (offline) | Yes | Run the rule→judge guard *offline* on RFT/GRPO rollouts before they enter the update. |
| Cross-stage OPD to prevent forgetting | **Yes** (LoRA student on 2 H100 + 1–2 H100 serving a same-tokenizer teacher in prefill-only scoring) | Yes | **TRL `GKDTrainer`** or **verl OPD** ([docs](https://verl.readthedocs.io/en/latest/algo/opd.html)) with teacher = *original Qwen3.8-27B* (self-OPD to restore general/IF capability after SFT) or a larger open Qwen3.8 **[availability UNVERIFIED]**. Group 1–4, K=32 support matching, lr ~1e-5 LoRA. ~5–10 H100-h. |
| DPO on success/failure pairs | Yes | Yes | TRL `DPOTrainer`, pairs = same task, verifier pass vs fail, both student-generated (on-policy pairs). Cheaper than GRPO; watch for length bias – use `rpo_alpha` or SFT-anchored variant. |
| GLM→Qwen token-level distillation | **No** – different tokenizer, no logprobs from the Z.ai API | No | Text-level only (SFT on trajectories). |
| Multi-task orchestrator, 1k concurrent rollouts, 10k envs, PD-disaggregation | No | No | Two task types, one async loop, ≤300 tasks. |

**Cheapest faithful ordering:** SFT (judge-filtered teacher + error masking) → RFT on the student's own verifier-passing rollouts (this is where Banking gains will come from, since the teacher is *weaker* than the student there) → short self-OPD pass against the pre-SFT checkpoint to un-forget → optional 2–4 h GRPO/DPO on Banking only if the RFT curve is still rising.

---

## 5. GLM-5.3-Flash as teacher

- **License**: plain **MIT** ([HF card](https://huggingface.co/zai-org/GLM-5.3-Flash)); no MaaS gate, no output/distillation clause. (GLM-5.3 full has a custom license with a $10B-MaaS-revenue security-review gate – irrelevant at our scale – and is also silent on distillation; 753B, API-only for us.)
- **Price**: $0.15 / $0.50 per 1M in/out, 83% cache discount, 47.5 tok/s. A 60k-token office trajectory ≈ $0.016; a Banking episode (8 turns, cumulative cached prefixes) ≈ $0.005–0.01 incl. user-sim.
- **Strengths**: GDPval-AA v2 Elo ~1770–1773 **[UNVERIFIED, secondary; ≈ GLM-5.3's 1769 and above Fable 5's 1743]**; documented render-and-inspect office behaviour; natively multimodal (can look at its own rendered pages); Toolathlon/AutomationBench lineage. **This is the right bulk teacher for Target B.**
- **Weaknesses**: **tau3-Banking 47.2 [UNVERIFIED]**, i.e. *below our student*. Text-only student cannot inherit vision-dependent checks unless `render` also returns structured diagnostics. No tau2/BFCL numbers published for any 5.x model. Z.ai reports rewards were machine-generated for only a subset – expect the teacher to occasionally "declare done" on unverified deliverables.
- **How to use it**:
  1. **Target B (GDPval)**: primary trajectory generator at `reasoning_effort=max`, in our sandbox, kept only on deterministic-pass + pairwise-win-vs-student. Budget ~$120 → 5–7k trajectories before filtering.
  2. **Target A (Banking)**: **not** as demonstrator by default. Use as (a) task synthesizer, (b) user simulator at `low`, (c) shortcut-audit solver. For hard task clusters where the student's best-of-8 pass rate is <20%, buy a few hundred demonstrations from a *stronger* model (GLM-5.3 full via API – pricing **[UNVERIFIED]** – or another frontier model) and keep only replay-verified successes.
  3. **Judge**: pairwise deliverable judge and flagged-step intent judge. Use a *different* model than the generator where affordable (GLM-5.3 full or another frontier model) to avoid self-preference bias; Flash as the judge only for the cheap, high-volume rule-flag screening.
  4. Always pass `reasoning_content` back; harvest with `clear_thinking=false` so every assistant message has reasoning to train on.
  5. Verify the 47.2 / ~1770 figures on [artificialanalysis.ai](https://artificialanalysis.ai) before committing spend; if Flash's Banking score is materially higher than 47.2, revisit (ii).

---

## 6. Prioritized, costed action list

| # | Action | API $ | H100-h | Why (GLM lesson) |
|---|---|---|---|---|
| **1** | **Harden Rho-Bank verifiers GLM-5.3 style**: rewrite verifiers reference-free (state-diff + forbidden-call + disclosure check), run 3 cheap student rollouts per task, patch every false pass. Add 13-gram decon filter vs tau/tau2/GDPval texts. | ~$5 | ~3 (rollouts) | 5.3 attributes its whole gain to trustworthy binary rewards; shortcut-hardening is the cheapest part of that. |
| **2** | **Banking RFT / expert iteration** (student is the teacher): N=8 rollouts per task at T=1 with Flash user-sim, keep verifier passes, **retain-and-mask erroneous turns**, token-level loss, 1 sample per user-turn boundary, LoRA SFT. 2 rounds. | ~$40 (user-sim + flagged-step judge) | ~25 (rollouts 2×~6 h vLLM + 2×~5 h training) | GLM-5 error masking + 4.5 success-only filtering; correct when teacher < student. |
| **3** | **GDPval-analog synthesis + Flash trajectories**: 40 occupations × ~30 tasks; sandbox with docx/xlsx/pptx + render diagnostics; keep on deterministic-pass + pairwise-win-vs-student (position-swapped). | ~$120 teacher + ~$60 judge | ~2 (rendering/CPU mostly) | 5.3-Flash's documented render-and-inspect behaviour is exactly the GDPval-AA skill; teacher ≫ student here. |
| **4** | **Combined SFT** on (2)+(3) + ≤10% zai-org/DeepDive `trajectories_sft` (858, MIT) for generic multi-step tool use; Qwen3.8 template round-trip test on masks before training. | $0 | ~15 (27B LoRA, ~5–7k samples, 2 epochs) | 4.5/5 SFT recipe; DeepDive is the only SFT-ready Z.ai agentic set ([hf.co/zai-org](https://huggingface.co/zai-org)). |
| **5** | **Self-OPD un-forgetting pass**: teacher = pre-SFT Qwen3.8-27B served by vLLM (prompt_logprobs, K=32), student = SFT LoRA, reverse-KL with top-K support matching + special-token masking on general + IF prompts. | $0 | ~8 | GLM-5 cross-stage distillation / Thinking Machines IF-eval 45→83 recovery; guards tau3 conversational quality. |
| **6** | **Eval gate**: tau3-Banking-style held-out Rho-Bank set + pairwise GDPval-analog win-rate vs. base student after each stage; stop any stage that regresses. | ~$20 (judge) | ~4 | 5.3 admitted human-in-the-loop remains; our loop is this gate. |
| **7** | *(Optional, if #2 curve still rising)* **GRPO or DPO on Banking**, TRL LoRA + colocated vLLM, ≤300 tasks, group 4, offline rule→judge guard on rollouts before update. Smoke-test slime Qwen3.8 conversion in parallel, but do not depend on it. | ~$20 | ~10–15 | Smallest faithful stand-in for 5.2's binary-reward RL; slime is the right template but 27B dense full-param does not fit ≤8 H100. |
| **8** | *(Only if Flash's tau3-Banking figure verifies well above 47.2, or for <20%-pass clusters)* buy 200–400 replay-verified demonstrations from a stronger model. | ~$30 | 0 | Targeted teacher use where student self-play stalls. |

**Totals**: ≈ $295 API, ≈ 65–75 H100-hours (fits "tens" with margin if #7 is skipped).

**Things to verify before spending**: Flash tau3-Banking 47.2 and GDPval ~1770 on Artificial Analysis; Qwen3.8 chat template's last-query think-dropping rule and `{% generation %}` support; whether a larger open Qwen3.8 exists for non-self OPD; GLM-5.3 full API pricing; tau3-Banking / GDPval-AA v2 exact protocols; GLM-4.5/5 pretrain numbers cited from memory in §1.