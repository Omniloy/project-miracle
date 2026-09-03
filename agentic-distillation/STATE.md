# Agentic distillation of Qwen3.8-27B — project state

_Last updated: 2026-09-03 19:30 UTC (auto-saved periodically; see `save_state.sh`)._

## Objective
Produce the best open-weights model at its size on the Artificial Analysis **Agentic Index** (= mean of **GDPval-AA v2** and **τ³-Bench Banking**), by distilling stronger teachers into **Qwen3.8-27B**, with a clear measured margin over the base on both benchmarks, reproducibly via the public harnesses. Stretch: make it *smaller* (depth-prune + heal) while better.

## Where things live
| What | Where |
|---|---|
| Code (this dir) | `agentic-distillation/pipeline/**` — task generator, validator, decontamination, verifiers, SFT converters, GPU job scripts |
| Research briefs | `agentic-distillation/research/*.md` (compression brief, agentic plan, GLM brief) |
| Private HF work repo (data, trajectories, adapters, results) | `Enriqueag26/agentic-distill-work` (dataset, private): `bundle/`, `data/` (v0), `data_v1/`, `bundle/data_synth_tasks/` (79 synthetic tasks), `status/`, `adapter_v0/`, `results/` |
| Harness | `sierra-research/tau2-bench` v1.0.1 (+ `harness/tau2_config_openrouter_judge.patch`: routes NL judge / env-interface / review LLMs through OpenRouter) |
| Secrets | **never in the repo**; kept in the session scratchpad `.env` (OPENROUTER_API_KEY, VAST_API_KEY, HF_TOKEN). Rotate after the project. |

## Key numbers so far (all measured by us unless noted)
- **Baseline Qwen3.8-27B on τ³-Banking**, AA-exact config (`bm25_grep`, seed 300, GPT-5.4 Mini medium user-sim, 200 steps, 1 trial): **44/97 = 45.4% pass^1** (AA: 47.4 medium / 48.0 xhigh) → harness reproduces AA within noise. Banking has **0/97 NL-assertion tasks** (DB/ACTION rewards only) so the judge model is irrelevant there.
- AA reference (2026-09-03): Banking — Muse Spark 1.3 52.4, Qwen3.8-Max 51.3, GLM-5.3 50.3, **Qwen3.8-27B 48.0 (#3)**, GLM-5.3-Flash 47.2. GDPval-AA v2 — Fable 5.1 1853, GLM-5.3-Flash 1765, GLM-5.3 1758, Qwen3.8-Flash-Next 1743, Qwen3.8-Max 1721, **Qwen3.8-27B 1543**. ⇒ Banking headroom ~3 pts; GDPval headroom ~200 Elo.
- **Teacher memorization probe**: 0/12 exact tool-suffix hits for Qwen3.8-Max, GLM-5.3, GLM-5.3-Flash without KB access → no evidence teachers memorized the public test set.
- Synthetic tasks (real Rho-Bank KB, fresh entities, replay-validated, decontaminated): **79 live** (batch1: 27 GLM + 12 Max + 5 pilot; batch2 harder: 35 GLM). Generation ≈ $0.03–0.06/task.
- Rollouts (relaxed outcome verifier = DB state minus read-log table and timestamp): GLM-5.3 29/64 on batch1; GLM-5.3 10/24 on Max-authored tasks; Qwen3.8-Max 3/30 on the 15 GLM-unsolved tasks; **Qwen3.8-Flash-Next 0/28** on the same; student 16/18 on GLM-solved tasks, ~42% overall. **13/15 "hard" tasks unsolved by every agent → treat as broken (solvability gate).**
- SFT v0: 42 verified trajectories → 37 train / 5 dev (3 held-out tasks) → 68 windows (≤16K tokens, tool results capped at 8k chars), 805 trained assistant turns. v1 adds student self-passes (auto-built when the self-rollout run exits).

## Decisions (with reasons)
1. **Student = Qwen3.8-27B** (Apache-2.0, 55.6 GB bf16 / 30.9 GB FP8, `model_type qwen3_5`, trainable on one 96 GB GPU). Not GLM: GLM-5.3 is 753B API-only, GLM-5.3-Flash is 328 GB bf16 and *below* the 27B on Banking; GLM-4.7-Flash (31B-A3B) has Intelligence Index 23 vs 52. Not Qwen3.8-Flash-Next: 360 GB bf16 / 135 GB NVFP4; experts are fused 3-D tensors → PEFT/bitsandbytes cannot LoRA/QLoRA them; no trainer supports `qwen4_exp`; 0/28 on our hard set; deployable artifact 135–185 GB contradicts "smaller".
2. **Teachers**: GLM-5.3 primary (newest, strongest on GDPval), Qwen3.8-Max on residual failures, Qwen3.8-Flash-Next as cheap 3rd teacher for GDPval-style tasks only. GLM-5.3-Flash = user-simulator / cheap judge (it is below the student on Banking).
3. **Verifier-first, multi-teacher cascade** (user's strategy): every trajectory kept only if the environment's DB-state comparison passes; union of verified successes from any teacher + the student's own verified passes (self-distillation). Tasks nobody solves are dropped, not trained toward.
4. **Contamination**: never train on the 97 test tasks or paraphrases; synthetic tasks decontaminated by 5-gram Jaccard, identical gold-action signature, and entity overlap (names/emails/ids/phones) — disclose "Track B (real KB, fresh entities)" in the model card.
5. **GPU**: RTX PRO 6000 Blackwell 96 GB at ~$1.3–1.6/hr (kernels verified: `flash-linear-attention` imports; torch 2.12.1+cu130, transformers 5.16.1, peft 0.20.0). H100 only if DeltaNet LoRA fails on Blackwell. **No SSH egress from the controller container** → all GPU jobs are unattended `--onstart` scripts that pull from / push to the HF work repo; monitored via `vastai logs` + HF `status/`.
6. **Training**: bf16 LoRA r=64/α=128 on attention+MLP (+DeltaNet `linear_attn.in_proj_qkv/in_proj_z/out_proj` if the smoke test trains them), Axolotl, seq 16K, `enable_thinking: false` (teacher trajectories carry no reasoning; AA shows Banking insensitive to thinking budget), eval both modes.

## Gotchas discovered (save time later)
- The benchmark DB hash also covers a **frozen clock** (`get_current_time` = `2025-11-14 03:40:00 EST`; verification-record id derives from it) and a **read-log table** of allowlisted lookup calls (allowlist = reads present in the task's gold actions). Synthetic gold actions must use the frozen timestamp; use `relaxed_verify.py` (ignores read-log table + timestamp) for training-data filtering only.
- tau2 loads only `task_*.json` files from `data/tau2/domains/banking_knowledge/tasks/`; use `TAU2_DATA_DIR=<parallel dir>` with symlinked `documents/`, `db.json`, `prompts/`, `user_simulator/` to run synthetic tasks.
- LiteLLM cost-table "isn't mapped yet" errors are cosmetic; the default NL judge (`gpt-4.1` direct OpenAI) needs the patch to route via OpenRouter (matters for retail/airline, not Banking).
- Teachers sometimes nest `evaluation_criteria` under `initial_state`, return list-valued `relevant_policies`, dict personas, or invent tables → `normalize()` in the generator + validator coercions handle these.
- Qwen3.8-Max/GLM with thinking on exhaust `max_tokens` on long JSON outputs → `reasoning.effort=low`, `max_tokens 16000`.
- OpenRouter rate-limits `qwen/qwen3.8-flash` heavily at concurrency 8.

## GPU run findings (v0, 2026-09-03 ~20:20 UTC onward)
- RTX PRO 6000 Blackwell (sm_120): `flash-linear-attention` 0.4.1 imports and trains; **`flash_attn` and `causal_conv1d` were not installed** in `axolotlai/axolotl-cloud:main-latest` → attention fell back to SDPA and DeltaNet conv to the slow path → **~85 trained tok/s, ~20 min per optimizer step** (17 steps ≈ 6 h ≈ $8 for v0). Peak memory 82 GB at ~13K-token samples, so 16K windows fit on 96 GB with the DeltaNet LoRA targets included (PEFT matched `linear_attn.in_proj_qkv/in_proj_z/out_proj`).
- v0 training signal: loss 0.47 → 0.44 over the first 3 steps; eval ppl 1.63 before training.
- The standalone smoke script crashed on tokenizer input (chat template rejected the tool-call rows outside Axolotl's normalizer) — fixed; Axolotl's own preprocessing handles the rows.
- **SFT v1 assembled**: 97 verified trajectories over 46 tasks (GLM-5.3 39, Max 3, student 55) → 84 train / 13 dev (6 held-out tasks) → **161 windows (~3.7M tokens)**; on HF at `data_v1/`. For v1 training: install `causal-conv1d` + `flash-attn` at job start (time-capped) or use an H100 if wheels are unavailable for sm_120.
- Solvability gate on batch 2: student solved 21/35 tasks (38/70 episodes relaxed); GLM-5.3 solved **0/14** of the student's failures. Across all 79 synthetic tasks: **46 solved by ≥1 agent, 33 solved by nobody → dropped** (`runs/solvability_gate.json`). A 42% unsolvable rate means the generator's gold actions are often not reproducible (mismatches concentrate in `credit_limit_increase_requests`, `credit_card_accounts`, `accounts`); diagnosing before generating batch 3.
- Batch-2 failure analysis (gold vs agent DB diffs): four generator flaws — duplicate/contradictory gold writes (submit twice; approve AND deny), inconsistent initial state (credit-limit task without a credit-card row), free-text args stored in rows (`reason`), `log_verification` for a different user id. Fixes: `relaxed_verify.py` v2 compares rows by content ignoring free-text fields and order-dependent row ids; `validate_task.py` adds consistency checks (duplicates, approve+deny, unknown entity ids, verification user id); generator prompt rule 9. The static checks catch only 3/13 batch-2 unsolvable tasks — the rest are wrong gold *decisions*, so the empirical solvability gate stays mandatory.
- Solvability gate v2 (content-based verifier): **48 solvable / 31 unsolvable** of 79. Verified trajectories: **133** (student 83, GLM-5.3 41, Max 9) → **SFT v1 = 79 train / 17 dev trajectories, 157 train windows (~3.5M tokens)** on HF `data_v1/`.
- Gating policy from batch 3 on: student N=2 first (cheap, self-distillation) → Qwen3.8-Max N=1 only on the student's failures → drop the rest. (GLM-5.3 solved 0/14 of the student's batch-2 failures; teacher-on-failures is low-yield.)
- Batch 3 (stricter validator + prompt rule 9): 40 generated → 31 clean → **20/31 solvable** (65% vs 58% for batch 2) under the student-first gate (student 32/62 episodes; Max 3/14 on the student's failures). **Overall: 110 synthetic tasks, 68 solvable, 42 dropped.**
- **SFT v1 (final for this round): 168 verified trajectories over 68 tasks → 113 train / 18 dev trajectories (10 held-out tasks) → 216 train windows (~4.8M tokens), 48 dev windows**; on HF `data_v1/`. Sources: student 130, GLM-5.3 25 (after per-task caps), Max 8.
- Data generation paused here to reserve budget for evaluation (dev + 97-task test for base and adapter; 5-trial final ≈ $95).
- OpenRouter spend at this point: ~$126 (of ~$300) (rollouts dominate; each teacher episode ≈ $0.4–0.7, student ≈ $0.15, user-sim Flash ≈ $0.02).

## Spend (2026-09-03 19:30 UTC)
OpenRouter ≈ **$85** of the ~$300 distillation budget (baseline $20, teacher/student rollouts ~$55, generation ~$4, probes ~$1). Vast ≈ **$0.5** of $111 (instance #1 destroyed; instance #2 running the smoke+LoRA job at $1.34/hr).

## In flight / next
1. GPU job (Vast 49781678): smoke tests → LoRA v0 → `adapter_v0/` on HF. **Destroy the instance after JOB_COMPLETE.**
2. Student self-rollouts (88 episodes) → SFT v1 (auto). Batch-2 solvability gate: student N=2 (running) → GLM-5.3 on student failures → keep verified passes.
3. Launch `remote_eval.sh` with `ADAPTER=adapter_v0`: synthetic dev set (Flash user-sim), then 97 real tasks under AA config vs 45.4 baseline (1 trial dev; 5 trials for any published number, ~$95).
4. If gain: LoRA v1 on the larger set; GDPval-style track (Flash-Next / GLM-5.3-Flash teachers, Stirrup harness); depth-prune + heal experiment ("smaller and better").
