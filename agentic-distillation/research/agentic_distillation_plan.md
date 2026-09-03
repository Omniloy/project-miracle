# Executable Plan: Distilling a ≤35B Open Model into the Best-in-Class Agent on the Artificial Analysis Agentic Index

**Target:** AA Agentic Index components = GDPval-AA v2 (220 tasks, Stirrup harness, 3-judge pairwise Elo, human expert = 1000) + tau3-Banking (tau2-bench v1.0.1, `banking_knowledge`, `bm25_grep`, GPT-5.4 Mini user sim, 97 tasks × 5 repeats, pass^1).
**Teachers:** `qwen/qwen3.8-max` ($2.00/$6.00 per M; cache read $0.25, cache write $2.50), `z-ai/glm-5.3` ($1.40/$4.40 list — the $1.15/$3.50 figure in the brief is only the Reka endpoint [VERIFIED via OpenRouter models API 2026-09-03]), `z-ai/glm-5.3-flash` ($0.075/$0.25) via OpenRouter. Compute on Vast.ai.
**Date of numbers:** 2026-09-03 unless stated. All leaderboard numbers below carry a verification tag: [VERIFIED], [CAVEAT], [OVERSTATED], [UNVERIFIED].

---

## 0. Framing: what "winning" means here

We are **not** trying to beat frontier closed models. The goal is: **the highest-scoring open-weights model at ≤~35B total parameters on the AA Agentic Index, with a measured, reproducible margin over its own base model on both component benchmarks, using the public harnesses (tau2-bench ≥ v1.0.1, Stirrup) with the AA configuration.**

The bar is concrete and, importantly, **much higher than the stale "no model exceeds ~27% on Banking" assumption** [OVERSTATED — that number is the pre-v1.0.1 tau-Knowledge paper ceiling, 25.52%; the July 2026 regrade raised scores by up to ~9 pts and newer models pushed the ceiling to ~52%; VERIFIED from AA chart JSON + tau2-bench CHANGELOG].

### Current numbers (AA, 2026-09-03)

| Model | Size | tau3-Banking pass^1 (AA) | GDPval-AA v2 Elo | AA Agentic Index |
|---|---|---|---|---|
| **Qwen3.8-27B (xhigh)** — proposed student | 27B dense | **48.0** (medium: 47.4) [VERIFIED] | **1543** (→ 52.2 Index pts) [CAVEAT: from AA GDPval page fetch in research; verifier could not re-extract; re-pull on day 1] | not extracted; ~50 if AA averages the two normalized components [UNVERIFIED — formula not confirmed; Qwen3.6-27B shows gdpval 31.9 / banking 16.7 → agenticIndex 27.5, which is neither a plain nor a 20:14 weighted mean] |
| Qwen3.6-27B (Reasoning) | 27B dense | 16.7 [VERIFIED] | 1138 [VERIFIED; the 1414 in the research brief is UNSUPPORTED] | 27.5 [VERIFIED] |
| Qwen3.6-35B-A3B (Reasoning) | 35B/3B active | 9.3 [VERIFIED] | 1056 [VERIFIED; 1297 is UNSUPPORTED] | 21.6 [VERIFIED] |
| Gemma 4 31B (Reasoning) | 31B dense | ~14.9 [CAVEAT: from benchmarklist.com mirror of AA, not AA page] | 814 [UNVERIFIED] | — |
| gpt-oss-120b (high) | 117B | ~12.8 [CAVEAT: mirror; the "12.03" and "Nemotron 3 5.77" figures in the brief are UNSUPPORTED] | 803 [UNVERIFIED] | — |
| *Teacher:* Qwen3.8 Max | closed | 51.3 [VERIFIED] (Sierra board: 55.15, gpt-5.2 user sim, `alltools` — not comparable) | 1721 [CAVEAT] | — |
| *Teacher:* GLM-5.3 (max) | 744B-class, bespoke license | 50.3 [VERIFIED] | — | — |
| *Teacher:* GLM-5.3-Flash | 320B/18B active, MIT | 47.2 [VERIFIED] — **below the student** | 1765 (Z.ai transcription: 1773) [CAVEAT: rank 5 overall; secondary source for the 1773] | — |
| Top of board | — | Muse Spark 1.3 (max) 52.4 [VERIFIED] | Claude Fable 5.1 (max) 1853 [CAVEAT] | — |

**Consequences that shape the whole plan:**

1. **Qwen3.8-27B is already within 3.3 pts of the best teacher on Banking** (48.0 vs 51.3) and above GLM-5.3-Flash. Naive teacher-trajectory SFT has a ~+3 pt ceiling on Banking. The lever is *verified* data (DB-hash-passing trajectories only — teacher pass^1 ≈ 50%, so unfiltered data is half wrong), the student's own rejection-sampled successes, and consistency (pass^k ≪ pass^1 everywhere: Qwen3.8 Max 55.2 → 35.1 pass^4 on Sierra).
2. **On GDPval-AA the gap is real: ~220 Elo to GLM-5.3-Flash, ~180 to Qwen3.8 Max.** +200 Elo = +10 Index pts on that half. This is where distillation has headroom, and GLM-5.3-Flash is both the strongest and cheapest teacher.
3. **Success criteria (state these in the model card):** tau3-Banking ≥ 51.5 pass^1 (5 repeats, bm25_grep, GPT-5.4 Mini) — i.e. above Qwen3.8 Max — with non-overlapping bootstrap CIs vs the base's 48.0 re-measured under our harness; GDPval-AA estimated Elo ≥ 1650 (+~100) by our single-judge anchored estimate, confirmed by a 3-judge final run. Anything that beats the base on both with CIs is publishable; anything that regresses either is not.
4. **Both test sets are fully public and were public before the teachers' training** (Banking since 2026-03-18; GDPval gold since 2025-10 incl. expert deliverables and rubrics). We cannot fix teacher exposure; we must not add to it and must probe it (§3, §6).

---

## 1. Student model decision

### Recommendation: `Qwen/Qwen3.8-27B` (dense, Apache 2.0)

Why: it is the only ≤35B open model already in the top tier on both components (48.0 Banking, ~1543 GDPval), and beating the base on both while remaining best-in-class is only possible from this starting point. Architecture facts [VERIFIED from HF config.json]: 64 layers = 16 × (3 Gated DeltaNet + 1 Gated Attention), 24 Q / 4 KV heads × 256 dim → 64 KB/token BF16 KV, 262,144 native context (YaRN to 1M), vision encoder (Stirrup's View Image), MTP head, `model_type = qwen3_5` (`Qwen3_5ForConditionalGeneration`) — i.e. it rides the existing Qwen3.5 code path in transformers/vLLM/Axolotl. Model card is Alibaba-self-reported [VERIFIED as self-reported]: Terminal-Bench 2.1 73.0, SWE-bench Pro 61.7 (modified task set), CoWorkBench 70.7 (in-house), JobBench 33.4; no tau/BFCL/decontamination statement on the card.

**Tooling evidence (Sept 2026):**

| Layer | Evidence | Status |
|---|---|---|
| vLLM serving | Day-0 blog 2026-08-12; flags `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`; recipes.vllm.ai says vLLM 0.17.0+; Docker Hub stable tags v0.27.1 (08-11), **v0.28.0 (08-26)** | [VERIFIED] Pin `vllm/vllm-openai:v0.28.0` (first stable after day-0). FP8 checkpoint `Qwen/Qwen3.8-27B-FP8` exists. |
| transformers | `src/transformers/models/qwen3_5` on main; no `qwen3_8` dir needed | [VERIFIED] |
| Axolotl | `examples/qwen3.8/27b-qlora.yaml` on main (`model_type qwen3_5`, CutCrossEntropy, sample packing); pins `fla-core==0.4.1`, `flash-linear-attention==0.4.1` (x86 only) | [VERIFIED on main; check whether in a tagged release] |
| Unsloth | Support shipped 2026-08-14 (`unsloth/Qwen3.8-27B`); docs nav has a Qwen3.8 page. Qwen3.5 fine-tune page says QLoRA not recommended for Qwen3.5 "no matter MoE or dense" but its own code comment says "dense 27B is fine" | [CAVEAT: Unsloth's page is internally inconsistent; the brief's "no official guide exists" is contradicted by Unsloth's nav entry + Axolotl's example] |
| TRL | Works through transformers `qwen3_5` class; no Qwen3.8-specific docs | [UNVERIFIED end-to-end] |
| Kernels | `flash-linear-attention` (fla) required for GDN layers; `causal-conv1d` requirement and sequence-parallel over GDN layers **not confirmed** | [CAVEAT — the #1 technical SPOF; smoke-test in GPU hour 1] |
| LoRA targets | Yotta Labs tutorial targets only `q/k/v/o/gate/up/down_proj`, not DeltaNet `in_proj`; whether GDN projections are LoRA-targetable is unattested | [CAVEAT — test both target sets in smoke run] |

**GPU footprint (arithmetic, not sourced):**

| Mode | Memory | Vast config | $/GPU-h (planning) |
|---|---|---|---|
| bf16 LoRA r=64, seq ≤32K, grad ckpt | 54 GB weights + acts | 1× H100 80GB | ~$2.0 on-demand / ~$1.1 interruptible [CAVEAT: live Vast prices UNVERIFIED — pricing pages are JS-rendered; budget $2–2.5] |
| bf16 LoRA, seq 64K | | 2× H100 | |
| QLoRA 4-bit | ~20 GB | 1× A100/H100 | Axolotl ships it; Unsloth warns on quality → use only for pipeline debugging |
| Full-param SFT (bf16 + fp32 Adam ≈ 432 GB) | | 8× H100 ZeRO-3 / FSDP (4× with 8-bit Adam + offload) | ~$16/h node |
| Serving for eval (FP8, 262K ctx) | 27 GB weights + 12.8 GB KV per 200K seq | 2× H100 TP2 (~7 concurrent 200K seqs) or 4× H100 as 2 replicas | |

Full-param SFT is **ruled out for the first pass** — LoRA first, escalate only if LoRA shows headroom (§5 full tier).

### Fallback: `google/gemma-4-31b` (dense, Apache 2.0 [VERIFIED via Google OSS blog])

Chosen because it removes the *kernel* risk entirely (standard interleaved sliding/global attention → every trainer supports it), has 256K context and the highest published tau2 average in class (76.9 [from model card; a 86.4 blog figure conflicts — use 76.9]). Cost: its Banking baseline is ~14.9 [CAVEAT: mirror] and GDPval ~814 [UNVERIFIED], so it cannot be "best at its size" — it is the fallback for *shipping a working pipeline and a clean margin over base*, not for the leaderboard goal. No coding/terminal numbers on its card (GDPval needs code exec).

Not chosen as fallback: **Qwen3.6-35B-A3B** shares the GDN kernel dependency (does not hedge the SPOF) and starts far lower (9.3 / 1056); it is the right swap only if *rollout cost*, not kernels, becomes binding (3B active ≈ 5–9× cheaper decode; vLLM ≥ 0.19.0 [VERIFIED as "recommended"]). **Qwen3-30B-A3B-Thinking-2507** (tau2 retail 58.8 / airline 58.0 / telecom 26.3) is the zero-risk *pipeline-debug* model — keep it in the same container image. Ruled out: gpt-oss-20b (AA II 15, 131K ctx), Nemotron 3 Nano (bespoke license, Mamba-2 + NeMo-centric), Qwen3-Next-80B (over budget), GLM-5.3-Flash (320B/18B — teacher only).

**Decision gate (GPU hour 1–2, ~$5):** on one H100 with `axolotlai/axolotl-cloud:main-latest`: (a) `pip install fla-core==0.4.1 flash-linear-attention==0.4.1 causal-conv1d` builds; (b) 100-step bf16 LoRA on 16K-token tool-call trajectories with loss decreasing, both LoRA target sets; (c) merged adapter serves in vLLM v0.28.0 and returns parsed `tool_calls` (not tool JSON in `content`); (d) prefix-cache hit rate > 0 on repeated system prompt (`/metrics`). Any failure → Gemma 4 31B.

---

## 2. Eval-first protocol

Rule: **baseline first, under the exact AA configuration, before any data is generated.** Every reported number states harness version, retrieval config, user-sim model, trials, and sampling params.

### 2.1 Infrastructure (one script, one API key)

```bash
pip install vastai && vastai set api-key $VAST_API_KEY
# eval server: on-demand (not interruptible — a half-finished 485-episode run wastes user-sim $)
vastai search offers 'gpu_name=H100_SXM num_gpus=2 verified=true rentable=true reliability>0.98 direct_port_count>=2 disk_space>=300 inet_down>1000 inet_up>500' -o 'dph_total'
vastai create instance $OFFER --image vllm/vllm-openai:v0.28.0 --disk 300 --ssh --direct \
  --env '-p 8000:8000 -e HF_TOKEN=... -e OPENROUTER_API_KEY=...' \
  --onstart-cmd 'nohup vllm serve Qwen/Qwen3.8-27B-FP8 --tensor-parallel-size 2 --max-model-len 262144 --max-num-seqs 32 --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-batched-tokens 32768 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --port 8000 > /workspace/vllm.log 2>&1 &'
```
Notes [VERIFIED from Vast docs]: `--ssh` mode overrides the image ENTRYPOINT, so start vLLM in `--onstart-cmd`; ports must be declared at create time (mapped to random host ports — read from `vastai show instance --raw`); volumes are host-local → all durable state to HF Hub. Interruptible instances are *paused* when outbid, disk preserved, storage still billed [VERIFIED; grace period UNVERIFIED].

Sampling for eval (Qwen's stated repro condition): thinking mode, `temperature 1.0, top_p 0.95, top_k 20`; record `reasoning_effort` (AA reports both medium and xhigh for the 27B; 48.0 vs 47.4 — Banking is not compute-limited).

### 2.2 tau3-Banking (AA-equivalent)

```bash
git clone https://github.com/sierra-research/tau2-bench && cd tau2-bench
git checkout v1.0.1   # tag → fc0055d, 2026-07-16 [VERIFIED; the brief's 07-22 date is wrong]
uv sync --extra knowledge   # bm25_grep needs no embeddings, no sandbox
export HOSTED_VLLM_API_BASE=http://localhost:8000/v1   # avoids the openai/ base-URL collision with the user sim
export OPENAI_API_KEY=...   # for GPT-5.4 Mini user sim
tau2 run --domain banking_knowledge --retrieval-config bm25_grep \
  --agent-llm hosted_vllm/Qwen/Qwen3.8-27B-FP8 \
  --agent-llm-args '{"temperature":1.0,"top_p":0.95,"extra_body":{"top_k":20}}' \
  --user-llm openai/gpt-5.4-mini --user-llm-args '{"reasoning_effort":"medium"}' \
  --num-trials 5 --max-steps 200 --seed 300 --max-concurrency 16 --save-to qwen38_27b_baseline
tau2 view
```
[CAVEAT: CLI flags `--agent-llm/--agent-llm-args/--user-llm/--num-trials/--num-tasks/--max-steps/--max-concurrency/--seed/--retrieval-config` are VERIFIED from cli.py; the `hosted_vllm/` LiteLLM provider path for `api_base` is the presumed fix for the agent/user-sim base-URL collision and must be confirmed on the first run; there is no `--judge` flag — the NL judge is a constant in config.py and is **irrelevant for Banking: 0 of 97 tasks have NL assertions (88 DB-state + 9 ACTION reward)** [VERIFIED from repo].]

Expected result: 48.0 ± ~3 (5 trials). If we cannot land within ±3 of AA's 48.0 with the same config, fix the harness before anything else.

**Cost per full run (485 sims):**
- GPU: 2× H100 FP8 TP2, 6–8 h → ~$25–40 (up to 2× if prefix-cache hit rate is poor — under `bm25_grep` the shared prefix is only system prompt + tool schemas, not the 195K corpus [VERIFIED]; vLLM GDN prefix caching has open hit-rate bugs #40696/#45238 [VERIFIED]).
- User sim: GPT-5.4 Mini $0.75/$4.50 (cache read $0.075) [VERIFIED]; ~60K in / ~5K out incl. reasoning per sim → ~$0.07–0.20/sim → **$40–100/run**.
- Judge: $0.
- **Total ≈ $70–150 per 5-trial run; ~$20–35 for a 1-trial run.**

**Statistics:** pass^1 = mean success over 485 sims. SE at p≈0.5 ≈ 2.3 pts if independent; sims are clustered by task so the effective SE is worse. Use task-level bootstrap CIs. **Rule:** 1 trial for smoke checks only; 3 trials for go/no-go gates; 5 trials for any number we publish. Report pass^k too (consistency is a cheap, honest secondary win).

**Dev-time cheap approximation:** never iterate on the 97 tasks. Hold out ~40 of our own synthetic tasks (§3) in a *separate synthetic institution* as the dev set and run 3 trials each (~120 sims, GLM-5.3-Flash as user sim ≈ $5 + ~1 GPU-h). Touch the 97 only at tier gates.

### 2.3 GDPval-AA v2 (approximate; exact AA Elo is not reproducible)

Facts [VERIFIED]: Stirrup HEAD 247f24d has `AGENT_MAX_TURNS=30`, `CONTEXT_SUMMARIZATION_CUTOFF=0.7`, `SANDBOX_TIMEOUT=600s`; **no GDPval task loader, judge, Abandon tool, or Elo pipeline is shipped** — we write them. AA production: 250 turns, E2B sandbox, tools Web Fetch / Web Search (Brave) / View Image / Code Exec / Finish / Abandon; judges GPT-5.5 (medium), Gemini 3.1 Pro Preview (high), Claude Opus 4.8 (high); docs parsed as text + rendered images; Bradley-Terry anchored to human deliverables = 1000; balanced then active sampling. The 220 AA tasks = HF `openai/gdpval` (220 rows, 44 occ × 5) [VERIFIED by counts + AA statement; not by task-ID cross-check].

```bash
pip install 'stirrup[all]'   # E2B_API_KEY, BRAVE_API_KEY
```
```python
from stirrup.clients import ChatCompletionsClient
client = ChatCompletionsClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen3.8-27B-FP8",
                               max_tokens=32768, context_window_tokens=262144, api_key="dummy")
# our loop: load HF openai/gdpval row -> stage reference_files into E2B -> run agent (max_turns=250, add Abandon tool)
# -> collect finish.paths -> render deliverables (LibreOffice headless -> PDF -> PNG) -> judge
```

**Two-level grading:**
1. **Fast dev signal (~$5/run):** grade outputs against the row's `rubric_json` with GLM-5.3-Flash (rule checks for file existence/extension/sheet names, LLM for content). *Eval-only use of the rubrics; they never enter training data.*
2. **Anchored Elo estimate:** single frontier judge (GPT-5.5 medium via OpenRouter; same text + page-image parsing) doing pairwise student-vs-**human expert deliverable** (185/220 tasks have one), both orderings, plus student-vs-GLM-5.3-Flash outputs as a second anchor of known Elo (~1765). BT logistic: `Elo ≈ 1000 + 400·log10(p/(1−p))` from win rate p vs humans; cross-check with the teacher anchor. Expect ±30–60 Elo noise and judge bias (OpenAI's single grader agreed with humans 65.7% vs 70.8% human–human [VERIFIED, OpenAI harness not AA's]).

**Dev subset:** 44 tasks (1 per occupation, fixed seed), single judge → per run ~$15 GPU (2× H100, ~2 h) + E2B ~$5 + judge ~$15–30 ≈ **$40–60**. **Final:** all 220, 3 judges × 2 orderings × (vs human + vs teacher anchor) ≈ 100M judge tokens ≈ **$300–600**, + GPU ~$40 (2× H100, 8–10 h) + E2B ~$30 → **~$400–700**. Do the 3-judge run twice in the whole project (baseline and final).

E2B [VERIFIED]: Hobby = $100 credit, **1 h session cap** (kills long tasks), 20 concurrent; Pro $150/mo, 24 h, 100 concurrent; ~$0.11/sandbox-hour. Pro is required from the first full 220-task run. Brave Search quota: UNVERIFIED — check tier before the run; cache fetches.

Historical anchor for token volume [VERIFIED via AA tweet]: a full v1 run was 40M tokens (GPT-5.1, $88) to 250M (GPT-5.2, $620) — our student is self-hosted so this is GPU time, not API $.

---

## 3. Data pipeline

### 3.1 Contamination boundaries (hard rules, enforced by scripts in CI)

| Source | Status | Rule |
|---|---|---|
| `tau2-bench/data/tau2/domains/banking_knowledge/tasks.json`, `tasks/*.json`, `tasks_voice.json`, their `notes`, gold `actions`, personas (Sarah Bosch, Wei Chen, …), the `db.json` entity IDs those 97 tasks touch, any paraphrase | **FORBIDDEN** (this *is* the AA test set; public since 2026-03-18) | 8-gram overlap + embedding cos ≥ 0.8 blocklist vs the 97 `user_scenario.instructions`; entity-ID blocklist |
| Teacher or student rollouts *on* task_001–task_102 | **FORBIDDEN** | never launch `tau2 run` on banking for data generation |
| Divergence-point DPO recipe (arXiv 2606.23112) as published | **FORBIDDEN** — paper states "all trajectory experience and evaluation come from the same tau2-bench task set… within-benchmark self-improvement", banking pairs included [VERIFIED — CRITICAL]; its banking 0.062→0.072 is in-sample [CAVEAT: baseline is 0.062, not 0.058] | reuse the *method* only on synthetic domains |
| The 698 Rho-Bank documents + `tools.py` as **grounding for new tasks** | **HIGH RISK** [verifier] — 97 tasks over 21 categories means new tasks on the same docs collide on policy combinations/amounts | Default: do not use. Optional Track B (§3.2) only at standard+ tier, ≤20% of banking data, with dedup + disclosure |
| The 698 docs + `tools.py` + `db.json` as an **architectural template** (schema, tool-discovery protocol, retrieval configs, user-sim guidelines, `policy_header.md`) | Allowed | copy structure, regenerate all content |
| `openai/gdpval` `prompt`, `reference_files`, `deliverable_files`, `rubric_json` | **FORBIDDEN** for training (eval-only) | grep the GDPval canary string; 8-gram overlap vs the 220 prompts; never stage the reference files into a training sandbox |
| tau2 airline/retail/telecom **train** splits (30/74/74) + telecom `tasks_full.json` (2,285 minus base test IDs) | Safe (disjoint domains/DBs from banking) | ≤20% of banking-style mix; do not use test splits if we ever report tau2 |
| `nvidia/Nemotron-Agentic-v1` | Low risk; **sizes corrected**: 19,028 `interactive_agent` + 316,094 `tool_calling` = 335,122 [VERIFIED; the "1,127,100 / 70,794" figures are UNSUPPORTED]; no decontamination statement, generators Qwen3-235B/GPT-OSS-120B | filter: drop any conversation whose tool names match banking tools or whose text hits the blocklists |
| Toucan-1.5M (MCP tools; BFCL V3 gain for Qwen2.5-7B only 55.1→58.3 [VERIFIED — modest]), Open-SWE-Traces (207,489 traces, students Qwen3-30B-A3B, best 61.7 SWE-V [VERIFIED]) | Low/none | optional general warm-up; same filters |
| Hermes 4 | Low; **19B tokens, not 60B** [OVERSTATED] | optional |

### 3.2 Banking-style data (target: tau3-Banking)

**Teacher memorization probe (day 3, ~$20):** give Qwen3.8-Max and GLM-5.3, *without tools or KB*, 10 banking task instructions and ask for the exact final action and amount, and for verbatim policy text by document title (e.g., task_074's $14.50 refund). High recall ⇒ teacher has ingested the repo; its 51.3 is partly memorization and its edge on *new* tasks will be smaller. Risk rating [verifier]: Qwen3.8-Max MEDIUM-HIGH (Sierra lead 55.15 vs Opus 5 48.71, Qwen3.6 card self-reports "TAU3-Bench 67.2", AgenticQwen trains the Qwen lineage in tau2-measured RL rounds, no decontam statement anywhere); GLM-5.3 MEDIUM-LOW (GLM-5.2 banking 37.11 unremarkable; GLM-5 report has no decontam section). Probe result decides the 70/30 teacher split below.

**Track A (default): synthetic institutions.** Build 5 fictional banks (standard tier) using the tau2-bench `banking_knowledge` code as a template but with *all content regenerated*:
1. Teacher (GLM-5.3) writes a product taxonomy (~20 categories), typed variables (fees, rates, limits, tie-break rules), then ~600–700 natural-language policy articles per institution with cross-document interdependencies (fee allowances across accounts, promotional-vs-base-rate traps) — the failure modes that dominate the real benchmark (search inefficiency/assumptions ~23%, product interdependencies ~14.5%, subtask ordering ~5%, over-trusting user ~4% [VERIFIED from tau-Knowledge paper]).
2. Generate 40–60 discoverable tools with new name+suffix patterns (`file_dispute_case_7712`), 14 permanent tools, seed DB with fresh customers/accounts/transactions; wire into a copied `banking_knowledge` domain with `bm25_grep` retrieval; unit-test replay.
3. APIGen-MT-style task blueprints: teacher writes `user_scenario.instructions` in the repo's "How to Behave" style + gold action list; **execution-verify by replay → DB hash**; LLM committee check for policy consistency; keep ~70% (APIGen-MT yields 70% with feedback).
4. Rollouts inside the same orchestrator: agent = teacher via OpenRouter, user sim = GLM-5.3-Flash with the repo's `simulation_guidelines_tools.md`, `max_steps 200`, `bm25_grep`. Keep **only DB-hash-passing** trajectories (expected 45–55% teacher pass rate on new tasks). Also roll out the **student itself** (N=4) and keep its passes — self-distillation is free on our GPU and is on-policy.
5. Behavior-tree expansion (AgenticQwen): branch each blueprint on a policy condition and add an adversarial user variant that steers toward the wrong branch.
6. Hold out 1 institution × 40 tasks as the dev set.

**Track B (optional, standard+ only, ≤20%):** new tasks on the real Rho-Bank KB with fresh entities (new `user_id`s injected into a `db.json` copy). Verifier rates collision risk HIGH; if used: require zero overlap with any of the 97 tasks on (discoverable-tool set ∪ required_documents) *and* embedding cos < 0.8 on scenario text, and disclose in the model card. Skip entirely at pilot tier.

**Auxiliary (≤20%):** airline/retail/telecom train splits, teacher rollouts with the same DB-verified filter — teaches one-action-per-turn hygiene, verification, ###STOP### handling; do not expect it to move Banking (frontier models score 80–99 there and 40–52 on Banking).

**Volumes and cost (retrieval mode: ~40 steps × ~25K ctx ≈ 1M input / 40K output per trajectory, uncached):**

| Item | Pilot | Standard | Full |
|---|---|---|---|
| Synthetic institutions / tasks | 2 / 400 | 5 / 1,800 | 10 / 6,000 |
| Teacher rollouts (N=2–4) | 800 | 4,500 | 15,000 |
| Teacher mix | 100% GLM-5.3 | 70% GLM-5.3 / 30% Qwen3.8-Max (hardest 30% of tasks) | same, plus best-of-4 Max on dev-failure clusters |
| Passing teacher trajectories (~50%) | ~400 | ~2,200 | ~7,500 |
| Student self-rollouts (N=4, GPU only) | 1,600 | 7,200 | 24,000 |
| Teacher API cost (uncached: GLM-5.3 ≈ $1.58/traj, Max ≈ $2.24; with caching ≈ $0.7 / $1.2) | ~$600–1,300 → **cap $700** by relying on caching + Flash user sim | ~$3.5–7.5K uncached → **~$1.8–2.5K with caching** | ~$6–8K with caching |
| User sim (Flash, ~$0.03/episode) | $25 | $150 | $500 |
| Token volume for SFT (assistant tokens only, ~40K/traj) | ~15M | ~90M | ~300M |

[CAVEAT: the per-trajectory token estimate is arithmetic; measure on the first 50 rollouts and rescale. OpenRouter routes GLM-5.3 across ~25 providers with different prices/logprob support — pin the provider.]

**GLM-5.3-Flash is not a banking teacher** (47.2 < student 48.0); it is the user simulator and first-pass judge only.

### 3.3 GDPval-style data (target: GDPval-AA v2)

**Teacher:** GLM-5.3-Flash for volume (Elo ~1765, ≈$0.40/trajectory uncached at ~4.8M in / 150K out, ~$0.15 cached), Qwen3.8-Max (1721) for a best-of-4 slice on hard task types (~$10.5 uncached / ~$3–4 cached). Probe both teachers first: ask, without tools, for the deliverable description of 5 GDPval task_ids by prompt text; near-verbatim rubric recall ⇒ that teacher must not be prompted with anything resembling the 220 prompts (Z.ai advertises GDPval-AA v2 gains for GLM-5.3-Flash: 1773 vs GLM-5.2 1504 [CAVEAT: secondary transcription]).

**Synthesis (OfficeVerse-style, small):** Solar Open 2's pipeline is the only published recipe — 11 domains × 12 task types (cognitive operation × deliverable format), grounded in real public data, weighted rubrics with deterministic gates, "decontaminated against the benchmarks used" [VERIFIED; dataset sizes undisclosed; Ko-GDPval is an in-house 170-task Korean benchmark, not GDPval-AA].
1. Seed from **O*NET occupation descriptions** for the 44 GDPval occupations (never from the 220 prompts).
2. Deliverable mix matched to the test distribution [VERIFIED from HF]: pdf ~37%, xlsx ~28%, docx ~28%, pptx ~7%; ~57% of tasks with 1–2 reference files, ~43% none; prompt = role framing + numbered instructions naming the exact output filename/format (mean ~2.2K chars).
3. Per task: context → synthetic reference files (xlsx/pdf/docx built from real public data via code) → prompt → weighted rubric mixing rule-checkable (file exists, sheet named X, section Y present) and LLM-judged criteria; deterministic gate at each stage.
4. Decontam: canary-string grep, 8-gram overlap vs the 220 prompts, embedding dedup.
5. Rollouts in the **real Stirrup harness** (native function calling, `code_exec`/`web_fetch`/`web_search`/`view_image`/`finish` + our Abandon tool, `max_turns` 100–250, summarization on) with `stirrup[docker]` for cheap volume and E2B for a fidelity slice. Give the **teacher** the GDPval-paper formatting/render-and-inspect system prompt (it moved OpenAI's win rate +5pp, pptx formatting errors 86%→64%, self-inspection 15%→97% [VERIFIED]) and train the **student on the plain Stirrup base prompt** (prompt distillation).
6. Filter: rubric pass (rule + Flash judge), fabrication check, "rendered and inspected the deliverable" behavior present, finish paths valid. Pairwise judge teacher-vs-student outputs; keep teacher wins for SFT, both for DPO pairs.

| Item | Pilot | Standard | Full |
|---|---|---|---|
| Synthetic tasks | 200 | 2,000 | 6,000 |
| Teacher trajectories | 200 Flash | 2,000 Flash + 200 × best-of-4 Max | 6,000 Flash + 600 × best-of-4 Max |
| Kept after filter (~55–65%) | ~120 | ~1,300 | ~4,000 |
| Teacher API | ~$60 | ~$400–900 (Max slice dominates) | ~$1.5–3K |
| Sandbox | Docker (free) + E2B Hobby credit | E2B Pro $150/mo + ~$50 | Pro + ~$150 |
| Judge/filter (Flash + some GPT-5.5) | ~$20 | ~$200 | ~$600 |

### 3.4 General mix

70–80% general/agentic warm-up (filtered Nemotron-Agentic-v1, a slice of Toucan, a chat/writing set to protect deliverable prose) vs 20–30% target data — AgentTuning's 0.2/0.8 ratio [VERIFIED] and Nemotron practice (tool calls in thinking-mode format). Writing quality matters directly: GDPval is judged pairwise against human experts.

---

## 4. Training

### Stage 0 — smoke tests (day 1–2, <$20)
As in §1's decision gate. Also validate the trajectory→chat-template conversion round-trips through vLLM's `qwen3_coder` parser (same tool schema at train and test time).

### Stage 1 — SFT on verified trajectories
- **Method:** bf16 LoRA (r=64, α=128, dropout 0.05) on all linear projections incl. DeltaNet `in_proj` if the smoke test shows it trains; Axolotl `examples/qwen3.8/27b-qlora.yaml` as the base config with `load_in_4bit: false`, `adapter: lora`, `sample_packing: true`, CutCrossEntropy plugin, `chat_template: tokenizer_default` (pass `reasoning_effort`).
- **Loss:** assistant tokens only (thinking + tool calls); mask system, user, tool-result messages — APIGen-MT, Nemotron, Hermes 4 (loss-masking) all do this.
- **Long contexts:** no verified sequence parallel for GDN layers → **chunk** trajectories into per-turn samples with truncated history (system + policy summary + last ~24K tokens), keeping full-history samples up to 32K (1× H100) / 64K (2× H100). Include summarized-context bridge messages so the student learns Stirrup's 70% summarization regime.
- **Hyperparameters others used:** APIGen-MT full SFT ≤3 epochs, AdamW, bf16, ZeRO-3; AgentTuning 0.2/0.8 mix; Amity RL temp 0.9. Ours: LoRA lr 1e-4 cosine, warmup 3%, 2 epochs over target data (general data 1 epoch), effective batch ~64 sequences, grad ckpt, max seq 32K.
- **Compute:** pilot ~15–25M target tokens + ~40M general → 1× H100 ~6–8 h ≈ $15; standard ~100M + 150M → 2× H100 ~20–30 h ≈ $100–150 (interruptible OK: checkpoint every 15 min, auto-resume).
- **Log:** train/eval loss split by source; exact-match accuracy of tool-call JSON on held-out teacher turns; parse-failure rate through vLLM on the dev set; dev-set (synthetic institution) pass^1 at 3 trials; GDPval 44-task rubric pass rate; throughput and prefix-cache hit rate.

### Stage 2 — rejection-sampling FT + DPO (the highest-EV stage on Banking)
- Roll out the Stage-1 student N=4–8 on all synthetic banking tasks and 2,000 GDPval-style tasks (GPU only; Flash user sim ≈ $0.03/episode). Keep DB-hash passes / rubric-and-judge wins; dedupe near-identical trajectories; prefer shortest passing trajectory per task (minimal, exact write set — an extra `unlock` changes the DB hash).
- SFT again (LoRA continued) on teacher passes + student passes (RAFT-30B-A3B precedent: Qwen3-30B-A3B-Thinking-2507 on synthetic retail with disjoint entities → 82.5 retail pass^1 [VERIFIED; custom submission, Claude user sim]).
- **DPO** on success/failure pairs aligned at the divergence point (method from 2606.23112, **data only from synthetic domains**): QLoRA/LoRA, β=0.1, lr 5e-7, 1 epoch, seq 8–16K, 1× H100, ~2–4 h. Monitor: chosen-reward stays positive (negative chosen reward ⇒ catastrophic forgetting in the paper); identical prompts at train and inference (their prompt mismatch dropped reward 0.349→0.296). Expected effect is small (+1–3 pts); it is a cheap add-on, not the lever.

### Stage 3 (optional; standard = 100-step probe, full = real run) — RL / on-policy distillation
- **Not available:** textbook on-policy distillation through OpenRouter — the API returns top-20 logprobs on the teacher's *own* output tokens only, no echo/prompt_logprobs on student continuations [VERIFIED from OpenRouter parameters doc + models API]; GLM tokenizer ≠ Qwen tokenizer anyway.
- **GRPO with verl:** LoRA policy + vLLM rollout workers on 4× H100 (dense 27B full-param multi-turn RL is 4,600–9,200 H100-h in SWE settings [VERIFIED: SkyRL-Agent 4,601 vs DeepSWE 9,180] — out of scope). Reward: sparse end-state DB hash + format penalty **only**; dense per-turn rewards degraded tau results by 6.5pp (Amity, arXiv 2604.02869 [VERIFIED: Qwen3-30B-A3B 58.0→69.5 on tau2 airline, 8× H20, verl+Megatron — but trained on tau-bench v1 airline, i.e., near-in-domain; cite as within-domain only, do not extrapolate to Banking]). N=8 rollouts, 8 prompts/step, temp 0.9, max 40 turns, user sim Flash, 200–400 steps; ~1–2 steps/h → 100–200 h × $8/h ≈ **$0.8–1.6K GPU** + ~$300 user-sim. For GDPval-style tasks use a RULER-style relative LLM-judge reward (Flash judge ranks the group) on ≤500 tasks; budget-cap $1K.
- **True OPD (full tier only):** self-host a same-tokenizer open teacher with vLLM `prompt_logprobs`. Candidate: Qwen3.8-Flash-Next (open weights — Axolotl has an example; GDPval-AA 1743 > student 1543, but Banking 45.4 < student 48.0) → OPD for the **GDPval half only**; reverse-KL per-token advantage, 4 samples/prompt, 150–300 steps [CAVEAT: Thinking Machines' recipe; the widely quoted "1,800 GPU-h" is Qwen's tech-report OPD row (74.4% AIME), not TM's run — OVERSTATED in the brief]. Size/serving cost of Flash-Next: UNVERIFIED — check before budgeting.

### Checkpoints
`hub_strategy="every_save"`, private repo per stage (`<org>/qwen38-27b-agent-sft-v1`, `-rft-v1`, `-dpo-v1`); merge LoRA before final eval (`vllm serve` on merged FP8 or `--enable-lora --lora-modules` for quick dev evals). 54 GB BF16 upload ≈ 15 min at 500 Mbps → choose hosts with `inet_up>500`, check egress fees in the offer JSON. Every checkpoint's card records: harness versions, retrieval config, user sim, trials, sampling params, data sources and dedup stats.

---

## 5. Budget tiers

| | **Pilot ($600–1,000)** | **Standard ($3.5–5K)** | **Full ($15–25K)** |
|---|---|---|---|
| Smoke tests + baselines | $20 GPU; Banking 1-trial ($30) + 3-trial ($80); GDPval 44-task single judge ($50) | + Banking 5-trial baseline ($120); GDPval 220 single-judge ($150) | + GDPval 220 3-judge baseline ($500) |
| Banking data | 2 institutions, 400 tasks, 800 GLM-5.3 rollouts (cached) ≈ $400; Flash user sim $25 | 5 inst., 1,800 tasks, 4,500 rollouts 70/30 ≈ $1.8–2.5K | 10 inst., 15K rollouts ≈ $6–8K |
| GDPval data | 200 Flash trajectories, Docker sandbox ≈ $80 | 2,000 Flash + 200×4 Max, E2B Pro ≈ $600–1,100 | 6,000 + 600×4 ≈ $2–3.5K |
| Training | LoRA SFT 1× H100 ≈ $15; RFT/DPO ≈ $30 | SFT 2× H100 ≈ $150; RFT rollouts ≈ $80 GPU + $150 sim; DPO $30; optional 100-step GRPO probe ≤ $600 | full-param SFT 8× H100 ≈ $300; GRPO 400–800 steps ≈ $3–4K GPU + $1K sim; OPD teacher hosting ≈ $2–3K |
| Final evals | Banking 3-trial ($80); GDPval 44 ($50) | Banking 5-trial ×2 ($250); GDPval 220 single-judge ×2 ($300) + one 3-judge ($500) | Banking 5-trial ×3 ($400); GDPval 3-judge ×2 ($1K) |
| Contingency | 15% | 15% | 20% |
| **Expected outcome** | Pipeline proven end-to-end; harness reproduces 48.0 ± 3; Banking +0–3 (likely inside noise); GDPval rubric pass rate up, Elo movement unmeasurable | **Banking 50–54** (≥ Qwen3.8 Max at 51.3 is the target), pass^4 up ≥ 5; **GDPval +60–120 Elo** (≈1600–1660) | Banking 52–56 (ceiling ~52–55 set by frontier); GDPval +120–200 Elo; consistency gains |
| **Buys nothing on** | any RL; teacher Max; 3-judge eval | dense full-param RL; OPD | beating frontier |

**Decision gates**
- **Pilot → Standard (end of week 2):** all four smoke tests pass on Qwen3.8-27B; our Banking baseline within ±3 of 48.0 at 3 trials; pilot SFT+RFT model **does not regress** Banking (3 trials) *and* improves dev-set pass^1 by ≥ 5 pts; GDPval 44-task rubric pass rate up ≥ 5 pts; teacher memorization probes documented. If Banking regresses > 3 pts → data/format bug; do not scale.
- **Standard → Full (end of week 8):** 5-trial Banking ≥ +3 pts over our own baseline with non-overlapping task-bootstrap 90% CIs, or ≥ 51.5 absolute; GDPval single-judge estimate ≥ +60 Elo, confirmed by the 3-judge run within ±40; no regression on a small general-capability check (IFEval-style + a writing rubric). If Banking gain < 2 and GDPval gain > 80 → Full tier funds GDPval only (OPD from Flash-Next) and freezes Banking data.

---

## 6. Risks and single points of failure

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| 1 | **GDN training kernels** (fla/causal-conv1d build, DeltaNet LoRA targets, no sequence parallel) on Qwen3.8-27B | Medium / blocks student | Hour-1 smoke test; Axolotl main pins fla 0.4.1; chunked-context SFT; fallback Gemma 4 31B in same image |
| 2 | **Small Banking ceiling**: student 48.0 vs best teacher 51.3; teacher pass rate ~50% on new tasks | High / caps gains at +3–5 | DB-verified filtering, self-distillation, consistency (pass^k) as secondary metric; set honest targets; do not promise > 55 |
| 3 | **Contamination** (97 tasks, 220 gold tasks + deliverables + rubrics public; divergence-point recipe train-on-test; Track B collisions) | Medium / invalidates result | Blocklists + canary grep in CI; synthetic institutions default; Track B ≤20% with dedup + disclosure; eval-only use of `rubric_json`; never roll out on the 97 |
| 4 | **Teacher memorization** (Qwen3.8-Max MEDIUM-HIGH; GLM-5.3 MEDIUM-LOW; GLM-5.3-Flash on GDPval MEDIUM) | Medium / inflated teacher edge, stylistic leakage | No-tools probes (§3.2, §3.3) before spending; shift split toward GLM-5.3 if positive; seed GDPval tasks from O*NET only |
| 5 | **Judge cost and noise** on GDPval (single judge ±30–60 Elo; 65.7% agreement) | High / mis-ranked checkpoints | Two-level grading; 44-task dev subset; 3-judge only at gates; report win-rate vs humans with CIs, call it "estimated Elo" |
| 6 | **Vast preemption / host quirks** (pause on outbid, storage billed, host-local volumes, entrypoint override, random port mapping) | Medium / lost runs | Interruptible only for resumable SFT (ckpt every 15 min → HF); on-demand for eval servers; per-task idempotent eval writes; no reliance on volumes |
| 7 | **Teacher rate limits / routing**: qwen3.8-max has a single provider (Alibaba); GLM-5.3 spans ~25 providers with different prices and logprob support | Medium / stalls or 2× cost | Pin provider (`provider.order`), exponential backoff, concurrency ≤ 16, nightly generation, cache-friendly prompt ordering |
| 8 | **vLLM prefix caching on GDN** (open bugs: 0% hits below block size, align-mode checkpoint landing in unique tokens) | Medium / eval 2–4× slower | Check `/metrics` in the pilot; fall back to 4× H100 (2 replicas) |
| 9 | **E2B 1 h Hobby cap; Brave quota** | High if ignored / truncated GDPval runs | Pro tier from first 220 run; Docker backend for training data; Brave tier check + fetch cache |
| 10 | **LiteLLM base-URL collision** (local agent vs OpenAI user sim) | High if ignored / silent wrong-model runs | `hosted_vllm/` provider for the agent; assert model IDs in logs on first 5 sims |
| 11 | **Version drift / non-comparable numbers** (pre-v1.0.1 vs v1.0.1; AA GPT-5.4 Mini vs Sierra gpt-5.2 sims; `bm25_grep` vs `alltools`) | High / apples-to-oranges | Pin tau2-bench v1.0.1 (fc0055d), vLLM v0.28.0; every number carries config; re-pull AA board on eval days |
| 12 | Unverified planning prices (Vast $/h, GPT-5.5 judge price, Flash-Next size) | Medium / budget ±30% | 15–20% contingency; confirm on day 1 |

---

## 7. Week-by-week timeline (Standard tier, 8 weeks, one engineer + scripts)

- **Week 1 — Infra + baselines.** Vast scripts; vLLM v0.28.0 serving Qwen3.8-27B-FP8; four smoke tests (kernels, LoRA, tool-call parsing, prefix cache); tau2-bench v1.0.1 install; resolve LiteLLM base-URL; Banking 1-trial then 3-trial baseline (target 48 ± 3); Stirrup task loader + Abandon tool + rendering + judge scripts; GDPval 44-task baseline with single judge + rubric grading. Teacher memorization probes. **Gate: student confirmed or fallback.**
- **Week 2 — Environment generator.** Copy `banking_knowledge` as template; generate institutions #1–2 (taxonomy → docs → tools → DB → replay tests); blueprint generation + execution-verify; contamination CI (blocklists, canary, dedup); first 200 teacher rollouts (measure real tokens/trajectory and pass rate; rescale budget). O*NET-seeded GDPval task generator with deterministic gates; 100 Flash trajectories in Stirrup/Docker.
- **Week 3 — Data at scale.** Institutions #3–5; 1,800 tasks; 4,500 teacher rollouts (70/30 GLM-5.3/Max, cached); 2,000 Flash GDPval trajectories + 200 × best-of-4 Max; filtering, judge passes; assemble mix with Nemotron-Agentic-v1 (filtered) + general chat. Freeze dev set (institution #5 × 40 tasks).
- **Week 4 — Stage-1 SFT.** 2× H100 LoRA, 2 epochs; dev-set 3-trial evals on 3 checkpoints; GDPval 44-task rubric + single judge; merge, push to HF. Banking 3-trial on the 97 for the best checkpoint only.
- **Week 5 — Stage-2 RFT.** Student rollouts N=8 on all synthetic tasks + 2,000 GDPval tasks; keep passes; continued SFT; divergence-point pairs → DPO (synthetic only), chosen-reward monitoring. Dev evals.
- **Week 6 — Optional GRPO probe** (100 steps, LoRA, verl, 4× H100, ≤ $600) on synthetic banking; keep only if dev pass^1 improves ≥ 3 with no format regressions. Otherwise spend the week on a second RFT round.
- **Week 7 — Final evaluation.** Best checkpoint: Banking 5 trials (AA config) ×1, plus base model re-run 5 trials for a paired comparison; GDPval 220 single-judge for student and base, then 3-judge on the student; task-level bootstrap CIs; general-capability regression check.
- **Week 8 — Release + gate.** Model card (all configs, data sources, dedup stats, probes, CIs), HF release (merged BF16 + FP8), reproduction scripts, submission to Sierra board (note: Sierra uses `alltools` + gpt-5.2 — separate run, ~$150) and AA request. Standard→Full gate decision.

---

## 8. Verification verdicts applied (summary of corrections made in this plan)

- **"No model exceeds ~27% on Banking" — OVERSTATED/stale.** Pre-v1.0.1 paper ceiling was 25.52; AA v1.0.1 board tops at 52.4 (Muse Spark 1.3), teachers at 51.3 / 50.3, student at 48.0. All targets in §0 and §5 use v1.0.1 numbers; §6 #11 forbids mixing eras.
- **Small-model GDPval Elos — UNSUPPORTED as briefed.** Qwen3.6-35B-A3B is 1056 (not 1297), Qwen3.6-27B is 1138 (not 1414); Qwen3.8-27B's 1543 is retained with a CAVEAT (not re-extracted) and must be re-pulled on day 1.
- **GLM-5.3 price — CORRECTED** to the $1.40/$4.40 list price (the $1.15/$3.50 figure is one provider); Qwen3.8-Max cache *writes* cost $2.50/M, so caching gains are smaller than assumed; GPT-5.4 Mini is $0.75/$4.50.
- **Nemotron-Agentic-v1 sizes — CORRECTED** (19,028 + 316,094 = 335,122); no decontamination statement → filtered before use. **Hermes 4 — CORRECTED** to 19B tokens. **MUA-RL** is Qwen3-based (not Qwen2.5); its telecom gain is +3.5pp — used only as a "hard domains move little" prior.
- **Divergence-point DPO — CRITICAL:** published recipe is train-on-test on tau2 incl. banking; adopted as a *method on synthetic domains only*; its banking baseline is 0.062 (not 0.058).
- **Amity MT-GRPO +11.5pp — VERIFIED but within-domain** (tau v1 airline → tau2 airline); cited as such, not extrapolated to Banking. **Thinking Machines OPD "1,800 GPU-h" — OVERSTATED** (Qwen's tech-report row); OPD is anyway unavailable via OpenRouter (VERIFIED) and appears only as a self-hosted full-tier option.
- **698 docs as task grounding — HIGH risk per verifier** vs "allowed environment" per research: resolved as Track A (synthetic institutions, default) vs Track B (real KB, ≤20%, standard+ only, dedup + disclosure).
- **Tooling claims — CAVEATED:** Unsloth's QLoRA warning is internally inconsistent and Axolotl ships a Qwen3.8 QLoRA example; "no official fine-tune guide" is contradicted by Unsloth's nav + Axolotl's example; kernel builds and DeltaNet LoRA targetability remain UNVERIFIED → hour-1 smoke test. vLLM: architecture is `qwen3_5`, so v0.27.1 may already load it; v0.28.0 is the first stable tag after day-0.
- **Dates/facts — CORRECTED:** tau2-bench v1.0.1 tag is fc0055d, 2026-07-16 (not 07-22); the "GPT-5.2 24.74→32.22" pair is a synthesis, not a CHANGELOG entry; the Fireworks-submitted GLM-5.2 29.64 used v1.0.0 and a GLM user sim (not comparable); Qwen3.5-397B's 9.79 used embedding retrieval, not `alltools`.
- **Banking judge cost — $0** (0 NL-assertion tasks, VERIFIED); only the GPT-5.4 Mini user sim is billed.
- **Live Vast prices, Brave quota, GPT-5.5 judge price, Qwen3.8-Flash-Next size — UNVERIFIED;** budgets carry 15–20% contingency and a day-1 price check.

### Key sources
- AA methodology: https://artificialanalysis.ai/methodology/intelligence-benchmarking · tau3-Banking board: https://artificialanalysis.ai/evaluations/tau3-banking · GDPval-AA: https://artificialanalysis.ai/evaluations/gdpval-aa
- tau2-bench: https://github.com/sierra-research/tau2-bench (CHANGELOG, `docs/cli-reference.md`, `src/tau2/knowledge/README.md`) · tau-Knowledge paper: https://arxiv.org/abs/2603.04370 · Sierra board: https://taubench.com/
- Stirrup: https://github.com/ArtificialAnalysis/Stirrup · GDPval dataset: https://huggingface.co/datasets/openai/gdpval · GDPval paper: https://arxiv.org/abs/2510.04374
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B · vLLM day-0: https://vllm.ai/blog/2026-08-12-qwen3.8 · Axolotl: https://github.com/axolotl-ai-cloud/axolotl · Unsloth Qwen3.5 fine-tune: https://unsloth.ai/docs/models/qwen3.5/fine-tune · Gemma 4: https://ai.google.dev/gemma/docs/core/model_card_4
- Recipes: APIGen-MT https://arxiv.org/abs/2504.03601 · AgenticQwen https://arxiv.org/abs/2604.21590 · AgentScaler https://arxiv.org/abs/2509.13311 · Amity MT-GRPO https://arxiv.org/abs/2604.02869 · Divergence-point DPO https://arxiv.org/abs/2606.23112 · Solar Open 2 / OfficeVerse https://arxiv.org/abs/2607.20062 · Kimi K2 https://arxiv.org/abs/2507.20534 · AgentTuning https://arxiv.org/abs/2310.12823 · Thinking Machines OPD https://thinkingmachines.ai/blog/on-policy-distillation/ · GAD https://arxiv.org/abs/2511.10643
- Data: https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1 · Toucan https://arxiv.org/abs/2510.01179 · Open-SWE-Traces https://arxiv.org/abs/2606.16038
- Infra: https://docs.vast.ai/cli/commands · https://docs.vast.ai/instances/rental-types · https://e2b.dev/pricing · https://openrouter.ai/docs/api-reference/parameters · https://github.com/volcengine/verl