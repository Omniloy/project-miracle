# Inference speed for the tau2 evals — Qwen3.8-27B student on one RTX PRO 6000 (2026-09-04)

Scope: make the 97-task tau2 banking eval (and the 5-trial final) 3-5x faster than the current ~3 h/trial without changing what is measured.
Method: literature/code research (vLLM 0.28.0 wheel source, vLLM issues/PRs, Qwen model card, HF discussions), a skeptical re-verification pass, and a
46-min A/B benchmark on a spare RTX PRO 6000 WS (Vast 49856926, $1.3 total; results `runs/bench_results.jsonl`, logs `runs/bench_status/`).
Benchmark load: 12 real training turns from `data_v1/train_turns.jsonl` (mean prompt 10.8k tokens, shared system prefix, tools attached), driven at
concurrency 8 and 16 for 100 s each after a warm-up pass (prefix-cache hit rate 0.97, i.e. the tau2 "warm turn" regime); sampling identical to the
eval (T=1.0, top_p 0.95, top_k 20), max_tokens 1200, default chat template (thinking ON).

## 0. Lead's caveat on the recommendation (added after review, 2026-09-04 13:20 UTC)

The headline config E (merged + online FP8 + MTP) is **not acceptable for evaluating an adapter**: adapter_v0's weight delta is tiny relative to the
base weights (mean|delta| 3.3e-5 vs mean|W| ~0.01, i.e. ~0.3 % relative). bf16 has ~0.4 % relative resolution, which is why the bf16 merge already loses
~20 % of the delta's L1 mass and rounds away ~47 % of its elements; FP8 e4m3 has ~6-12 % relative resolution, so an FP8 merge erases the adapter
almost entirely and config E measures (approximately) the base model in FP8. The benchmark's "sanity 0.97" only checked output well-formedness, not that
the adapter's effect survived. The unmerged LoRA path applies the low-rank product in bf16 activations and is therefore the *most faithful* way to serve
a small-delta adapter. Decision: evals keep `SERVE_MODE=lora`, `QUANT=none`; the speed levers to adopt are **concurrency 16** (measured 1.8x aggregate
on the LoRA path) and **MTP k=2 on top of LoRA** (vLLM V1 supports LoRA + speculative decoding since PR #21068; the draft runs base weights, the
target pass applies the LoRA), pending a follow-up bench with output samples that also resolves the unexplained degenerate-output result for config A.

## 0b. Follow-up benchmark on adapter-preserving configs (2026-09-04 13:26–14:03 UTC, Vast 49861771, ~$0.8, `runs/bench2_results.jsonl`)

Same client and prompts, output samples saved (`status_bench/samples_*.jsonl`). All LoRA-path configs produced normal turns this time
(sanity 0.94–1.00, 36–42 % tool calls): the first run's all-degenerate result for config A was a one-off on that box, not a property of the config.

| config (bf16 base + LoRA, unmerged) | conc | agg gen tok/s | per-stream tok/s | mean compl. tok | sanity | MTP acc. len |
|---|---|---|---|---|---|---|
| A  thinking on (current eval cfg)   | 8  | 142.6 | 21.4 | 292 | 0.94 | – |
| A  thinking on                      | 16 | 238.9 | 19.2 | 331 | 0.95 | – |
| AN thinking off                     | 8  | 150.5 | 19.5 | 125 | 1.00 | – |
| AN thinking off                     | 16 | 253.9 | 16.2 | 128 | 1.00 | – |
| **AM thinking on + MTP k=2**        | 8  | **214.7** | **31.3** | 336 | 0.94 | 2.30 |
| **AM thinking on + MTP k=2**        | 16 | **322.7** | 23.0 | 335 | 0.95 | 2.33 |
| AMN thinking off + MTP k=2          | 8  | 179.7 | 23.3 | 119 | 1.00 | 2.49 |
| AMN thinking off + MTP k=2          | 16 | 233.1 | 15.0 | 119 | 1.00 | 2.52 |

**Decision**: evals use `SERVE_MODE=lora QUANT=none SPEC=mtp CONC=16` — adapter fully preserved (no merge, no FP8), no crashes at either
concurrency, 2.3x the current eval's aggregate throughput and 1.5x per stream at 8. Thinking-off turns are ~2.4x shorter (125 vs 300 tokens)
at the same tool-call rate; the adapter was trained with thinking off, so `THINKING=off` is a legitimate second configuration to score, but
the headline comparison stays in thinking mode to match the 45.4 baseline.

## 1. Measured table (vLLM 0.28.0, sm_120, 96 GB)

| config | conc | status | agg gen tok/s | per-stream tok/s | warm TTFT s | mean compl. tok | sanity | tool-call rate | trunc | MTP acc. len (pos1/pos2) |
|---|---|---|---|---|---|---|---|---|---|---|
| A bf16 + LoRA r64 (**current eval cfg**) | 8 | ok | 180.4 | 22.6 | 0.62 | 1200 | 0.00 | 0.00 | 1.00 | - |
| A bf16 + LoRA | 16 | ok | 324.8 | 20.3 | 1.12 | 1200 | 0.00 | 0.00 | 1.00 | - |
| B merged bf16 | 8 | aborted @34 s (segfault in CUDA-graph replay) | ~185 (engine log 181-191) | 23.4 | 0.38 | 235 | 1.00 | 0.35 | 0.00 | - |
| C merged bf16 + MTP k=2 | 8 | ok | 249.0 | 34.6 | 0.52 | 348 | 0.93 | 0.36 | 0.07 | 2.34 (0.76/0.58) |
| C merged bf16 + MTP k=2 | 16 | **crashed** (illegal memory access) | - | - | - | - | - | - | - | - |
| D merged + online FP8 | 8 | ok | 325.6 | 40.7 | 0.34 | 1200 | 0.00 | 0.00 | 1.00 | - |
| D merged + online FP8 | 16 | ok | 556.8 | 34.8 | 0.62 | 1200 | 0.00 | 0.00 | 1.00 | - |
| **E merged + FP8 + MTP k=2** | 8 | ok | **393.0** | **48.8** | 0.36 | 308 | 0.97 | 0.38 | 0.02 | 2.36 (0.77/0.59) |
| **E merged + FP8 + MTP k=2** | 16 | ok | **505.1** | **35.0** | 0.51 | 312 | 0.97 | 0.39 | 0.03 | 2.32 (0.76/0.57) |
| F merged + ngram spec | 8 | ok | 153.5 | 20.7 | 0.22 | 395 | 0.83 | 0.32 | 0.17 | 1.46 (acc 0.15) |
| F merged + ngram spec | 16 | ok | 221.1 | 17.1 | 0.30 | 312 | 0.85 | 0.36 | 0.05 | 1.62 (acc 0.20) |
| C3 merged bf16 + MTP k=3 | 8 | **crashed** (device-side assert right after warm-up) | - | - | - | - | - | - | - | - |

Other facts: merge = 38 s on GPU, 400 modules, scale alpha/r = 2.0, 15 `mtp.*` tensors preserved, max|delta| 8.4e-4, mean|delta| 3.3e-5.
LoRA effect check on A: 3/3 greedy outputs differ student vs base (`OK_differ`), so the adapter is applied (guards vLLM #49354).
KV cache: bf16 weights -> 466k tokens (7.1x 65k contexts); FP8 weights -> 829k tokens (12.6x). GPU at 500-520 W of 600 W, 100 % util in all configs.
"sanity" = fraction of completions that finished with a parseable reply/tool call and no `<tool_call>` leakage; "kind match" vs the reference turn was 0.75-0.94 for B/C/E/F.
The running eval (49838262, LoRA path, read-only log) averages ~435 generated tokens/request, 3882 requests in 4.2 h (~15.4 req/min), 120 tok/s aggregate at 8 concurrent.

## 2. Root causes of the slow baseline (120 tok/s aggregate, 15 tok/s per stream)

1. **bf16 weight bandwidth, not the LoRA kernels.** A 27B dense model reads ~54 GB of weights per decode step; at ~1.8 TB/s that caps a
   stream at ~33 tok/s and vLLM reaches 22-23 tok/s. Merging the adapter (B) gives 23.4 vs 22.6 tok/s/stream -> the Punica/Triton LoRA path
   costs only ~3-4 % at 8 streams. The STATE.md hypothesis "default LoRA kernels are the problem" is refuted; FP8 weights (halved bytes/step) are what
   nearly doubles per-stream speed (40.7 tok/s).
2. **Small decode batch.** At 8 streams the GPU is far from compute-bound: going 8 -> 16 streams costs ~10 % per stream and gains 1.8x aggregate in
   every config. tau2 `--max-concurrency 8` leaves half the machine idle; the user simulator's latency (OpenRouter gpt-5.4-mini) further reduces the
   number of streams actually decoding at any moment.
3. **One token per step.** The checkpoint ships a 1-layer MTP head; with `num_speculative_tokens=2` the mean accepted length is 2.35 under T=1.0
   sampling on real 10k-token banking prompts (per-position 0.77/0.58), i.e. ~1.2-1.5x more tokens per step (C vs B: 34.6 vs 23.4 tok/s at c=8).
4. Long contexts in the eval (10-50k prefix, 20-40 turns) explain the gap between the bench's 180 tok/s and the eval's 120 tok/s for the same
   config: attention over long KV plus the per-turn prefill of the new user message; prefix caching already removes the shared-prefix cost (hit rate 0.97).

## 3. Recommended serving config for the evals (config E, concurrency 16)

vLLM 0.28.0 in the same venv, same image (`axolotlai/axolotl-cloud:main-latest`), merged weights, online FP8, native MTP k=2:

```
vllm serve /workspace/merged --served-model-name student --port 8000 \
  --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching \
  --quantization fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
tau2 run ... --max-concurrency 16
```
Startup: 120-170 s (online FP8 quantization + CUDA-graph capture). Expect `SpeculativeConfig(method='mtp', ...)` and periodic
`SpecDecoding metrics: Mean acceptance length: ~2.3` lines in `status_eval/vllm_tail.log`. Do NOT use k=3 (crashes on sm_120/0.28.0), do not run
merged bf16 (B/C) at 16 streams (segfault / illegal memory access), and ngram speculation is a net loss (acceptance 0.15-0.20).
`remote_eval.sh` now exposes this as `SERVE_MODE=merged QUANT=fp8 SPEC=mtp CONC=16` (defaults keep the old LoRA path; see section 6).

LoRA-merge procedure (embedded in `remote_eval.sh`, identical to the benchmark's `merge_lora.py`):
- PEFT-equivalent `W += (alpha/r) * B @ A` computed in fp32 on the GPU tensor-by-tensor over the ORIGINAL safetensors shards, written back in bf16 to
  `/workspace/merged` with the same shard layout, `model.safetensors.index.json`, config, tokenizer and chat template copied verbatim.
- Done this way rather than `transformers`+`peft merge_and_unload` because the transformers Qwen3_5 classes drop the `mtp.*` tensors on load and re-key
  the checkpoint; the manual merge keeps all 15 MTP tensors so MTP works on the merged model.
- Asserts: LoRA type, no DoRA/bias/rank_pattern, every adapter target exists in the base index, every indexed tensor (incl. `mtp.*`) exists after
  writing. 38 s of compute + ~56 GB disk -> launch with DISK_GB >= 180.

## 4. Expected eval wall-clock at the recommended config

Assumptions: ~435 generated tokens per agent call (observed), one call per turn, user-simulator latency ~5 s/turn (OpenRouter gpt-5.4-mini medium),
97 tasks x ~30 turns ~ 2900 calls per trial; setup (55 GB download, venv, merge, FP8 load) ~20-25 min. GPU cost at ~$1.3-1.6/h.

| run | current (bf16+LoRA, c=8): 15 tok/s -> ~35 s/turn | recommended (E, c=16): 35 tok/s -> ~18 s/turn, 2x streams |
|---|---|---|
| 97 tasks x 1 trial | ~3 h eval (+15 min setup), ~$4.5 | **~45-60 min eval (+20-25 min setup) ~ 1.2-1.5 h, ~$2** |
| 97 tasks x 5 trials | ~15 h, ~$22 | **~4-4.5 h eval + setup ~ 4.5-5 h, ~$7-8** |
| 10-task dev set x 2 trials (Flash user-sim) | ~30 min | ~10 min |

That is ~3-4x end-to-end (3.3x per-stream at c=8, 4.2x aggregate at c=16, minus the user-simulator share that speculation cannot touch). The final
tail of each run (last few long episodes at low concurrency) and OpenRouter latency spikes are the main sources of variance; OpenRouter spend
(~$95 for 5 trials) is unchanged.

## 5. Quality caveats (read before trusting a number from config E)

1. **Not lossless, twice.** (a) The bf16 merge discards ~47 % of the adapter's delta elements (mean|delta| 3.3e-5 vs bf16 step ~4e-5 at |W|~0.01) but
   keeps 72-92 % of the per-module L1 delta mass -> the merged model carries ~80 % of adapter_v0. (b) Online FP8 (e4m3, per-tensor dynamic scale)
   adds weight-quantization noise 1-2 orders of magnitude larger than the adapter delta per element; the adapter's effect survives only statistically
   (matmul sums). MTP speculative decoding itself is distribution-preserving by construction (rejection sampling), but vLLM documents that outputs
   differ from non-speculative runs under sampling. The exact-adapter alternative is `SERVE_MODE=lora QUANT=fp8 SPEC=none` (LoRA in bf16 on an FP8
   base): supported by vLLM, ~1.8x, but untested here and the non-MTP FP8 run (D) showed the degeneracy below.
2. **Comparability with the 45.4 baseline.** The baseline was served by OpenRouter (`openrouter/qwen/qwen3.8-27b`, provider precision unknown,
   most likely FP8), so an FP8 student is if anything closer to the baseline's precision than our bf16 LoRA path. Still, the eval numbers are only
   comparable within noise (1 trial = +-5 pts; 5 trials = +-2 pts). Rule: base-vs-student comparisons must use the SAME serving config; re-measure the
   base at config E once (~$2) before claiming a delta, and report the serving config with every number. Treat any E-vs-A difference on the 10-task
   dev set larger than ~5 pts as a merge/quantization regression.
3. **Unresolved bench anomaly (hard gate).** Configs A (bf16+LoRA, the running eval's exact flags) and D (FP8, no MTP) returned 1200-token,
   1-char-per-token degenerate reasoning on 100 % of bench requests, while B/C/E/F produced normal 235-350-token turns with tool calls. The live eval on
   the A config produces sane 435-token turns, so the trigger is something in the bench requests (most likely thinking ON via the default template
   versus the adapter trained with `enable_thinking: false`, or an interaction with the 1200-token cap) - the client did not save samples.
   Before using E for a reported number: run the 5-minute sample-dumping check (bench_client with output dump, thinking on AND off, configs A and E),
   and run the 10-task dev set on E and compare with the A-path dev score from the running eval.
4. Stability: E ran 89 + ~180 requests at c=8/16 with 0 errors; C (bf16+MTP) crashed at 16 streams and k=3 crashed outright -> keep k=2,
   `--max-num-seqs 32`, concurrency <= 16, and keep the `STEP vllm FAILED` / heartbeat markers so a mid-run engine death is visible on HF.
5. Warm-TTFT "rollback tax" of MTP on hybrid models (#53479): not visible at 10.8k prefix (0.36 vs 0.34 s); unmeasured at 40-50k prefix.

## 6. What was refuted or corrected during verification

- "MTP alone closes the 3-5x gap" - refuted: MTP on bf16 gives 1.35x at c=8 (C vs B) and 1.2x on FP8 (E vs D). The 3.3x comes from FP8 (1.8x) x MTP
  (1.2x) x concurrency 16 (1.3x aggregate at ~10 % per-stream cost).
- "LoRA kernels (Triton JIT, default configs) are the reason for 120 tok/s" (STATE.md) - refuted: merged bf16 decodes at the same per-stream speed.
- "k=5 measured 10.9 tok/s" (HF discussion #9) - the number came from a card with a broken power sensor (9.1 tok/s even without MTP); k=2 remains the
  sweet spot for other reasons (per-position acceptance 0.77/0.58 here; k=3 crashes on 0.28.0/sm_120).
- LoRA + speculative decoding "supported" - it works in 0.28.0 code (PR #21068) but the official 0.28.0 compatibility matrix marks SD x LoRA as
  unsupported; not benchmarked (the AM config was only a fallback for a failed merge). Not recommended.
- Attribution errors in the research pass: PR #46384 is the partial-prefix-hit PR, not the "skip EAGLE margin for MambaSpec" fix; "prefix caching + MTP
  broken through 0.27.x" overstated (fix #51113 was a 0.27.0 cherry-pick); the "~90 % -> ~50 % tool calling" quote could not be found in #43559.
- Prompt-throughput / warm-TTFT worries: with prefix caching at 0.97 hit rate the warm TTFT is 0.3-0.6 s in every config; not a bottleneck.
- ngram speculation as a cheap alternative - refuted (acceptance 0.15-0.20, 15-20 % slower than no speculation).
- Online FP8 on sm_120 "unusable" - partially: D produced degenerate output but E (same FP8 weights + MTP) was sane, so the FP8 weights are fine and
  the degeneracy correlates with something else (see caveat 3). Offline block-128 FP8 (llm-compressor) remains the vetted alternative if D-style
  degeneracy shows up in E under the eval's requests.

## 7. Decision
Adopt **E (merged bf16 -> online FP8 + MTP k=2) at tau2 concurrency 16** for all v1 evals and the 5-trial final, gated by: (1) the sample-dump
sanity check with thinking on/off, (2) the 10-task dev set on E within ~5 pts of the A-path dev score, (3) a one-off base-model run on the same
config for the comparison baseline. Fallback if the gate fails: `SERVE_MODE=lora QUANT=fp8 SPEC=none CONC=16` (exact adapter, ~1.8-3x), then the
old path with CONC=16 only (~1.8x aggregate, zero quality risk).
