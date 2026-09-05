import json, html, collections
S="/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad"
p=json.load(open(f"{S}/research/phone_agent_plan.json"))
E=html.escape
def spoken(s): return E(s)

# ---------- taxonomy ----------
groups=collections.OrderedDict()
for x in p["pattern_taxonomy"]: groups.setdefault(x["group"],[]).append(x)
order=["opening/verification","identifier capture & validation","before-tool","during-wait","result narration","disambiguation","empty/too-many/partial results","errors & denials","confirmations & irreversible actions","hold/transfer","repair & interruptions","closing","anti-patterns"]
titles={"opening/verification":"Opening and verification","identifier capture & validation":"Identifier capture and validation","before-tool":"Before a tool call","during-wait":"While waiting","result narration":"Narrating results","disambiguation":"Disambiguation","empty/too-many/partial results":"Empty, too many, partial","errors & denials":"Errors and denials","confirmations & irreversible actions":"Confirmations and irreversible actions","hold/transfer":"Hold and transfer","repair & interruptions":"Repair and interruptions","closing":"Closing","anti-patterns":"Anti-patterns (reject)"}
tax=[]
for g in order:
    items=groups.get(g,[])
    cls=" anti" if g=="anti-patterns" else ""
    rows="".join(f'<details class="pat{cls}"><summary><span class="pname">{E(x["pattern"])}</span><span class="ptrig">{E(x["trigger"])}</span></summary><div class="pbody"><p class="ex">{E(x["example"])}</p><p class="chk"><span class="lbl">Check</span>{E(x["check"])}</p></div></details>' for x in items)
    tax.append(f'<section class="tgroup{cls}"><h3>{E(titles[g])} <span class="count">{len(items)}</span></h3>{rows}</section>')
tax_html="".join(tax); n_pat=len(p["pattern_taxonomy"])

# ---------- tables ----------
def table(headers, rows, cls=""):
    th="".join(f"<th>{E(h)}</th>" for h in headers)
    tr="".join("<tr>"+"".join(f"<td>{c}</td>" for c in r)+"</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'

data_rows=[[f'<b>{E(d["stage"])}</b>', E(d["what"]), E(d["size"]), E(d["cost_estimate"]), E(d["gate"])] for d in p["data_plan"]]
eval_rows=[[f'<b>{E(d["name"])}</b>', E(d["what"]), E(d["baseline"])] for d in p["eval_plan"]]
risk_rows=[[E(d["risk"]), E(d["mitigation"])] for d in p["risks"]]
budget_rows=[[E(d["item"]), E(d["estimate"])] for d in p["budget"]]
m=p["models"]
model_rows=[["Primary trace generator", E(m["primary_generator"])],["Bulk / diversity generator", E(m["bulk_generator"])],["Judge panel", E(m["judge"])],["User simulator", E(m["user_simulator"])]]
timeline="".join(f'<li><span class="wk">{E(t["week"])}</span><p>{E(t["milestones"])}</p></li>' for t in p["timeline"])
rep=[x for x in p["replicate_vs_improve"] if x.startswith("REPLICATE")]; imp=[x for x in p["replicate_vs_improve"] if x.startswith("IMPROVE")]; dont=[x for x in p["replicate_vs_improve"] if x.startswith("DO NOT")]
li=lambda xs: "".join(f"<li>{E(x.split(':',1)[1].strip() if ':' in x else x)}</li>" for x in xs)
train_li="".join(f"<li>{E(x)}</li>" for x in p["training_plan"])
serve_li="".join(f"<li>{E(x)}</li>" for x in p["serving_plan"])
diff_li="".join(f"<li>{E(x)}</li>" for x in p["differentiators"])
unv_li="".join(f"<li>{E(x)}</li>" for x in p["refuted_or_unverified"])
src_li="".join(f'<li><a href="{E(u)}">{E(u.replace("https://",""))}</a></li>' for u in p["sources"])

summ=p["phonellm_summary"]
a=summ.find("Evaluated on"); b=summ.find("Serving:"); c=summ.find("Key facts for us:")
desc=summ[:a].strip(); evalp=summ[a:b].strip(); servp=summ[b:c].strip(); facts=summ[c+len("Key facts for us:"):].strip()
import re
fact_items=[f.strip(" ;.") for f in re.split(r"\(\d\)", facts) if f.strip(" ;.")]
board=[("Gemini 3.6 Flash","78.6"),("GPT-5.6 Terra","72.4"),("PhoneLLM Alpha 1","72.3"),("Qwen3.8-27B, untuned","70.0"),("Claude Sonnet 5","68.9"),("Nemotron 3 Nano base","28.6")]
board_rows="".join(f"<tr><td>{E(n)}</td><td style=\"font-family:var(--mono);text-align:right\">{v}</td></tr>" for n,v in board)
phonellm_html=f"""<p>{E(desc)}</p>
<div class="two"><div><p>{E(evalp)}</p><p>{E(servp)}</p></div>
<div><div class="tw"><table><thead><tr><th>PhoneBench Alpha 1</th><th style="text-align:right">score</th></tr></thead><tbody>{board_rows}</tbody></table></div><p class="small">Fifteen models, human-calibrated judge panel, seven axes. The harness is not public, so these numbers are only comparable within Pipecat's board.</p></div></div>
<h3>What this means for us</h3><ul class="plain">{"".join(f"<li>{E(x)}</li>" for x in fact_items)}</ul>"""
spec=p.get("identifier_spec",[])
spec_rows="".join(f"<tr><td><b>{E(x['identifier'])}</b></td><td>{E(x['format'])}</td><td>{E(x['checksum'])}</td><td>{E(x['spoken_forms_es'])}</td><td>{E(x['asr_confusions'])}</td><td>{E(x['readback_policy'])}</td></tr>" for x in spec)
probes=p.get("identifier_eval_probes",[])
probe_rows="".join(f"<tr><td>{E(x['language'].upper())}</td><td><i>{E(x['utterance'])}</i></td><td><code>{E(x['expected_normalized'])}</code></td><td>{E(x['expected_behaviour'])}</td></tr>" for x in probes[:12])
id_html=f"""
<h2>Identifier capture and validation</h2>
<p>Added after review: callers hand over DNI, NIE, passport and phone numbers in dozens of spoken shapes, with dots, in pairs or hundreds, "doble cinco", "efe de Francia", "cero" versus "o", and telephony ASR mishears them in predictable ways. This family fixes what the agent must do at each step: name the document type, normalize the spoken form to the canonical string, validate the checksum where one exists (DNI and NIE mod 23, CIF mod 10, IBAN mod 97, card Luhn), read back in chunks, re-ask exactly one failing segment on a bounded budget, and never call a write tool with an unvalidated value. The {len([x for x in p["pattern_taxonomy"] if x["group"]=="identifier capture & validation"])} patterns sit in the taxonomy below; this is the reference table they lean on.</p>
<div class="note"><b>Open design decision.</b> The research recommends that a passing checksum replaces the confirmation turn for lookups (Google conversation-design practice: explicit confirmation only for high-cost steps), while the phase-1 authoring brief requires a chunked read-back of every identifier before any write. Phase 1 keeps read-back always for consistency; phase 2 should make it per identifier: implicit for checksummable values used only in lookups, explicit for anything that feeds a write tool or has no checksum.</div>
<div class="tw"><table><thead><tr><th>Identifier</th><th>Format</th><th>Checksum</th><th>Spoken forms (ES)</th><th>ASR confusions</th><th>Read-back policy</th></tr></thead><tbody>{spec_rows}</tbody></table></div>
<h3>Scripted probes (first 12 of {len(probes)})</h3>
<p class="small">Each probe is a caller utterance with the canonical value the agent must derive and the behaviour a correct agent shows. They double as unit tests for the validators and as eval items.</p>
<div class="tw"><table><thead><tr><th>Lang</th><th>Caller says</th><th>Canonical</th><th>Expected behaviour</th></tr></thead><tbody>{probe_rows}</tbody></table></div>
<p class="small">Validator pseudocode, the ASR-corruption spec and the full probe list are in <code>plans/switchboard_plan.json</code>; working implementations are in <code>pipeline/switchboard/validators.py</code> and <code>spoken_digits.py</code>, exercised by the phase-1 generation run.</p>
"""
page=f"""<title>Switchboard Plan</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#EEF2EF;--surface:#F7F9F7;--ink:#17221F;--muted:#5D6B66;--rule:#C9D3CE;--accent:#1E7A5A;--accent-ink:#0F4D38;--amber:#B7791F;--amber-bg:#F6EBD3;--red:#A63D2F;--red-bg:#F3E0DC;--green-bg:#DCEDE4;--code-bg:#E4EAE6;
--display:"Archivo Narrow","Arial Narrow",Arial,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0F1614;--surface:#16201D;--ink:#E6EDE9;--muted:#97A6A0;--rule:#2B3733;--accent:#4FBF93;--accent-ink:#8FE0BE;--amber:#E0A83A;--amber-bg:#2A2416;--red:#E07A6A;--red-bg:#2C1A17;--green-bg:#17302A;--code-bg:#1D2926}}}}
:root[data-theme="dark"]{{--bg:#0F1614;--surface:#16201D;--ink:#E6EDE9;--muted:#97A6A0;--rule:#2B3733;--accent:#4FBF93;--accent-ink:#8FE0BE;--amber:#E0A83A;--amber-bg:#2A2416;--red:#E07A6A;--red-bg:#2C1A17;--green-bg:#17302A;--code-bg:#1D2926}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15.5px;line-height:1.55;margin:0}}
main{{max-width:1080px;margin:0 auto;padding:40px 28px 80px}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;letter-spacing:.005em;margin:0}}
h1{{font-size:44px;font-weight:700;line-height:1.02}}
h2{{font-size:26px;font-weight:600;margin:64px 0 14px;padding-top:18px;border-top:2px solid var(--ink)}}
h3{{font-size:18px;font-weight:600;margin:26px 0 8px}}
p,li{{max-width:70ch}}
p{{margin:0 0 12px}}
a{{color:var(--accent-ink)}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
header .lede{{font-size:18px;max-width:64ch;margin-top:16px}}
.strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);margin:28px 0 8px}}
.strip div{{background:var(--surface);padding:12px 14px}}
.strip .k{{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.strip .v{{font-family:var(--display);font-size:22px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}}
.strip .s{{font-size:12.5px;color:var(--muted)}}
.note{{border-left:3px solid var(--amber);background:var(--amber-bg);padding:10px 14px;margin:14px 0;max-width:78ch}}
.decision{{border-left:3px solid var(--accent);background:var(--green-bg);padding:10px 14px;margin:14px 0;max-width:78ch}}
figure{{margin:22px 0}}
figure svg{{max-width:100%;height:auto;display:block;color:var(--ink);font-family:var(--body)}}
figcaption{{font-size:13px;color:var(--muted);margin-top:8px;max-width:78ch}}
.tw{{overflow-x:auto;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th{{text-align:left;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--ink)}}
td{{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:top}}
td b{{font-family:var(--display);font-size:15px;font-weight:600}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media (max-width:760px){{.two{{grid-template-columns:1fr}}}}
.cols h3{{margin-top:8px}}
ul.plain{{padding-left:18px}} ul.plain li{{margin:6px 0}}
ol.steps{{padding-left:0;list-style:none;counter-reset:s}}
ol.steps li{{position:relative;padding:10px 0 10px 40px;border-left:1px solid var(--rule);margin-left:12px}}
ol.steps li::before{{counter-increment:s;content:counter(s);position:absolute;left:-13px;top:12px;width:24px;height:24px;border-radius:50%;background:var(--accent);color:#fff;font-family:var(--mono);font-size:12px;display:flex;align-items:center;justify-content:center}}
ol.steps li .wk{{font-family:var(--mono);font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
ol.steps li p{{margin:2px 0 0}}
.tgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px 28px;margin-top:10px}}
.tgroup h3{{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ink);padding-bottom:4px}}
.tgroup .count{{font-family:var(--mono);font-size:12px;color:var(--muted)}}
details.pat{{border-bottom:1px solid var(--rule)}}
details.pat summary{{cursor:pointer;list-style:none;padding:8px 0;display:grid;grid-template-columns:1fr;gap:2px}}
details.pat summary::-webkit-details-marker{{display:none}}
.pname{{font-weight:600;font-size:14px}} .pname::before{{content:"+ ";color:var(--accent);font-family:var(--mono)}}
details[open] .pname::before{{content:"– "}}
.ptrig{{font-size:12.5px;color:var(--muted)}}
.pbody{{padding:0 0 10px 14px;border-left:2px solid var(--accent);margin:0 0 8px 2px}}
.ex{{font-style:italic;margin:4px 0 6px;font-size:14px}}
.ex::before{{content:"\\201C"}} .ex::after{{content:"\\201D"}}
.chk{{font-size:13px;color:var(--muted);margin:0}}
.lbl{{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent-ink);margin-right:8px}}
.anti .pname::before{{color:var(--red)}} .anti .pbody{{border-color:var(--red)}} .anti .ex{{font-style:normal;font-family:var(--mono);font-size:12.5px}} .anti .ex::before,.anti .ex::after{{content:""}}
code{{font-family:var(--mono);font-size:12.5px;background:var(--code-bg);padding:1px 5px;border-radius:3px}}
.small{{font-size:13px;color:var(--muted)}}
ul.src{{font-family:var(--mono);font-size:11.5px;columns:2;column-gap:28px;padding-left:16px}} ul.src li{{margin:2px 0;overflow-wrap:anywhere}}
@media (max-width:760px){{ul.src{{columns:1}} h1{{font-size:34px}}}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
</style>
<main>
<header>
<div class="eyebrow">Plan · phase-1 data generation in progress · updated 2026-09-05 · Omniloy agentic-distillation</div>
<h1>Switchboard: a no-thinking phone agent on Qwen3.8-27B</h1>
<p class="lede">Distil the way a good human agent runs a call, acknowledging before a lookup, narrating results in plain speech, handling too-many, none, and near-miss results, confirming before anything irreversible, into Qwen3.8-27B served without thinking, then compress it to NVFP4 with the multi-token-prediction head intact. Reference product: Pipecat's PhoneLLM Alpha 1. Everything below is a plan; nothing has been launched.</p>
<div class="strip">
<div><div class="k">Reference score</div><div class="v">72.3</div><div class="s">PhoneLLM Alpha 1 on PhoneBench</div></div>
<div><div class="k">Our base, untuned</div><div class="v">70.0</div><div class="s">Qwen3.8-27B on the same board</div></div>
<div><div class="k">Patterns specified</div><div class="v">{n_pat}</div><div class="s">13 groups, each with a check</div></div>
<div><div class="k">Lean budget</div><div class="v">~$1.6k</div><div class="s">GPU ~$200 + API ~$1.35k</div></div>
<div><div class="k">Timeline</div><div class="v">8 wk</div><div class="s">Sep 7 to Oct 30, gated</div></div>
</div>
</header>

<h2>What PhoneLLM is, and what is actually open</h2>
{phonellm_html}
<div class="two cols">
<div><h3>Replicate</h3><ul class="plain">{li(rep)}</ul></div>
<div><h3>Improve</h3><ul class="plain">{li(imp)}</ul><h3>Do not chase</h3><ul class="plain">{li(dont)}</ul></div>
</div>

<h2>The NVFP4 decision</h2>
<div class="decision"><b>Decision.</b> Do not fine-tune a pre-quantized community NVFP4 checkpoint. Train in bf16 with a larger adapter than v0, merge, then quantize selectively with the MTP head and the delta-sensitive layers protected. Keep two fallbacks alive from day one.</div>
<figure>
<svg viewBox="0 0 1040 400" role="img" aria-label="Three candidate paths from the bf16 base to a served phone agent, with the gates that decide between them">
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<g font-size="12">
<rect x="20" y="150" width="150" height="70" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="95" y="178" text-anchor="middle" font-weight="600">Qwen3.8-27B bf16</text><text x="95" y="196" text-anchor="middle" fill="var(--muted)">Apache-2.0, MTP head</text>
<!-- P1 -->
<text x="220" y="52" font-family="var(--mono)" font-size="11" fill="var(--accent)">P1 · PRIMARY</text>
<rect x="220" y="60" width="170" height="64" rx="4" fill="none" stroke="var(--accent)" stroke-width="2"/>
<text x="305" y="85" text-anchor="middle" font-weight="600">LoRA r=128, all linear</text><text x="305" y="103" text-anchor="middle" fill="var(--muted)">LR 1e-4, 2 epochs, 30M tok</text>
<rect x="430" y="60" width="150" height="64" rx="4" fill="none" stroke="var(--accent)" stroke-width="2"/>
<text x="505" y="85" text-anchor="middle" font-weight="600">Merge to bf16</text><text x="505" y="103" text-anchor="middle" fill="var(--muted)">fp32 math, MTP kept</text>
<rect x="620" y="60" width="190" height="64" rx="4" fill="none" stroke="var(--accent)" stroke-width="2"/>
<text x="715" y="80" text-anchor="middle" font-weight="600">Selective NVFP4 PTQ</text><text x="715" y="96" text-anchor="middle" fill="var(--muted)">MLP W4A4 g16 · attn/GDN FP8</text><text x="715" y="112" text-anchor="middle" fill="var(--muted)">conv1d, lm_head, mtp.* bf16</text>
<rect x="850" y="60" width="170" height="64" rx="4" fill="var(--green-bg)" stroke="var(--accent)" stroke-width="2"/>
<text x="935" y="85" text-anchor="middle" font-weight="600">Serve NVFP4 + MTP k=2</text><text x="935" y="103" text-anchor="middle" fill="var(--muted)">vLLM 0.28, one RTX PRO 6000</text>
<line x1="170" y1="170" x2="220" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="390" y1="92" x2="430" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="580" y1="92" x2="620" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="810" y1="92" x2="850" y2="92" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="715" y="146" text-anchor="middle" font-size="11" fill="var(--amber)">gate: within 0.5 pt of bf16 merged · tau3 within 1.0 · MTP acceptance ≥ 0.75 · KL ≤ 1.5× base</text>
<!-- P2 -->
<text x="220" y="212" font-family="var(--mono)" font-size="11" fill="var(--muted)">P2 · IF THE DELTA DOES NOT SURVIVE PTQ</text>
<rect x="220" y="220" width="170" height="64" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="305" y="245" text-anchor="middle" font-weight="600">Quantize base first</text><text x="305" y="263" text-anchor="middle" fill="var(--muted)">ModelOpt nvfp4 --compress</text>
<rect x="430" y="220" width="150" height="64" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="505" y="245" text-anchor="middle" font-weight="600">QLoRA r=32–64</text><text x="505" y="263" text-anchor="middle" fill="var(--muted)">trained on FP4 activations</text>
<rect x="620" y="220" width="190" height="64" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
<text x="715" y="245" text-anchor="middle" font-weight="600">Serve FP4 base + LoRA</text><text x="715" y="263" text-anchor="middle" fill="var(--muted)">LoRA kernels cost ~40% decode</text>
<line x1="170" y1="190" x2="220" y2="252" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="390" y1="252" x2="430" y2="252" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="580" y1="252" x2="620" y2="252" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="715" y="306" text-anchor="middle" font-size="11" fill="var(--amber)">pre-test: temperature-0 determinism A/B (vLLM issue #50059 on SM120)</text>
<!-- P3 -->
<text x="220" y="342" font-family="var(--mono)" font-size="11" fill="var(--muted)">P3 · PROVEN TODAY</text>
<rect x="220" y="350" width="360" height="40" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="400" y="375" text-anchor="middle" font-weight="600">bf16 base + unmerged LoRA + MTP k=2 · 215–323 tok/s at 8–16 streams</text>
<line x1="170" y1="205" x2="220" y2="370" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="640" y="375" font-size="11" fill="var(--muted)">a legitimate release, not a failure; NVFP4 ships later</text>
</g></svg>
<figcaption>Three paths from the same base. P1 is the plan; P2 trains the adapter against quantized activations if the merged delta does not survive four-bit rounding; P3 is the serving path already measured this week. The amber lines are the gates that pick between them.</figcaption>
</figure>
<p class="small">Why not train on a quantized checkpoint directly: no official Qwen or NVIDIA NVFP4 of this model exists, only community ones; PhoneLLM itself was trained in bf16 and quantized afterwards; full-model quantization-aware training of a 27B needs several 80 GB GPUs; and our v0 adapter's delta (about 0.3% of weight magnitude) already fell below FP8's precision, so a four-bit grid with 25 to 50% relative steps would erase a small delta entirely. That last point is inference from the format, not a measurement; the KL and eval gates are the actual test.</p>

<h2>The data factory</h2>
<figure>
<svg viewBox="0 0 1040 330" role="img" aria-label="Data generation pipeline from taxonomy strata through mock environments, teacher and user simulator, programmatic filters and a judge, into training, calibration and evaluation splits">
<g font-size="12">
<rect x="20" y="30" width="180" height="96" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="110" y="52" text-anchor="middle" font-weight="600">Strata grid</text>
<text x="110" y="70" text-anchor="middle" fill="var(--muted)">{n_pat} patterns × 8 verticals</text><text x="110" y="86" text-anchor="middle" fill="var(--muted)">× narrate / silent</text><text x="110" y="102" text-anchor="middle" fill="var(--muted)">× EN / ES = ~1,600 strata</text>
<rect x="240" y="30" width="180" height="96" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="330" y="52" text-anchor="middle" font-weight="600">Seeded mock tools</text>
<text x="330" y="70" text-anchor="middle" fill="var(--muted)">deterministic DBs that force</text><text x="330" y="86" text-anchor="middle" fill="var(--muted)">the pattern: 47 John Smiths,</text><text x="330" y="102" text-anchor="middle" fill="var(--muted)">a John Smyth, a timeout on call 2</text>
<rect x="460" y="10" width="180" height="64" rx="4" fill="var(--green-bg)" stroke="var(--accent)" stroke-width="2"/>
<text x="550" y="34" text-anchor="middle" font-weight="600">Teacher: Qwen3.8-Max</text><text x="550" y="52" text-anchor="middle" fill="var(--muted)">thinks, then reasoning stripped</text>
<rect x="460" y="90" width="180" height="64" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="550" y="114" text-anchor="middle" font-weight="600">Caller: GLM-5.3-Flash</text><text x="550" y="132" text-anchor="middle" fill="var(--muted)">persona card + ASR noise script</text>
<rect x="680" y="30" width="160" height="96" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="760" y="52" text-anchor="middle" font-weight="600">Programmatic filters</text>
<text x="760" y="70" text-anchor="middle" fill="var(--muted)">say/do and do/say</text><text x="760" y="86" text-anchor="middle" fill="var(--muted)">raw-data regex, slot tracker</text><text x="760" y="102" text-anchor="middle" fill="var(--muted)">word caps, transfer rate</text>
<rect x="880" y="30" width="140" height="96" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="950" y="52" text-anchor="middle" font-weight="600">Judge: Sonnet 5</text>
<text x="950" y="70" text-anchor="middle" fill="var(--muted)">seven axes, ≥ 4/5 each</text><text x="950" y="86" text-anchor="middle" fill="var(--muted)">different family</text><text x="950" y="102" text-anchor="middle" fill="var(--muted)">from the teacher</text>
<line x1="200" y1="78" x2="240" y2="78" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="420" y1="60" x2="460" y2="42" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="420" y1="96" x2="460" y2="118" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<path d="M550 74 L550 90" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<path d="M600 90 L600 74" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="618" y="86" font-size="10.5" fill="var(--muted)">turn by turn</text>
<line x1="640" y1="78" x2="680" y2="78" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="660" y="70" text-anchor="middle" font-size="10.5" fill="var(--muted)">transcript</text>
<line x1="840" y1="78" x2="880" y2="78" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<!-- outputs -->
<line x1="950" y1="126" x2="950" y2="190" stroke="currentColor" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="962" y="162" font-size="10.5" fill="var(--muted)">accepted ≈ 70%</text>
<rect x="60" y="190" width="960" height="1" fill="currentColor"/>
<rect x="60" y="210" width="330" height="64" rx="4" fill="var(--green-bg)" stroke="var(--accent)" stroke-width="2"/>
<text x="225" y="236" text-anchor="middle" font-weight="600">Train · ~3,500 conversations</text><text x="225" y="254" text-anchor="middle" fill="var(--muted)">~42k assistant turns, ~25M tokens, + 15% tau2 replay</text>
<rect x="410" y="210" width="190" height="64" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="505" y="236" text-anchor="middle" font-weight="600">Calibration · 1,024</text><text x="505" y="254" text-anchor="middle" fill="var(--muted)">for NVFP4 PTQ, eval-disjoint</text>
<rect x="620" y="210" width="200" height="64" rx="4" fill="none" stroke="var(--amber)" stroke-width="2"/>
<text x="720" y="236" text-anchor="middle" font-weight="600">PhonePatterns-Eval · 300</text><text x="720" y="254" text-anchor="middle" fill="var(--muted)">held out before training</text>
<rect x="840" y="210" width="180" height="64" rx="4" fill="none" stroke="var(--amber)" stroke-width="2"/>
<text x="930" y="236" text-anchor="middle" font-weight="600">Human labels · 200 turns</text><text x="930" y="254" text-anchor="middle" fill="var(--muted)">judge calibration, ρ ≥ 0.7</text>
<text x="60" y="305" font-size="11" fill="var(--muted)">Quotas: 60% EN / 40% ES · 50/50 narrate vs silent · 25% calls ≥ 20 turns with a mid-call slot change · 15% tool errors · 20% ASR-corrupted user turns · 10% transfer-warranted vs 10% transfer-requested-but-in-scope · no stratum &gt; 3% · MinHash near-dup &lt; 5%</text>
</g></svg>
<figcaption>The environment owns every tool result, so the teacher can never invent one. The teacher reasons while generating and its reasoning is discarded, which is how a no-thinking student learns a thinking model's decisions. Programmatic checks run before any LLM judge, and the judge family differs from the generator to avoid self-preference.</figcaption>
</figure>
{table(["Stage","What","Size","Cost","Gate"], data_rows)}

{id_html}
<h2>The pattern taxonomy</h2>
<p>Every pattern names its trigger in terms of the tool-result shape or the conversational state, gives one utterance in the register we want, and defines a check that can score it. Open a pattern to see its example and check. Anti-patterns are rejection rules that run programmatically before any judge.</p>
<div class="note"><b>One design choice worth flagging.</b> Pipecat's reference prompt forbids narrating tool use ("no let me check, no one moment"), while the brief for this project wants a human-like acknowledgement before a lookup. The plan trains both as prompt-conditional styles, so the same model obeys whichever the system prompt asks for. Half the dataset uses each.</div>
<div class="tgrid">{tax_html}</div>

<h2>Models for traces, judging and simulation</h2>
{table(["Role","Choice and reasoning"], model_rows)}
<p class="small">{E(m["rationale"])} {E(m.get("license_note",""))}</p>

<h2>Training recipe</h2>
<ul class="plain">{train_li}</ul>

<h2>Serving, reusing what this week established</h2>
<ul class="plain">{serve_li}</ul>

<h2>Evaluation</h2>
{table(["Eval","What it measures","Baselines and gate"], eval_rows)}

<h2>What would make this publishable</h2>
<ul class="plain">{diff_li}</ul>

<h2>Timeline with gates</h2>
<ol class="steps">{timeline}</ol>

<h2>Budget</h2>
{table(["Item","Estimate"], budget_rows)}

<h2>Risks and mitigations</h2>
{table(["Risk","Mitigation"], risk_rows)}

<h2>What the skeptic pass refuted or could not verify</h2>
<p class="small">Each research report was re-checked against primary sources by an independent reviewer. These items were downgraded or removed from the claims above and must be settled by the week-1 smoke tests or a human before they inform a decision.</p>
<ul class="plain small">{unv_li}</ul>

<h2>Sources</h2>
<ul class="src">{src_li}</ul>
</main>
"""
open(f"{S}/artifacts/switchboard_plan.html","w").write(page)
print("written", len(page), "chars;", n_pat, "patterns")
