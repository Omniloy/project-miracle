#!/usr/bin/env python3
"""Compare an evaluated model against the baseline: 97-task test (strict tau2 reward, AA config) and the synthetic
held-out dev tasks (relaxed verifier). Also reports per-task flips and a task-level bootstrap CI for the test delta.
Usage: analyze_eval.py BASE_TEST_results.json MODEL_TEST_results.json [--dev MODEL_DEV_relaxed.json --dev-ref dev_base_reference.json --holdout split.json]
"""
import argparse, json, random
def by_task(results):
    d=json.load(open(results)); out={}
    for s in d['simulations']:
        out.setdefault(s['task_id'],[]).append(1.0 if (s.get('reward_info') or {}).get('reward')==1.0 else 0.0)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('model'); ap.add_argument('--dev'); ap.add_argument('--dev-ref'); ap.add_argument('--holdout'); a=ap.parse_args()
    B,M=by_task(a.base),by_task(a.model); tasks=sorted(set(B)&set(M))
    pb=sum(sum(B[t])/len(B[t]) for t in tasks)/len(tasks); pm=sum(sum(M[t])/len(M[t]) for t in tasks)/len(tasks)
    print(f"TEST (97 real tasks, strict, AA config): base {pb:.1%}  model {pm:.1%}  delta {100*(pm-pb):+.1f} pts over {len(tasks)} tasks")
    flips_up=[t for t in tasks if max(M[t])>max(B[t])]; flips_dn=[t for t in tasks if max(B[t])>max(M[t])]
    print(f"  tasks gained: {len(flips_up)} {flips_up[:12]}\n  tasks lost:   {len(flips_dn)} {flips_dn[:12]}")
    rng=random.Random(0); deltas=[]
    for _ in range(2000):
        smp=[rng.choice(tasks) for _ in tasks]; deltas.append(sum(sum(M[t])/len(M[t])-sum(B[t])/len(B[t]) for t in smp)/len(smp))
    deltas.sort(); print(f"  task-bootstrap 90% CI for delta: [{100*deltas[100]:+.1f}, {100*deltas[1899]:+.1f}] pts")
    if a.dev and a.holdout:
        hold=set(json.load(open(a.holdout))['holdout_tasks']); rep=json.load(open(a.dev))
        dm=[r for r in rep if r['task_id'] in hold]; ok=sum(1 for r in dm if r.get('relaxed')==1.0)
        ref=json.load(open(a.dev_ref)) if a.dev_ref else None
        print(f"DEV (held-out synthetic, relaxed): model {ok}/{len(dm)} = {ok/max(1,len(dm)):.0%}" + (f" | base reference {ref['base_relaxed_pass']}/{ref['base_n']} = {ref['base_relaxed_pass']/max(1,ref['base_n']):.0%}" if ref else ''))
if __name__=='__main__': main()
