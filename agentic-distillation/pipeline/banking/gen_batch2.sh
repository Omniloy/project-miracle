set -x; cd /tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/pipeline/banking
/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/.venv/bin/python gen_banking_tasks.py --n 40 --model z-ai/glm-5.3 --out synth/b2_glm.json --seed 303 --workers 6 --start-idx 3000 --tools-per-task 2,3
/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/.venv/bin/python validate_task.py synth/b2_glm.json synth/b2_glm.valid.json --report synth/b2_glm.report.json
python3 decontaminate.py /tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json synth/b2_glm.valid.json --report synth/b2_glm.decon.json
echo BATCH2_DONE
