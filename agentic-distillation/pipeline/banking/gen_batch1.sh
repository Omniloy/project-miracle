set -x
cd /tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/pipeline/banking
/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/.venv/bin/python gen_banking_tasks.py --n 40 --model z-ai/glm-5.3 --out synth/b1_glm.json --seed 101 --workers 6 --start-idx 1000
/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/.venv/bin/python gen_banking_tasks.py --n 20 --model qwen/qwen3.8-max --out synth/b1_max.json --seed 202 --workers 5 --start-idx 2000
/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/.venv/bin/python validate_task.py synth/b1_glm.json synth/b1_glm.valid.json --report synth/b1_glm.report.json
/tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/.venv/bin/python validate_task.py synth/b1_max.json synth/b1_max.valid.json --report synth/b1_max.report.json
python3 decontaminate.py /tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json synth/b1_glm.valid.json --report synth/b1_glm.decon.json
python3 decontaminate.py /tmp/claude-0/-home-user-project-miracle/43fff345-b582-59b0-a287-d38e1b7005e8/scratchpad/harness/tau2-bench/data/tau2/domains/banking_knowledge/tasks.json synth/b1_max.valid.json --report synth/b1_max.decon.json
echo BATCH1_DONE
