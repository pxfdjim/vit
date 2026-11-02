"""Restartable multi-GPU sweep on GPUs with available memory."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/pengxf/jim/VLM4TS')
CODE = ROOT / 'experiments/vit4ts_final7'
OUT = ROOT / 'data/runs/20261016_vit4ts_final7'
PYTHON = '/home/pengxf/anaconda3/envs/vit/bin/python'
GPUS = (0, 1, 3, 4, 5, 7)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
started = time.time()
processes, attempts, finished, failed = {}, {i: 0 for i in range(len(GPUS))}, set(), set()


def launch(i):
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(GPUS[i]), 'CLIP_PRETRAINED': 'openai',
                'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
    with (OUT / f'worker_{i}.log').open('a') as stream:
        proc = subprocess.Popen(['nice', '-n', '10', PYTHON, '-u', str(CODE / 'worker.py'),
                                 '--worker', str(i), '--workers', str(len(GPUS))],
                                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
    processes[i] = proc
    attempts[i] += 1
    (OUT / f'worker_{i}.pid').write_text(str(proc.pid) + '\n')
    print('LAUNCHED', i, 'gpu', GPUS[i], 'pid', proc.pid, 'attempt', attempts[i], flush=True)


for worker in range(len(GPUS)):
    launch(worker)

while len(finished) + len(failed) < len(GPUS):
    time.sleep(10)
    for i, proc in list(processes.items()):
        code = proc.poll()
        if code is None:
            continue
        del processes[i]
        if code == 0:
            finished.add(i)
            print('WORKER_OK', i, flush=True)
        elif attempts[i] < 3:
            print('RESTART', i, code, flush=True)
            launch(i)
        else:
            failed.add(i)
            print('WORKER_FAILED', i, code, flush=True)
    counts = {ds: len(list((OUT / 'results' / ds).glob('*.json')))
              for ds in ('artificialWithAnomaly', 'realAWSCloudwatch', 'realAdExchange',
                         'realTraffic', 'realTweets', 'MSL', 'SMAP')}
    progress = {'completed_by_dataset': counts, 'completed_total': sum(counts.values()),
                'expected_total': 125, 'finished_workers': sorted(finished),
                'failed_workers': sorted(failed), 'updated_at': datetime.now(timezone.utc).isoformat()}
    (OUT / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')

summary_code = subprocess.call([PYTHON, str(CODE / 'summarize.py')], cwd=ROOT) if not failed else None
completion = {'completed_total': progress['completed_total'],
              'finished_workers': sorted(finished), 'failed_workers': sorted(failed),
              'summary_exit_code': summary_code, 'wall_seconds': round(time.time() - started, 2),
              'completed_at': datetime.now(timezone.utc).isoformat()}
(OUT / 'finished.json').write_text(json.dumps(completion, indent=2) + '\n')
print('FINISHED', json.dumps(completion), flush=True)
if failed or summary_code:
    sys.exit(1)
