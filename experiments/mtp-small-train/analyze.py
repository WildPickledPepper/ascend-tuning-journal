"""Validate private four-question receipts and write filtered training evidence.

Usage: python analyze.py PRIVATE_RUN OLD_PREFETCH_PRIVATE FILTERED_JSON
Never exports prompts, answers, addresses, process IDs, or model files.
"""
import ast
import json
from pathlib import Path
import re
import statistics
import sys

root, old, output = map(Path, sys.argv[1:4])
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

# Reuse the already-reviewed counter parser without running its profile analysis.
source = Path(__file__).parents[1] / 'dflash2-vllm-graph/analyze.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('metrics', 'acceptance')]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'))

baseline = read(old/'model-run/results/baseline-sglang-requests.json')
candidate = read(root/'results/finetuned-sglang-requests.json')
assert candidate['example_count'] == 4 and len(candidate['records']) == 4
def generation_request(record):
    request = dict(record['request'])
    request['extra_body'] = dict(request['extra_body'])
    request['extra_body'].pop('return_token_ids',None)
    return request
assert [generation_request(r) for r in baseline['records']] == [generation_request(r) for r in candidate['records']]
assert all(a['request']['extra_body'].get('return_token_ids') is True and
           'return_token_ids' not in b['request']['extra_body']
           for a,b in zip(baseline['records'],candidate['records']))
assert baseline['evaluator_source_sha256'] == candidate['evaluator_source_sha256']
train = read(root/'training-progress.json')
integrity = read(root/'candidate-integrity.json')
separation = read(root/'data-separation.json')
assert train['phase'] == 'complete' and train['optimizer_updates'] == 32
assert train['frozen_tables_exactly_unchanged'] and train['native_mtp_changed']
assert integrity['non_head_tensors_exactly_unchanged'] and not integrity['original_directory_written']
assert separation['normalized_overlap'] == 0
cleanup = read(root/'eval-state.json')
assert cleanup['phase'] == 'complete' and all(cleanup['cards_empty'].values())
audit, = read(root/'graph-audit-after.json')['results']
for k,v in {'num_speculative_tokens':7, 'draft_use_cuda_graph':True,
            'draft_enforce_eager':False, 'tp_size':1, 'dtype':'torch.bfloat16',
            'context_length':4096, 'max_num_seqs':1, 'prefix_caching':False,
            'mamba_ssm_cache_dtype':'bfloat16', 'gpu_memory_utilization':.97}.items():
    assert audit[k] == v, (k,audit[k])
assert any(w['is_draft_runnable'] and any(e['captured'] for e in w['entries']) for w in audit['wrappers'])
rates = [read(old/f'model-run/results/{n}-sglang-requests.json')['throughput_tok_s'] for n in ['baseline-r1','baseline-r2','baseline']]
questions = []
for i,(a,b) in enumerate(zip(baseline['records'],candidate['records']),1):
    ac,bc = a['response']['choices'][0],b['response']['choices'][0]
    questions.append({'question':i, 'baseline_tokens':a['response']['usage']['completion_tokens'],
        'trained_tokens':b['response']['usage']['completion_tokens'],
        'baseline_elapsed_s':a['elapsed_s'], 'trained_elapsed_s':b['elapsed_s'],
        'token_ids_equal':None,
        'final_content_equal':ac['message']['content'] == bc['message']['content'],
        'finish_reason':bc['finish_reason']})
assert all(r['response']['choices'][0].get('token_ids') is None for r in candidate['records'])
memory = re.findall(r'Process memory\(MB\):(\d+)',(root/'eval-card6-ready-private.txt').read_text())
result = {'model':'Qwen3.8-27B', 'hardware':'Ascend910B2', 'card_count':1, 'dtype':'BF16',
    'evaluation_examples':4, 'new_baseline_requests':0, 'new_profiler_requests':0,
    'generation_requests_equal':True,
    'response_metadata_difference':'Old baseline requested return_token_ids=True; candidate omitted this response-only flag. Token-ID equality cannot be checked; not a byte-identical HTTP payload.',
    'data_separation':separation, 'teacher':read(root/'teacher-progress.json'),
    'feature_verification':read(root/'feature-summary.json'),
    'training':train, 'candidate_integrity':integrity,
    'baseline_three_rates':rates, 'baseline_median_tok_s':statistics.median(rates),
    'baseline_accuracy':baseline['accuracy'],
    'trained_benchmark':{k:v for k,v in candidate.items() if k!='records'},
    'throughput_ratio':candidate['throughput_tok_s']/statistics.median(rates),
    'acceptance':{'baseline':acceptance(old/'model-run/baseline-metrics-before.txt',old/'model-run/baseline-metrics-after.txt'),
                  'trained':acceptance(root/'metrics-before.txt',root/'metrics-after.txt')},
    'questions':questions, 'graph_audit_verified':True,
    'ready_process_memory_mib':sum(map(int,memory)),
    'cleanup':{k:v for k,v in cleanup.items() if k!='server_pid'},
    'limits':['Four questions only; existing baseline was measured earlier, not a simultaneous A/B.',
              'Benchmark includes prefill and normal request overhead; no profiler during timing.',
              'Training uses one-step teacher-forced CE, not seven-step recursive training.',
              'Training CE is the online optimization average, not a fixed endpoint evaluation.',
              'Small pilot does not establish long-run correctness or generalization.']}
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['baseline_median_tok_s','trained_benchmark','throughput_ratio','acceptance','cleanup']},ensure_ascii=False,indent=2))
