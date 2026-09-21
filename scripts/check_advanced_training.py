"""Disposable integration check; does not evaluate or modify an experiment."""
from pathlib import Path
import ast,json,math,tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
notebook=json.loads((ROOT/'pacman_dqn.ipynb').read_text())
reference=json.loads((ROOT/'experiments/scaled_1069/pacman_dqn.ipynb').read_text())
for i,c in enumerate(notebook['cells']):
 if c['cell_type']=='code':
  source=''.join(c['source'])
  if not source.startswith('%'): ast.parse(source)
for i in [15,17,21,30]:
 assert notebook['cells'][i]['source']==reference['cells'][i]['source'],i
ns={}
for i in [2,8,10,12,15,17,19,21,23,34,36,38,47]:
 exec(compile(''.join(notebook['cells'][i]['source']),f'notebook-cell-{i}','exec'),ns)
assert ns['EVAL_SEEDS']==[101,202,303,404,505]
assert ns['EVAL_EXPLORATION']==.05 and ns['MAX_STEPS']==3000
assert torch.backends.mps.is_available(), 'Actual MPS device required for preflight'
ns.update(DEVICE=torch.device('mps'),MAX_STEPS=180,EPISODES=8,LEARNING_STARTS=64,
          WARMUP_STEPS=32,DEMO_EVERY=1000,TRAINING_TIME_LIMIT_SECONDS=180,
          RUN_DIR=Path(tempfile.mkdtemp(prefix='advanced_preflight_',dir=ROOT/'.execution')))
ns['N_ACTIONS']=9
ns['model']=ns['DQN'](9).to(ns['DEVICE'])
source=torch.load(ROOT/'pacman_runs/20260920_152728_434799/trained.pt',map_location='cpu',weights_only=True)
ns['model'].load_state_dict(source['model'])
ns['target']=ns['DQN'](9).to(ns['DEVICE']);ns['target'].load_state_dict(ns['model'].state_dict());ns['target'].eval()
ns['optimizer']=torch.optim.Adam(ns['model'].parameters(),lr=ns['LEARNING_RATE'])
ns['replay']=ns['PrioritizedNStepReplay'](500,n_step=3,gamma=.99,alpha=.5,n_envs=4)
ns.update(train_rng=ns['random'].Random(42),history=[],demo_history=[],training_states=[],
          total_steps=0,completed_episodes=0,updates=0,episodes_started=0,
          loss_sum=0.,training_lives_lost=0,started=ns['time'].time())
envs=[ns['make_env']() for _ in range(4)]
try:
 for slot,env in enumerate(envs): ns['training_states'].append(ns['start_training_game'](env,slot))
 ns['train_batched_games'](envs)
finally:
 for env in envs:env.close()
assert ns['completed_episodes']==8 and ns['episodes_started']==8
assert ns['total_steps']==sum(row['steps'] for row in ns['history'])
assert ns['updates']==ns['total_steps']//4-math.ceil(64/4)+1
assert all(math.isfinite(row['mean_loss']) for row in ns['history'])
assert all(np.isclose(row['shaped_return'],row['score']*.01-row['lives_lost']*.5) for row in ns['history'])
assert len({row['seed'] for row in ns['history']})==8
assert not any(ns['replay'].pending)
assert all(torch.isfinite(x).all() for x in ns['model'].state_dict().values())
assert any(not torch.equal(x.detach().cpu(),source['model'][k]) for k,x in ns['model'].state_dict().items())
# Confirm the priority-aware model loss changes live MPS weights and state saves.
p=ns['RUN_DIR']/'snapshot.pt';torch.save({'replay':ns['replay'].state_dict(copy_arrays=False),'optimizer':ns['optimizer'].state_dict()},p)
loaded=torch.load(p,map_location='cpu',weights_only=False)
restored=ns['PrioritizedNStepReplay'](500,n_step=3,gamma=.99,alpha=.5,n_envs=4);restored.load_state_dict(loaded['replay'])
assert len(restored)==len(ns['replay'])
report={'status':'passed','temporary_run_directory':str(ns['RUN_DIR']),'real_environment_games':8,
        'decisions':ns['total_steps'],'updates':ns['updates'],'lives_lost':ns['training_lives_lost'],
        'scores':[row['score'] for row in ns['history']],
        'evaluation_source_unchanged':True,'snapshot_roundtrip':True,
        'disposable_short_games_not_classroom_scores':True}
(ROOT/'.execution/advanced_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
