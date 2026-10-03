"""Preregistered B/C reference matrix on one complete local pretrained model.

Training never downloads. All configurations and seeds are frozen before updates;
all arms are trained before final panels are opened. Negative method outcomes are
reported. Development-unqualified SFT stops without opening the final panels.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import random
import sys
import time
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from lesson_runtime import reserve_output_dir
from reference_instrumentation import environment_receipt, resource_probe, token_alignment, tokenization_manifest
from pretrained_adapter import LocalAdapterLM, digest_state
from model_quality_data import fixed_quality_data, protocol_manifest, development_gate, quality_gate
from model_training_lab import advantages, verify_answer, verifier_checks, clipped_policy_loss
from reference_reporting import build_report, single_factor_difference, create_evidence_package

DEFAULT_CONFIG={"schema":"maedaily-pretrained-bc-reference-v1","seeds":[17,23,41],
 "sft_steps":20,"sft_learning_rate":0.0003,"batch_size":8,"dpo_steps":6,"dpo_learning_rate":0.0001,
 "beta":0.2,"beta_ablation":0.1,"rl_rounds":3,"rl_learning_rate":0.00005,
 "groups":4,"group_size":4,"group_size_ablation":8,"temperature":2.0,"clip":0.2,
 "format_reward":0.1,"kl_coefficient":0.01,"max_new_tokens":8,"sampling_baseline_size":4,
 "adapter_rank":4,"adapter_alpha":8,"bootstrap_replicates":2000,"threads":1}
ARMS=("base","no_update","dpo","dpo_beta","continue_sft","rlvr","rlvr_group_size","rlvr_no_format","sampling_only")


def now(): return datetime.now(timezone.utc).isoformat()
def canonical(v): return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(v): return hashlib.sha256(canonical(v)).hexdigest()
def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def source_versions():
 code={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
 for name in ('scripts/lesson_runtime.py','scripts/reference_instrumentation.py'):
  code[name]=file_hash(ROOT/name)
 return code


def boundary_records(records):
 shortest=min(records,key=lambda r:(len(r['question'])+len(r['answer']),r['id']))
 negative=next(r for r in records if r['task']=='arithmetic' and r['answer'].startswith('-'))
 zero_copy=max((r for r in records if r['task']=='copy' and r.get('variant')=='two-leading-zeros'),key=lambda r:(len(r['question']),r['id']))
 return [shortest,negative,zero_copy]


def sft_probability_receipt(model,records):
 batch=model.batch(records);mask=batch['mask'][:,1:]
 with torch.no_grad():token_lp,sequence_lp,counts=model.logprobs(batch)
 numerator=float(-token_lp[mask].sum());denominator=int(mask.sum())
 rows=[]
 for index,record in enumerate(records):
  selected=mask[index];values=token_lp[index,selected].cpu().tolist()
  rows.append({'sample_id':record['id'],'effective_tokens':int(selected.sum()),'target_positions':(selected.nonzero().flatten()+1).cpu().tolist(),'token_logprobs':values,'negative_log_likelihood':-sum(values),'sequence_logprob':float(sequence_lp[index])})
 return {'normalization':'whole-batch target-token mean, answer plus EOS','negative_log_likelihood_numerator':numerator,'effective_token_denominator':denominator,'loss':numerator/denominator,'per_sample':rows}


def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write_lines(path,rows):
 with Path(path).open('x',encoding='utf-8') as f:
  for row in rows: f.write(canonical(row).decode()+'\n')


class Stages:
 def __init__(self,path): self.path=path;self.previous={}
 @contextmanager
 def stage(self,name,*,depends=(),inputs=None):
  if any(self.previous.get(x)!='passed' for x in depends): raise AssertionError('stage dependency not passed')
  started=now();checks={};before=self.artifact_hashes()
  self.append({'stage':name,'status':'running','started_at':started,'input_artifact_hashes':inputs or before})
  try:
   yield checks
  except Exception as error:
   self.previous[name]='failed';self.append({'stage':name,'status':'failed','started_at':started,'finished_at':now(),'failure_reason':str(error),'checks':checks,'input_artifact_hashes':inputs or before,'output_artifact_hashes':self.artifact_hashes()});raise
  self.previous[name]='passed';self.append({'stage':name,'status':'passed','started_at':started,'finished_at':now(),'checks':checks,'failure_reason':None,'input_artifact_hashes':inputs or before,'output_artifact_hashes':{k:v for k,v in self.artifact_hashes().items() if before.get(k)!=v}})
 def artifact_hashes(self):
  return {p.relative_to(self.path.parent).as_posix():file_hash(p) for p in self.path.parent.rglob('*') if p.is_file() and p!=self.path}
 def append(self,row):
  with self.path.open('a',encoding='utf-8') as f:f.write(canonical(row).decode()+'\n')


def normalize_config(raw):
 config={**DEFAULT_CONFIG,**raw}
 if set(config)!=set(DEFAULT_CONFIG): raise ValueError('unknown reference configuration fields')
 if len(config['seeds'])<3 or len(set(config['seeds']))!=len(config['seeds']) or any(type(x)is not int or x<0 for x in config['seeds']): raise ValueError('at least three unique training seeds required')
 for key in ('sft_steps','batch_size','dpo_steps','rl_rounds','groups','group_size','group_size_ablation','max_new_tokens','sampling_baseline_size','adapter_rank','bootstrap_replicates','threads'):
  if type(config[key])is not int or config[key]<=0:raise ValueError('positive integer required: '+key)
 if config['batch_size']%2 or min(config['group_size'],config['group_size_ablation'])<2:raise ValueError('balanced even batch and multi-response groups required')
 if config['group_size']==config['group_size_ablation'] or config['beta']==config['beta_ablation']: raise ValueError('ablations must differ')
 for key in ('sft_learning_rate','dpo_learning_rate','rl_learning_rate','beta','beta_ablation','temperature','clip','adapter_alpha'):
  if not isinstance(config[key],(int,float)) or isinstance(config[key],bool) or not __import__('math').isfinite(config[key]) or config[key]<=0:raise ValueError('invalid '+key)
 if config['format_reward']<=0:raise ValueError('baseline format_reward must be positive for the no-format ablation')
 if not 0<config['clip']<1 or config['max_new_tokens']>16:raise ValueError('invalid clip/short response budget')
 for key in ('format_reward','kl_coefficient'):
  if not isinstance(config[key],(float,int)) or isinstance(config[key],bool) or not __import__('math').isfinite(config[key]) or config[key]<0:raise ValueError('invalid '+key)
 return config


def balanced(data,rng,size):
 pools={t:[r for r in data if r['task']==t] for t in ('arithmetic','copy')}
 rows=[rng.choice(pools[t]) for t in pools for _ in range(size//2)];rng.shuffle(rows);return rows


def train_sft(model,records,steps,lr,seed,size,path,contract):
 rng=random.Random(seed);opt=torch.optim.AdamW(model.parameters(),lr=lr);logs=[];initial=digest_state(model.snapshot());tokens=0;began=time.perf_counter()
 for step in range(steps):
  rows=balanced(records,rng,size);batch=model.batch(rows);n=int(batch['mask'].sum());tokens+=n
  logs.append({'step':step+1,'sample_ids':[r['id'] for r in rows],'effective_tokens':n,**model.checked_step(model.sft_loss(batch),opt)})
 model.save(path,opt,step=steps,rng=rng,contract_hash=contract)
 return {'optimizer_steps':steps,'effective_update_tokens':tokens,'elapsed_seconds':time.perf_counter()-began,'initial_weight_hash':initial,'final_weight_hash':digest_state(model.snapshot()),'logs':logs}


def train_dpo(model,reference,records,config,beta,seed,path,contract):
 opt=torch.optim.AdamW(model.parameters(),lr=config['dpo_learning_rate']);rng=random.Random(seed);logs=[];initial=digest_state(model.snapshot());ref_hash=digest_state(reference);tokens=0;began=time.perf_counter()
 pool=[r for r in records if r['task']=='arithmetic']
 for step in range(config['dpo_steps']):
  rows=[rng.choice(pool) for _ in range(config['batch_size'])];wrong=[{**r,'answer':str(int(r['answer'])+1)} for r in rows]
  chosen=model.batch(rows);rejected=model.batch(wrong)
  with torch.no_grad():
   _,ref_c,_=model.logprobs(chosen,snapshot=reference);_,ref_r,_=model.logprobs(rejected,snapshot=reference)
  _,pol_c,_=model.logprobs(chosen);_,pol_r,_=model.logprobs(rejected)
  logits=beta*((pol_c-pol_r)-(ref_c-ref_r));loss=-F.logsigmoid(logits).mean()
  count=int(chosen['mask'].sum()+rejected['mask'].sum());tokens+=count
  four={k:v.detach().cpu().tolist() for k,v in {'policy_chosen':pol_c,'policy_rejected':pol_r,'reference_chosen':ref_c,'reference_rejected':ref_r}.items()}
  logs.append({'step':step+1,'sample_ids':[r['id'] for r in rows],'effective_tokens':count,'four_logprobs':four,'preference_source':'synthetic exact arithmetic answer versus answer+1; not human preference',**model.checked_step(loss,opt)})
 if digest_state(reference)!=ref_hash or any(x.requires_grad or x.grad is not None for x in reference.values()):raise AssertionError('DPO reference mutated')
 model.save(path,opt,step=len(logs),rng=rng,contract_hash=contract)
 return {'optimizer_steps':len(logs),'effective_update_tokens':tokens,'elapsed_seconds':time.perf_counter()-began,'initial_weight_hash':initial,'final_weight_hash':digest_state(model.snapshot()),'reference_hash':ref_hash,'logs':logs}


def train_rl(model,reference,records,config,seed,path,contract,*,group_size,format_reward):
 rng=random.Random(seed);gen=torch.Generator(device=model.device).manual_seed(seed+1000)
 opt=torch.optim.AdamW(model.parameters(),lr=config['rl_learning_rate']);logs=[];collections=[];initial=digest_state(model.snapshot());ref_hash=digest_state(reference);tokens=generated=updates=0;began=time.perf_counter()
 pool=[r for r in records if r['task']=='arithmetic']
 for round_id in range(config['rl_rounds']):
  old=model.snapshot();old_hash=digest_state(old);rows=[];record_rows=[];groups=[]
  for group_id in range(config['groups']):
   record=rng.choice(pool);members=[model.generate(record,max_new_tokens=config['max_new_tokens'],temperature=config['temperature'],sample=True,generator=gen,snapshot=old) for _ in range(group_size)]
   rewards=[float(x['reward']['correct'])+format_reward*float(x['reward']['format_valid']) for x in members]
   values,stats=advantages(rewards)
   for index,item in enumerate(members):
    item.update({'group_id':group_id,'round':round_id,'advantage':values[index],'reward_total':rewards[index],'old_weight_hash':old_hash,'reference_weight_hash':ref_hash,'split':'train'});rows.append(item);record_rows.append(record)
   groups.append({'group_id':group_id,'sample_id':record['id'],'rewards':rewards,'advantages':values,**stats})
  batch=model.batch(record_rows,answers=[r['response_token_ids'] for r in rows]);mask=batch['mask'][:,1:];old_lp=torch.zeros_like(mask,dtype=torch.float32)
  for index,row in enumerate(rows):old_lp[index,mask[index]]=torch.tensor(row['behavior_token_logprobs'],device=model.device)
  with torch.no_grad():
   replay,_,_=model.logprobs(batch,snapshot=old,temperature=config['temperature']);ref_lp,_,_=model.logprobs(batch,snapshot=reference,temperature=config['temperature'])
  replay_error=float((replay[mask]-old_lp[mask]).abs().max())
  if replay_error>0.0003:raise AssertionError('sample distribution replay differs')
  current,_,_=model.logprobs(batch,temperature=config['temperature']);adv=torch.tensor([r['advantage'] for r in rows],device=model.device).detach()
  loss,ratio=clipped_policy_loss(current,old_lp.detach(),mask,adv,config['clip']);ratio_error=float((ratio[mask].detach()-1).abs().max())
  if ratio_error>0.0003:raise AssertionError('on-policy ratio differs from one')
  # Nonnegative k3 sample estimator under the frozen sampling distribution;
  # reference log-probs are detached, and only current policy gets gradients.
  difference=ref_lp.detach()-current;kl=(difference.exp()-difference-1)
  kl_loss=((kl*mask).sum(-1)/mask.sum(-1)).mean();total_loss=loss+config['kl_coefficient']*kl_loss
  active=any(g['active'] for g in groups)
  policy_gradient_norm=0.0
  if active:
   pg=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
   finite=[g for g in pg if g is not None]
   if not finite or any(not bool(torch.isfinite(g).all()) for g in finite):raise AssertionError('policy-gradient term has absent/nonfinite derivatives')
   policy_gradient_norm=float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in finite])))
   if policy_gradient_norm<=0:raise AssertionError('only KL may have a gradient; policy-gradient update is not proved')
  check=model.checked_step(total_loss,opt) if active else {'loss':float(total_loss.detach()),'gradient_norm':0.0,'parameter_delta_norm':0.0,'weight_before':digest_state(model.snapshot()),'weight_after':digest_state(model.snapshot())}
  updates+=int(active);n=int(mask.sum());tokens+=n if active else 0;generated+=n
  logs.append({'round':round_id+1,'status':'updated' if active else 'skipped_zero_advantage','active_groups':sum(g['active'] for g in groups),'policy_gradient_norm':policy_gradient_norm,'probability_replay_max_error':replay_error,'ratio_max_error_before':ratio_error,'policy_loss':float(loss.detach()),'kl_sample_estimate':float(kl_loss.detach()),'effective_tokens':n,'old_detached':not old_lp.requires_grad,'reference_detached':not ref_lp.requires_grad,'advantage_detached':not adv.requires_grad,**check})
  collections.append({'round':round_id+1,'groups':groups,'rollouts':rows,'action_mask':mask.cpu().tolist(),'old_token_logprobs':old_lp.cpu().tolist(),'reference_token_logprobs':ref_lp.cpu().tolist(),'current_token_logprobs_before':current.detach().cpu().tolist(),'ratios_before':ratio.detach().cpu().tolist(),'generator_state_after':gen.get_state().cpu().tolist()})
  if digest_state(old)!=old_hash or digest_state(reference)!=ref_hash:raise AssertionError('frozen snapshot changed')
  model.save(path.parent/f'{path.stem}_round{round_id+1}.pt',opt,step=round_id+1,rng=rng,generator=gen,contract_hash=contract)
 model.save(path,opt,step=config['rl_rounds'],rng=rng,generator=gen,contract_hash=contract)
 return {'optimizer_steps':updates,'collection_rounds':config['rl_rounds'],'effective_update_tokens':tokens,'generated_tokens':generated,'generated_samples':config['rl_rounds']*config['groups']*group_size,'elapsed_seconds':time.perf_counter()-began,'initial_weight_hash':initial,'final_weight_hash':digest_state(model.snapshot()),'reference_hash':ref_hash,'policy_gradient_proved':updates>0,'logs':logs,'collections':collections}


def select_sampling_answer(samples):
 # Selection never reads reward, answer, or any gold-oracle field.
 if not samples: raise ValueError("sampling baseline needs candidates")
 valid=[r for r in samples if r["termination_reason"]=="eos" and r["text"]!="-0" and len(r["text"])<=4 and re.fullmatch(r"-?(0|[1-9][0-9]*)",r["text"]) is not None]
 if not valid:return samples[0]
 counts={r["text"]:sum(x["text"]==r["text"] for x in valid) for r in valid}
 return max(valid,key=lambda x:counts[x["text"]])


def evaluate(model,panels,config):
 values={};pred=[]
 for name,records in panels.items():
  rows=model.generate_many(records,max_new_tokens=config['max_new_tokens'])
  values[name]={'correct':sum(r['reward']['correct'] for r in rows),'format_valid':sum(r['reward']['format_valid'] for r in rows),'total':len(rows)}
  pred.extend([{**r,'panel':name} for r in rows])
 return {'panels':values,'predictions':pred}


def run(raw_config,local_path,device,output):
 config=normalize_config(raw_config);torch.set_num_threads(config['threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 stages=Stages(output/'stages.jsonl');data=fixed_quality_data();manifest=protocol_manifest(data)
 code=source_versions()
 model_files={p.name:file_hash(p) for p in Path(local_path).iterdir() if p.is_file()}
 registration={'config':config,'arms':list(ARMS),'data':manifest,'model_file_hashes':model_files,'code_hashes':code,'final_used_for_selection':False,'preregistered_at':now(),
 'comparison_hypothesis':'From the same qualified SFT weights, report DPO/RLVR changes on unseen families with copy retention; no positive-effect requirement.',
 'reward_definition':'exact correct + registered format_reward; reward labels use train only','sampling_baseline':'majority canonical integer among N samples; tie earliest; oracle never used to select',
 'budget_matching':'Evaluation greedy arms have same protocol; sampling-only spends N times generations. Training and rollout budgets differ and are reported, no method superiority claim.'}
 contract=sha(registration);write(output/'preregistration.json',registration);write(output/'config.resolved.json',config);write(output/'data_manifest.json',{'data':data,'manifest':manifest});write(output/'source_versions.json',code)
 arm_configs={arm:dict(config) for arm in ARMS};arm_configs['dpo_beta']['beta']=config['beta_ablation'];arm_configs['rlvr_group_size']['group_size']=config['group_size_ablation'];arm_configs['rlvr_no_format']['format_reward']=0.0
 write(output/'arm_configs.json',arm_configs)
 with stages.stage('00_contract',inputs={'preregistration':file_hash(output/'preregistration.json')}) as checks:
  load_started=time.perf_counter()
  model=LocalAdapterLM(local_path,device=device,seed=config['seeds'][0],rank=config['adapter_rank'],alpha=config['adapter_alpha'])
  load_seconds=time.perf_counter()-load_started
  model.config['model_files_sha256']=sha(model_files)
  environment=environment_receipt(model,load_seconds);probe=resource_probe(model,boundary_records(data['train']))
  if probe.get('status')!='passed' or not probe.get('state_restored') or not probe.get('frozen_base_unchanged'):raise AssertionError('resource probe did not restore the caller state')
  write(output/'environment.json',environment);write(output/'resource_probe.json',probe)
  write(output/'tokenization_manifest.json',tokenization_manifest(model,data))
  checks.update({'environment_sha256':file_hash(output/'environment.json'),'resource_probe_sha256':file_hash(output/'resource_probe.json'),'tokenization_manifest_sha256':file_hash(output/'tokenization_manifest.json')})
  checks.update({'offline_model_loaded':True,'trainable_parameters':sum(p.numel() for p in model.parameters()),'base_parameters':sum(p.numel() for p in model.model.parameters()),'data_hash':sha(data),'contract_hash':contract})
 from adapter_checks import check_sft,check_dpo,check_lora_frozen,check_verifier_boundaries,check_exact_resume
 with stages.stage('01_batch',depends=('00_contract',)) as checks:
  chosen=data['train'][:2];wrong=[{**r,'answer':str(int(r['answer'])+1)} for r in chosen]
  checks.update({'sft':check_sft(model,chosen,atol=0.00003),'dpo':check_dpo(model,chosen,wrong,beta=config['beta'],atol=0.00003),'frozen':check_lora_frozen(model,full_base_hash=True),'verifier':check_verifier_boundaries()})
  boundary=boundary_records(data['train']);write_lines(output/'token_alignment.jsonl',token_alignment(model,boundary))
  checks.update({'alignment_source_ids':[r['id'] for r in boundary],'token_alignment_sha256':file_hash(output/'token_alignment.jsonl')})
  receipt=sft_probability_receipt(model,chosen);write(output/'sft_loss_receipt.json',receipt)
  if 'loss' in checks['sft'] and abs(receipt['loss']-checks['sft']['loss'])>0.00003:raise AssertionError('SFT probability receipt differs from independent CE')
  checks['sft_loss_receipt_sha256']=file_hash(output/'sft_loss_receipt.json')
  write(output/'batch_checks.json',checks)
 with stages.stage('exact_resume',depends=('01_batch',)) as checks:
  directory=output/'resume_checks';checks.update(check_exact_resume(model,data['train'],directory,learning_rate=config['sft_learning_rate'],contract_hash=contract));write(output/'resume_check.json',checks)
 initial=model.snapshot();all_snapshots={};budgets={};dev_reports=[];receipts=[]
 for seed in config['seeds']:
  seed_dir=output/f'seed_{seed}';seed_dir.mkdir();torch.manual_seed(seed)
  with torch.no_grad():
   for module in model.model.modules():
    if hasattr(module,'a') and hasattr(module,'b') and isinstance(module.a,torch.nn.Parameter):
     torch.nn.init.kaiming_uniform_(module.a,a=__import__('math').sqrt(5));module.b.zero_()
  model.config['seed']=seed;base=model.snapshot();all_snapshots[(seed,'base')]=base;model.save(seed_dir/'base.pt',contract_hash=contract)
  with stages.stage(f'{seed}:02_sft',depends=('exact_resume',)) as checks:
   b=train_sft(model,data['train'],config['sft_steps'],config['sft_learning_rate'],seed,config['batch_size'],seed_dir/'sft.pt',contract);budgets[(seed,'no_update')]=b
   write(seed_dir/'sft_train.json',b);ev=evaluate(model,{x:data[x] for x in ('dev','dev_copy')},config);gate=development_gate(ev['panels'])
   write(seed_dir/'development.json',{'evaluation':ev,'gate':gate});dev_reports.append({'seed':seed,'gate':gate,'panels':ev['panels']});all_snapshots[(seed,'no_update')]=model.snapshot();checks.update({'real_updates':b['optimizer_steps'],'development_qualified':gate['qualified'],'final_opened':False})
 if not all(r['gate']['qualified'] for r in dev_reports):
  result={'status':'development_unqualified','final_metrics':None,'planned_seeds':config['seeds'],'executed_training_seeds':config['seeds'],'development':dev_reports,'contract_hash':contract,'final_opened':False}
  write(output/'summary.json',result);return result
 # Freeze all SFT identities before building further branches; no final access yet.
 write(output/'sft_lock.json',{'contract_hash':contract,'development':dev_reports,'weights':{str(seed):digest_state(all_snapshots[(seed,'no_update')]) for seed in config['seeds']}})
 for seed in config['seeds']:
  directory=output/f'seed_{seed}';model.config['seed']=seed;reference=all_snapshots[(seed,'no_update')]
  with stages.stage(f'{seed}:03_dpo',depends=(f'{seed}:02_sft',)) as checks:
   for arm,beta in [('dpo',config['beta']),('dpo_beta',config['beta_ablation'])]:
    model.load_snapshot(reference);b=train_dpo(model,reference,data['train'],config,beta,seed+200,directory/f'{arm}.pt',contract);budgets[(seed,arm)]=b;all_snapshots[(seed,arm)]=model.snapshot();write(directory/f'{arm}_train.json',b)
   checks.update({'reference_hash':digest_state(reference),'beta_single_factor':single_factor_difference(arm_configs['dpo'],arm_configs['dpo_beta'],expected_path='beta')})
  with stages.stage(f'{seed}:04_rollout_05_update',depends=(f'{seed}:02_sft',)) as checks:
   model.load_snapshot(reference);b=train_sft(model,data['train'],config['rl_rounds'],config['sft_learning_rate'],seed+300,config['batch_size'],directory/'continue_sft.pt',contract);budgets[(seed,'continue_sft')]=b;all_snapshots[(seed,'continue_sft')]=model.snapshot();write(directory/'continue_sft_train.json',b)
   for arm,size,format_reward in [('rlvr',config['group_size'],config['format_reward']),('rlvr_group_size',config['group_size_ablation'],config['format_reward']),('rlvr_no_format',config['group_size'],0.0)]:
    model.load_snapshot(reference);b=train_rl(model,reference,data['train'],config,seed+400,directory/f'{arm}.pt',contract,group_size=size,format_reward=format_reward);budgets[(seed,arm)]=b;all_snapshots[(seed,arm)]=model.snapshot();write(directory/f'{arm}_train.json',b)
   if not all(budgets[(seed,arm)]['policy_gradient_proved'] for arm in ('rlvr','rlvr_group_size','rlvr_no_format')): raise AssertionError('RL variation has no proved policy-gradient update; final panels remain sealed')
   checks.update({'policy_gradient_proved':{arm:budgets[(seed,arm)]['policy_gradient_proved'] for arm in ('rlvr','rlvr_group_size','rlvr_no_format')},'group_size_single_factor':single_factor_difference(arm_configs['rlvr'],arm_configs['rlvr_group_size'],expected_path='group_size'),'reward_single_factor':single_factor_difference(arm_configs['rlvr'],arm_configs['rlvr_no_format'],expected_path='format_reward')})
  all_snapshots[(seed,'sampling_only')]=reference
 with stages.stage('checkpoint_reload',depends=tuple(f'{seed}:04_rollout_05_update' for seed in config['seeds'])) as checks:
  results=[]
  for seed in config['seeds']:
   model.config['seed']=seed
   for arm in ARMS:
    filename='sft.pt' if arm in ('no_update','sampling_only') else arm+'.pt'
    expected=all_snapshots[(seed,arm)];model.load_snapshot(expected);batch=model.batch(data['train'][:2])
    with torch.no_grad():before_lp=model.logprobs(batch)[1]
    model.restore(output/f'seed_{seed}'/filename,contract_hash=contract)
    with torch.no_grad():after_lp=model.logprobs(batch)[1]
    if digest_state(model.snapshot())!=digest_state(expected) or not torch.equal(before_lp,after_lp):raise AssertionError('saved checkpoint differs from evaluated snapshot')
    results.append({'seed':seed,'arm':arm,'file':filename,'weight_hash':digest_state(expected),'reload_exact':True})
  checks.update({'reloaded_checkpoints':results});write(output/'reload_checks.json',checks)
 if source_versions()!=code:raise AssertionError('experiment source changed after preregistration; final remains sealed')
 final_panels={x:data[x] for x in ('eval_main','eval_transfer','eval_regression')};protocol_hash=sha({'data':manifest,'config':config,'arms':ARMS});records=[];qualities=[]
 with stages.stage('06_eval',depends=('checkpoint_reload',),inputs={'sft_lock':file_hash(output/'sft_lock.json')}) as checks:
  write(output/'final_opening.json',{'at':now(),'protocol_hash':protocol_hash,'selection_locked':True,'no_retuning_after_final':True})
  for seed in config['seeds']:
   directory=output/f'seed_{seed}';model.config['seed']=seed
   for arm in ARMS:
    model.load_snapshot(all_snapshots[(seed,arm)]);checkpoint=digest_state(model.snapshot())
    if arm!='sampling_only':evaluation=evaluate(model,final_panels,config)
    else:
     predictions=[];gen=torch.Generator(device=model.device).manual_seed(seed+500)
     for panel,items in final_panels.items():
      for item in items:
       samples=[model.generate(item,max_new_tokens=config['max_new_tokens'],sample=True,generator=gen) for _ in range(config['sampling_baseline_size'])]
       selected=select_sampling_answer(samples)
       predictions.append({**selected,'panel':panel,'all_samples':samples,'evaluation_cost_tokens':sum(len(x['response_token_ids']) for x in samples),'elapsed_seconds':sum(x['elapsed_seconds'] for x in samples)})
     evaluation={'predictions':predictions,'panels':{panel:{'correct':sum(r['reward']['correct'] for r in predictions if r['panel']==panel),'format_valid':sum(r['reward']['format_valid'] for r in predictions if r['panel']==panel),'total':len(items)} for panel,items in final_panels.items()}}
    write(directory/f'{arm}_evaluation.json',evaluation)
    if arm=='no_update':qualities.append({'seed':seed,'qualification':quality_gate(evaluation)})
    origin=budgets.get((seed,arm),budgets[(seed,'no_update')]);steps=origin['optimizer_steps']
    if arm in ('base','sampling_only'): origin={**origin,'initial_weight_hash':checkpoint};steps=0
    receipts.append({'training_run_id':f'{seed}:{arm}','training_seed':seed,'arm':arm,'optimizer_steps':steps,'initial_weight_hash':origin['initial_weight_hash'],'final_weight_hash':checkpoint,'checkpoint_id':checkpoint,'source_training_run_id':f'{seed}:no_update' if arm=='sampling_only' else None,'source_checkpoint_id':digest_state(all_snapshots[(seed,'no_update')]) if arm=='sampling_only' else None})
    lookup={r['id']:r for items in final_panels.values() for r in items}
    for row in evaluation['predictions']:
     original=lookup[row['sample_id']]
     records.append({'sample_id':row['sample_id'],'family_id':original['family_id'],'panel':row['panel'],'bucket':original['family'],'arm':arm,'training_seed':seed,'training_run_id':f'{seed}:{arm}','checkpoint_id':checkpoint,'correct':row['reward']['correct'],'format_valid':row['reward']['format_valid'],'input_tokens':len(row['prompt_token_ids']),'output_tokens':len(row['response_token_ids']),'evaluation_cost_tokens':row.get('evaluation_cost_tokens',len(row['response_token_ids'])),'evaluation_cost_input_tokens':len(row['prompt_token_ids'])*(config['sampling_baseline_size'] if arm=='sampling_only' else 1),'elapsed_seconds':row['elapsed_seconds'],'output_text':row['text'],'prompt_text':model.tokenizer.decode(row['prompt_token_ids']),'termination_reason':row['termination_reason'],'question':row['question'],'expected_answer':row['answer'],'eval_protocol_hash':protocol_hash,'max_new_tokens':config['max_new_tokens']})
  checks.update({'all_arms_reported':list(ARMS),'all_seeds_reported':config['seeds'],'final_baseline_quality':qualities})
 with stages.stage('07_reproduce',depends=('06_eval',)) as checks:
  write_lines(output/'predictions.jsonl',records);write(output/'training_receipts.json',receipts)
  report=build_report(records,training_receipts=receipts,baseline_arm='no_update',bootstrap_replicates=config['bootstrap_replicates']);write(output/'report.json',report)
  mechanisms=all(budgets[(seed,arm)]['policy_gradient_proved'] for seed in config['seeds'] for arm in ('rlvr','rlvr_group_size','rlvr_no_format'))
  passed=mechanisms and all(q['qualification']['qualified'] for q in qualities)
  result={'status':'passed' if passed else 'completed_with_failed_gate','mechanisms_passed':mechanisms,'baseline_qualities':qualities,'executed_training_seeds':config['seeds'],'arms':list(ARMS),'contract_hash':contract,'budgets':{f'{seed}:{arm}':{k:v for k,v in b.items() if k not in ('logs','collections')} for (seed,arm),b in budgets.items()},'cuda_peak_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,'limitations':['Synthetic verified arithmetic/copy data; no natural-language benchmark claim.','Budgets differ and are explicitly reported.','Automatic blind-review package is pending actual reviewer scoring.','No retuning after final panels opened.']}
  write(output/'summary.json',result);checks.update({'report_recomputed':True,'mechanisms_passed':mechanisms,'baseline_qualified':passed})
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--output-dir',type=Path)
 args=parser.parse_args();raw={} if args.config is None else json.loads(args.config.read_text(encoding='utf-8'));normalize_config(raw)
 output=reserve_output_dir(args.output_dir,'project-bc-reference')
 try:result=run(raw,args.model_dir,args.device,output)
 except Exception as error:write(output/'failure.json',{'status':'failed','reason':str(error),'type':type(error).__name__});raise
 print(json.dumps({'status':result['status'],'output':str(output)},ensure_ascii=False));return 0 if result['status']=='passed' else 4
if __name__=='__main__':raise SystemExit(main())
