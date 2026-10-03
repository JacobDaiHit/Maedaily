"""Download/verify the pinned public reference model; training itself stays offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def verify(directory,manifest):
 failures=[]
 for name,info in manifest['files'].items():
  path=directory/name
  if not path.is_file():failures.append(name+': missing');continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
  if path.stat().st_size!=info['bytes'] or h.hexdigest()!=info['sha256']:failures.append(name+': hash/size differs')
 return failures

def main():
 parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
 parser.add_argument('--model-dir',type=Path,default=ROOT/'study_runs/models/Qwen2.5-0.5B-Instruct')
 parser.add_argument('--verify-only',action='store_true')
 args=parser.parse_args();manifest=json.loads((ROOT/'post_train/experiments/reference_model.json').read_text(encoding='utf-8'))
 directory=args.model_dir.resolve()
 if not args.verify_only:
  directory.mkdir(parents=True,exist_ok=True)
  command=[sys.executable,'-m','huggingface_hub.cli.hf','download',manifest['repo_id'],*manifest['files'],
           '--revision',manifest['revision'],'--local-dir',str(directory),'--max-workers','2']
  completed=subprocess.run(command,check=False)
  if completed.returncode:return completed.returncode
 failures=verify(directory,manifest)
 print(json.dumps({'status':'failed' if failures else 'verified','repo_id':manifest['repo_id'],
   'revision':manifest['revision'],'model_dir':str(directory),'file_count':len(manifest['files']),'failures':failures},ensure_ascii=False))
 return 1 if failures else 0
if __name__=='__main__':raise SystemExit(main())
