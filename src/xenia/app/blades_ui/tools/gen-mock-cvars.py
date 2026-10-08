#!/usr/bin/env python3
"""Regenerates src/mock-cvars.json: every config setting (DEFINE_* in the
emulator source) with its default, for the browser mock (npm run dev).

    python3 tools/gen-mock-cvars.py "$(cd ../.. && pwd)" src/mock-cvars.json

The app reports the same fields at runtime (blades_model.cc); this file only
lets the blades be built and screenshotted without it.
"""
import re,os,json,sys
src,out=os.path.realpath(sys.argv[1]),sys.argv[2]
pat=re.compile(r'DEFINE_(bool|int32|uint32|uint64|int64|double|string|path|unicode_string)\(\s*([a-zA-Z0-9_]+)\s*,(.*?)"\s*\)\s*;?\s*\n',re.S)
res={}
for root,_,fs in os.walk(src):
  if any(x in root for x in ('/d3d12','xinput','xaudio2','/blades_ui')): continue
  for f in fs:
    if not f.endswith(('.cc','.mm')) or '_test' in f or f.endswith('_win.cc') or 'shader_compiler_main' in f or 'trace_dump' in f: continue
    s=open(os.path.join(root,f),errors='ignore').read()
    for m in pat.finditer(s):
      t,name,body=m.groups(); body=body+'"'
      strs=re.findall(r'"((?:[^"\\]|\\.)*)"',body)
      if not strs: continue
      cat=strs[-1]
      first=body.split(',')[0].strip()
      desc="".join(strs[:-1] if not first.startswith('"') else strs[1:-1])
      desc=desc.encode().decode('unicode_escape',errors='ignore')
      typ={'bool':'bool','int32':'int','int64':'int','uint32':'uint','uint64':'uint','double':'float','string':'string','path':'path','unicode_string':'string'}[t]
      if typ=='bool': v= first=='true'
      elif typ in('int','uint'):
        try: v=int(first.rstrip('uULl'),0)
        except: v=0
      elif typ=='float':
        try: v=float(first.rstrip('f'))
        except: v=0.0
      else: v=strs[0] if first.startswith('"') else ""
      res[name]={"name":name,"category":cat,"type":typ,"value":v,"defaultValue":v,"description":desc.strip(),"changed":False}
json.dump(res,open(out,'w'),indent=0)
print(len(res))
