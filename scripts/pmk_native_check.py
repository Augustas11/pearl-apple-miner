#!/usr/bin/env python3
"""Local callback pipeline validation and R-A1 overhead measurement."""
import argparse
import asyncio
import json
import pathlib
import os
import platform
import subprocess
import sys
import time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'miner'))
import pearl_mining as pm
from pmk_miner.__main__ import gpu_lock,log
from pmk_miner.native import Native
from pmk_miner.pipeline import Pipeline,Shape
from pmk_miner.transport import GatewayJob
from pmk_miner.monitor import bits_to_target,choose_share_nbits

async def run(args):
    shape=Shape(args.m,args.n,args.k,2); shape.validate()
    header=pm.IncompleteBlockHeader(version=1,prev_block=bytes(32),merkle_root=bytes(32),timestamp=1,nbits=0x1e010000 if args.finds else 0x177fd82e)
    source=GatewayJob(bytes(header.to_bytes()),bits_to_target(header.nbits),3)
    native=Native(); records=[]; submissions=[]; completed=[]
    log('benchmark_environment',model=subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip(),
        os=platform.platform(),load_average=list(os.getloadavg()),
        power_source=subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0],
        probe_key=native.probe_key,python=sys.version.split()[0],rayon_threads=os.environ.get('RAYON_NUM_THREADS','native default'))
    def capture(event,**fields):
        log(event,**fields)
        if event=='python_overhead': completed.append(fields)
    pipeline=Pipeline(native,shape,capture)
    try:
        pipeline.set_template(source)
        first_a=bytes(pipeline.template.raw_root_a)
        roots=[bytes(native.build(pipeline.template,pipeline.b[0],shape.n,shape.k).raw_root_b) for _ in range(2)]
        assert roots[0]!=roots[1] and bytes(pipeline.template.raw_root_a)==first_a
        log('fixed_A_fresh_B',passed=True)
        share_nbits=choose_share_nbits(1e12)
        async def submit(job,proof):
            submissions.append(proof)
        async def lane(index):
            for _ in range(args.jobs//2):
                records.append(await pipeline.run(index,bits_to_target(share_nbits),share_nbits,submit,lambda _:True))
        start=time.monotonic()
        results=await asyncio.gather(lane(0),lane(1),return_exceptions=True)
        for result in results:
            if isinstance(result,BaseException): raise result
        elapsed=time.monotonic()-start
        assert all(r.history[-1]=='released' and not r.pending_finds for r in records)
        if args.finds: assert submissions, 'easy native pipeline did not produce a verified proof'
        percentages=[row['python_overhead_pct'] for row in completed]
        ratio=100*sum(r.python_seconds for r in records)/sum(row['wall_seconds'] for row in completed)
        report={'jobs':len(records),'verified_block_proofs':len(submissions),'python_overhead_pct':ratio,
            'max_job_python_overhead_pct':max(percentages),'min_job_python_overhead_pct':min(percentages),
            'elapsed_seconds':elapsed,'ops_per_second':shape.ops*len(records)/elapsed,
            'shape':[shape.m,shape.n,shape.k], 'timing_scope':'thread CPU orchestration excluding native calls; build through proof handoff wall denominator'}
        log('native_check',**report)
        if not args.finds: assert max(percentages)<2, report
    finally:
        native.close()

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--m',type=int,default=8192); parser.add_argument('--n',type=int,default=8192)
    parser.add_argument('--k',type=int,default=4096); parser.add_argument('--jobs',type=int,default=20); parser.add_argument('--finds',action='store_true')
    args=parser.parse_args(argv)
    with gpu_lock(log):
        asyncio.run(run(args))

if __name__=='__main__':
    main()
