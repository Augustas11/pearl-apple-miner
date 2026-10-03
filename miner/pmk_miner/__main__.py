"""Solo CLI; all network requests stay on loopback."""
from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib

from .runtime import gpu_lock, LabSession, RunState, ProbeSchedule, RoutineTelemetry, RotatingJsonlSink
from .native import Native, NativeError
from .pipeline import Pipeline, Shape, FatalDeviceError
from .transport import (GatewayClient, NodeRpcConfig, NodeRpcClient, GatewayLogTail,
                        SubmissionTracker, SubmissionOutcome, FatalCertVersionError,
                        TransportError, SubmissionLedger, GatewayJob)
from .transport import redact_credentials
from .monitor import (bits_to_target, choose_share_nbits,
                      expected_blocks_per_day, expected_shares, poisson_interval, validate_payout_startup,
                      validate_coinbase_payout)


def log(event,**fields):
    # Only explicit metadata enters logs. Never serialize config, requests, proofs,
    # arbitrary exceptions, addresses, node credentials, or native matrix buffers.
    print(json.dumps({'time':time.time(),'event':event,**fields},separators=(',',':')),flush=True)


def memory_limits(shape):
    ram=int(subprocess.check_output(['sysctl','-n','hw.memsize']))
    return ram//4,shape.validate(ram)

async def mine(args,config):
    shape=Shape(**{k:config.get(k,v) for k,v in dataclasses.asdict(Shape()).items()})
    budget,estimate=memory_limits(shape)
    gateway=GatewayClient(args.gateway)
    gateway_cfg=config.get('gateway',{})
    node_cfg=NodeRpcConfig.from_sources(toml_path=args.config,env_path=gateway_cfg.get('env_file'))
    node=NodeRpcClient(node_cfg)
    payout=config['payout']; script=payout['script']; hrp=payout['hrp']
    validate_payout_startup(node_cfg.mining_address,hrp,approved_script_hex=script)
    log_path=Path(gateway_cfg['log_file'])
    if not log_path.is_file():
        raise ValueError('gateway log must exist for consensus rejection monitoring')
    stop=asyncio.Event(); loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):
        loop.add_signal_handler(sig,stop.set)
    latest=None; fatal=None; pipeline=None; accepted=0
    state_dir=Path(config.get('run',{}).get('state_dir', Path.home()/'.local/state/pmk'))
    state=RunState(state_dir/'run.json')
    ledger=SubmissionLedger(state_dir/'submissions.jsonl')
    if state.data.get('halted') or ledger.fail_closed_entries():
        raise FatalDeviceError('durable safety halt requires operator investigation')
    tracking=set(); backups={}
    submitted={e.template_identity for e in ledger.entries().values()
               if e.outcome in {SubmissionOutcome.ACCEPTED,SubmissionOutcome.STALE,SubmissionOutcome.DUPLICATE}}
    retry_after={k:time.monotonic()+max(0,v-time.time())
                 for k,v in state.data.get('retry_after',{}).items() if v>time.time()}
    window_expected=state.data['window_expected']; window_shares=state.data['window_shares']
    start=time.monotonic()-state.data['elapsed_seconds']
    completed_ops=state.data['completed_ops']; daily=state.data['daily']
    max_accepted=config.get('run',{}).get('max_accepted',0)
    poll_interval=float(config.get('run',{}).get('template_poll_seconds',0.25))
    checkpoint_seconds=float(config.get('run',{}).get('checkpoint_seconds',5.0))
    routine_log=Path(config.get('run',{}).get('routine_log_file', state_dir/'routine.jsonl'))
    routine_sink=RotatingJsonlSink(routine_log,
        max_bytes=int(config.get('run',{}).get('routine_log_max_bytes',4*1024*1024)),
        backups=int(config.get('run',{}).get('routine_log_backups',4)))
    telemetry=RoutineTelemetry(log, seconds=float(config.get('run',{}).get('telemetry_interval_seconds',5.0)),
        routine_sink=routine_sink)

    async def poll_templates():
        nonlocal latest,fatal
        while not stop.is_set():
            try:
                candidate=await gateway.get_job()
                if candidate.target!=bits_to_target(candidate.bits):
                    raise FatalDeviceError('gateway target does not match immutable header')
                candidate.authorize_coinbase(approved_script_hex=script)
                latest=candidate
            except (FatalCertVersionError,FatalDeviceError) as exc:
                fatal=exc; stop.set()
                if pipeline: pipeline.cancel()
                log('alert',severity='P0',reason='cert_or_target_guard')
                return
            except (TransportError,TimeoutError,OSError):
                latest=None; log('gateway_unavailable')
            except Exception as exc:
                fatal=exc; stop.set()
                if pipeline: pipeline.cancel()
                log('alert',severity='P0',reason='malformed_job')
                return
            await asyncio.sleep(poll_interval)

    poller=asyncio.create_task(poll_templates())
    native=None
    try:
        # This initializes the production pipeline and runs all fail-closed probes.
        native=await asyncio.to_thread(Native)
        pipeline=Pipeline(native,shape,telemetry.log)
        probes=ProbeSchedule(float(config.get('run',{}).get('probe_interval_seconds',6*3600)))
        log('startup',probe_key=native.probe_key,shape=dataclasses.asdict(shape),
            memory_budget_bytes=budget,memory_estimate_bytes=estimate)

        async def track(job,tail,submission_id=None):
            nonlocal accepted,fatal
            outcome=await SubmissionTracker(node,gateway_log=tail).track(job)
            log('outcome',classification=str(outcome),template=job.template_identity)
            if outcome in {SubmissionOutcome.PROVING_ERROR,SubmissionOutcome.TRANSPORT}:
                submitted.discard(job.template_identity)
                retry_after[job.template_identity]=time.monotonic()+2
                state.save(retry_after={k:time.time()+max(0,v-time.monotonic()) for k,v in retry_after.items()})
            if outcome in {SubmissionOutcome.CONSENSUS_INVALID, SubmissionOutcome.UNKNOWN_SUBMISSION}:
                reason='consensus_invalid' if outcome==SubmissionOutcome.CONSENSUS_INVALID else 'unknown_submission'
                fatal=FatalDeviceError(f'P0 submission tracking failed closed: {outcome}')
                pipeline.cancel(); stop.set()
                state.save(halted=reason)
                log('alert',severity='P0',reason=reason)
            elif outcome==SubmissionOutcome.ACCEPTED:
                # Confirm the matching coinbase; tracker already checked header identity.
                block_hash=await node.get_best_block_hash()
                for _ in range(128):
                    block=await node.get_block(block_hash,2)
                    from .transport import _block_matches_job
                    if _block_matches_job(block,job):
                        validate_coinbase_payout(block,approved_script_hex=script,expected_hrp=hrp)
                        break
                    block_hash=block.get('previousblockhash')
                    if not block_hash: raise FatalDeviceError('accepted block disappeared before payout check')
                else: raise FatalDeviceError('accepted block beyond payout search limit')
                accepted+=1; log('payout_verified',accepted=accepted)
                if max_accepted and accepted>=max_accepted: stop.set()
            if submission_id is not None:
                ledger.finish(submission_id,outcome)
            backup=backups.pop(job.template_identity,None)
            if (backup and outcome in {SubmissionOutcome.PROVING_ERROR,SubmissionOutcome.TRANSPORT}
                    and not stop.is_set()):
                await asyncio.sleep(2)
                if current(backup[0]): await submit(*backup)

        def schedule_track(job,tail,submission_id):
            task=asyncio.create_task(track(job,tail,submission_id)); tracking.add(task)
            def track_failure(done):
                nonlocal fatal
                if not done.cancelled() and done.exception() is not None:
                    fatal=done.exception(); stop.set(); pipeline.cancel()
                    state.save(halted='confirmation_or_payout_failed')
                    log('alert',severity='P0',reason='confirmation_or_payout_failed',error_type=type(fatal).__name__)
            task.add_done_callback(track_failure)
            return task

        async def submit(job,proof):
            identity=job.template_identity
            if ledger.seen_proof(job,proof):
                log('outcome',classification='duplicate',template=identity)
                return
            if identity in submitted or time.monotonic()<retry_after.get(identity,0):
                # At most one already-verified backup per template, bounded globally.
                if identity not in backups and len(backups)<shape.slots:
                    backups[identity]=(job,proof)
                return
            entry=ledger.prepare(job,proof)
            if entry.outcome is not None:
                log('outcome',classification='duplicate',template=identity,submission_id=entry.submission_id)
                return
            job=job.with_submission_id(entry.submission_id)
            # Reserve before awaiting so simultaneous slots do not submit duplicate work.
            submitted.add(identity)
            tail=GatewayLogTail(log_path,offset=log_path.stat().st_size)
            try:
                await gateway.submit(job,proof)
            except (TransportError,TimeoutError,OSError):
                # An acknowledgement can be lost after admission. Persist the
                # reservation and confirm; never blindly retransmit this proof.
                log('submission_ack_lost',submission_id=entry.submission_id)
            schedule_track(job,tail,entry.submission_id)

        for entry in ledger.outstanding():
            try:
                header=bytes.fromhex(entry.header_hex)
                bits=int.from_bytes(header[72:76],'little')
                job=GatewayJob(header,bits_to_target(bits),3,submission_id=entry.submission_id)
            except ValueError:
                ledger.finish(entry.submission_id,SubmissionOutcome.UNKNOWN_SUBMISSION)
                raise FatalDeviceError('invalid outstanding ledger entry') from None
            submitted.add(job.template_identity)
            tail=GatewayLogTail(log_path,offset=0)
            schedule_track(job,tail,entry.submission_id)

        # Resolve pre-crash admissions before dispatching any new GPU work.
        if tracking:
            await asyncio.gather(*tracking)
        if fatal: raise fatal

        def current(job):
            return not stop.is_set() and latest is not None and latest.template_identity==job.template_identity

        while not stop.is_set():
            if probes.due:
                # All lane tasks have drained here; never probe while a job owns buffers.
                await asyncio.to_thread(probes.refresh,native)
                log('probe_refreshed',probe_key=native.probe_key)
            for task in list(tracking):
                if task.done():
                    tracking.remove(task)
                    task.result()
            if not latest or latest.template_identity in submitted or time.monotonic()<retry_after.get(latest.template_identity,0):
                await asyncio.sleep(0.05); continue
            source=latest
            if len(retry_after)>128:
                retry_after={k:v for k,v in retry_after.items() if v>time.monotonic()}
            await asyncio.to_thread(pipeline.set_template,source)
            async def lane(index):
                nonlocal completed_ops,daily,window_expected,window_shares
                while current(source) and not probes.due and source.template_identity not in submitted and time.monotonic()>=retry_after.get(source.template_identity,0):
                    # Retune within a long-lived template; expected counts use each job's own target.
                    local_bits=choose_share_nbits(completed_ops/(time.monotonic()-start) if completed_ops else 1e12)
                    record=await pipeline.run(index,bits_to_target(local_bits),local_bits,submit,current)
                    if record.cancelled: break
                    completed_ops+=shape.ops
                    shares=getattr(record,'share_count',0)
                    window_expected += expected_shares(shape.ops,local_bits)
                    window_shares += shares
                    # Disjoint windows retain each job's own S when targets change.
                    if window_expected >= 60:
                        lower,upper=poisson_interval(window_expected)
                        log('share_window',expected=window_expected,observed=window_shares,lower=lower,upper=upper)
                        if not lower <= window_shares <= upper:
                            log('alert',severity='health',reason='share_rate_poisson',expected=window_expected,observed=window_shares,lower=lower,upper=upper)
                        window_expected=0.0; window_shares=0
                    elapsed=time.monotonic()-start
                    if not daily or elapsed-daily>=86400:
                        daily=elapsed
                        rate=completed_ops/elapsed
                        log('expected_blocks_daily',expected=expected_blocks_per_day(rate,source.target),ops_per_second=rate,share_nbits=local_bits)
                    state.save_if_due(checkpoint_seconds,completed_ops=completed_ops,elapsed_seconds=elapsed,
                        window_expected=window_expected,window_shares=window_shares,daily=daily)
            tasks=[asyncio.create_task(lane(i)) for i in range(shape.slots)]
            results=await asyncio.gather(*tasks,return_exceptions=True)
            for result in results:
                if isinstance(result,BaseException): raise result
        pipeline.cancel()
        if tracking:
            await asyncio.gather(*tracking)
        if fatal: raise fatal
        state.flush(completed_ops=completed_ops,elapsed_seconds=time.monotonic()-start,
            window_expected=window_expected,window_shares=window_shares,daily=daily)
        telemetry.flush()
        log('stopped',accepted=accepted,completed_ops=completed_ops)
    finally:
        stop.set(); poller.cancel()
        await asyncio.gather(poller,return_exceptions=True)
        if pipeline: pipeline.cancel()
        if tracking:
            await asyncio.gather(*tracking,return_exceptions=True)
        state.flush()
        telemetry.flush()
        if native: native.close()


def main(*, standalone=False, pool_log=None, admission_refresh=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['solo','pool'],required=True)
    parser.add_argument('--gateway')
    parser.add_argument('--pool-url')
    parser.add_argument('--wallet-file',type=Path)
    parser.add_argument('--wallet-allowlist',type=Path)
    parser.add_argument('--worker')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--lab-owner-token',default=_env_first('B4_LAB_OWNER_TOKEN','PMK_B4_LAB_OWNER_TOKEN','PMK_LAB_OWNER_TOKEN'))
    parser.add_argument('--resume-hook',type=Path,default=_env_path('B4_LAB_RESUME_HOOK','PMK_B4_LAB_RESUME_HOOK','PMK_LAB_RESUME_HOOK'))
    parser.add_argument('--resume-report',type=Path,default=_env_path('B4_LAB_RESUME_REPORT','PMK_B4_LAB_RESUME_REPORT','PMK_LAB_RESUME_REPORT'))
    args=parser.parse_args()
    lab=None if standalone else LabSession(token=args.lab_owner_token,resume_hook=args.resume_hook,report=args.resume_report)
    config=None; native_redaction_config=None
    try:
        if sys.version_info[:2]!=(3,12): raise RuntimeError('Python 3.12 required')
        config=tomllib.loads(args.config.read_text())
        try:
            if args.mode != 'solo':
                raise ValueError('pool mode has no node credentials')
            if not args.gateway:
                raise ValueError('solo mode requires --gateway')
            native_redaction_config=NodeRpcConfig.from_sources(
                toml_path=args.config,env_path=config.get('gateway',{}).get('env_file'))
        except Exception:
            native_redaction_config=None
        if lab is not None:
            lab.check()
        try:
            with gpu_lock(log):
                if args.mode == 'pool':
                    from .pool_runtime import mine_pool
                    admission_deadline = admission_refresh() if admission_refresh is not None else None
                    asyncio.run(mine_pool(args,config,pool_log or log,memory_limits,
                                          admission_refresh=admission_refresh,
                                          admission_deadline=admission_deadline))
                else:
                    if not args.gateway:
                        raise ValueError('solo mode requires --gateway')
                    asyncio.run(mine(args,config))
        finally:
            outcome=lab.finish() if lab is not None else None
            if outcome: log('provider_resume',**outcome)
    except Exception as exc:
        fields={'error_type':type(exc).__name__}
        if isinstance(exc,(NativeError,FatalDeviceError)):
            fields.update(_native_error_fields(exc,native_redaction_config))
        log('fatal',**fields)
        return 1
    return 0


def _native_error_fields(exc, config=None):
    if isinstance(exc,FatalDeviceError):
        return {
            'gate': exc.gate,
            'native_function': exc.native_function,
            'native_code': exc.native_code,
            'error_message': redact_credentials(exc.native_message,config)[:256],
        }
    return {
        'native_function': exc.function,
        'native_code': exc.code,
        'error_message': redact_credentials(exc.message,config)[:256],
    }


def _env_first(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_path(*names):
    value = _env_first(*names)
    return Path(value) if value else None

if __name__=='__main__':
    raise SystemExit(main())
