"""Pool orchestration: immutable work, local verification and durable outcomes."""
from __future__ import annotations

import asyncio
from collections import Counter
import dataclasses
import math
from pathlib import Path
import signal
import time

from .monitor import poisson_interval, bits_to_target
from .native import Native
from .pipeline import Pipeline, Shape, FatalDeviceError, PoolSubmissionPolicy
from .runtime import RunState, ProbeSchedule, RoutineTelemetry, RotatingJsonlSink
from .transport import SubmissionLedger
from .pool import PoolClient, mask_wallet


def load_wallet(wallet_file, allowlist_file):
    if not wallet_file or not allowlist_file:
        raise ValueError('pool wallet file and operator allowlist are required')
    wallet = Path(wallet_file).read_text(encoding='utf-8').strip()
    allowed = {line.strip() for line in Path(allowlist_file).read_text(encoding='utf-8').splitlines()
               if line.strip() and not line.lstrip().startswith('#')}
    if not wallet or len(wallet) > 256 or not wallet.isascii() or any(c.isspace() for c in wallet):
        raise ValueError('invalid pool wallet file')
    if wallet not in allowed:
        raise ValueError('pool wallet is absent from operator allowlist')
    return wallet


def rejection_alarm(outcome, accepted_before):
    if outcome in {'invalid', 'low-difficulty'}:
        return ('pool rejected SG config; try next pool' if accepted_before == 0 else
                'P0: pool rejected locally verified share')
    return None


async def mine_pool(args, config, log, memory_limits, *, admission_refresh=None, admission_deadline=None):
    wallet = load_wallet(args.wallet_file, args.wallet_allowlist or config.get('pool', {}).get('wallet_allowlist'))
    emit = log
    def log(event, **fields):
        emit(event, **{key: value.replace(wallet, mask_wallet(wallet))
                      if isinstance(value, str) else value for key, value in fields.items()})
    if not args.pool_url or not args.worker:
        raise ValueError('pool URL and worker are required')
    # Pool startup never opens the gateway environment or constructs a node RPC client.
    shape = Shape(**{k: config.get(k, v) for k, v in dataclasses.asdict(Shape()).items()})
    budget, estimate = memory_limits(shape)
    run = config.get('run', {})
    pool_cfg = config.get('pool', {})
    seconds = float(run.get('max_seconds', 0))
    max_accepted = int(run.get('max_accepted', 0))
    max_submitted = int(run.get('max_submitted', 0))
    max_jobs = int(run.get('max_jobs', 0))
    if not math.isfinite(seconds) or seconds < 0 or min(max_accepted, max_submitted, max_jobs) < 0:
        raise ValueError('invalid pool run limit')
    state_dir = Path(run.get('state_dir', Path.home() / '.local/state/pmk-pool'))
    state = RunState(state_dir / 'pool-run.json')
    ledger = SubmissionLedger(state_dir / 'pool-submissions.jsonl')
    if state.data.get('halted') or ledger.fail_closed_entries():
        raise FatalDeviceError('durable pool safety halt requires operator investigation')
    # A previous process's session cannot be resumed and an uncertain share cannot
    # be retransmitted. Preserve its immutable reservation with a terminal outcome.
    for entry in ledger.outstanding():
        ledger.finish(entry.submission_id, 'transport')
    counts = Counter()
    completed_ops = observed = blocks = submitted = gate_failures = 0
    completed_jobs = reserved_jobs = 0
    expected = compact_expected = 0.0
    window_expected = 0.0
    window_observed = 0
    started = time.monotonic()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    client = PoolClient(args.pool_url, wallet, args.worker,
                        difficulty_floor=pool_cfg.get('difficulty_floor', 10000), log=log)
    def telemetry_log(event, **fields):
        if event == 'routine_telemetry':
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = completed_ops / elapsed
            target = client.latest.target if client.latest else 0
            shares_per_second = rate / 2 * target / 2**256
            fields.update(jobs_per_second=completed_jobs / elapsed, tops=rate / 1e12,
                accepted=counts['accepted'],
                rejected=sum(counts[k] for k in ('stale', 'duplicate', 'low-difficulty', 'invalid')),
                expected_share_seconds=1 / shares_per_second if shares_per_second else None)
        log(event, **fields)

    telemetry = RoutineTelemetry(telemetry_log, seconds=float(run.get('telemetry_interval_seconds', 5)),
        routine_sink=RotatingJsonlSink(state_dir / 'pool-routine.jsonl'))
    native = pipeline = None
    fatal = None
    last_template_build = float('-inf')
    supervisor = asyncio.create_task(client.run(stop))
    timer = None
    if seconds:
        async def deadline():
            await asyncio.sleep(seconds)
            stop.set()
        timer = asyncio.create_task(deadline())

    def current(job):
        return (not stop.is_set() and not (pipeline and pipeline.stopped)
                and client.latest is not None and job.session_id == client.session_id)

    def checkpoint():
        state.save_if_due(float(run.get('checkpoint_seconds', 5)),
                   pool_counts=dict(counts), completed_ops=completed_ops,
                   pool_observed=observed, pool_expected=expected,
                   elapsed_seconds=time.monotonic() - started)

    def pipeline_log(event, **fields):
        nonlocal completed_ops, observed, blocks, expected, compact_expected, completed_jobs
        nonlocal window_expected, window_observed
        if event == 'work_completed':
            ops = fields['ops']
            completed_jobs += 1
            completed_ops += ops
            observed += fields['shares']
            blocks += fields['blocks']
            expected += (ops // 2) * fields['target'] / 2**256
            compact_expected += (ops // 2) * bits_to_target(fields['share_nbits']) / 2**256
            window_expected += (ops // 2) * fields['target'] / 2**256
            window_observed += fields['shares']
            if window_expected >= 60:
                lower, upper = poisson_interval(window_expected)
                log('pool_share_window', expected=window_expected, observed=window_observed,
                    lower=lower, upper=upper, poisson_ok=lower <= window_observed <= upper)
                if not lower <= window_observed <= upper:
                    log('alert', severity='health', reason='share_rate_poisson')
                window_expected = 0.0
                window_observed = 0
            checkpoint()
            return
        telemetry.log(event, **fields)

    async def submit(job, proof):
        nonlocal submitted
        if not current(job) or (max_submitted and submitted >= max_submitted):
            return
        if ledger.seen_proof(job, proof):
            return
        entry = ledger.prepare(job, proof, cfg=pipeline.config)
        submitted += 1
        outcome = str(await client.submit(job, proof))
        accepted_before = counts['accepted']
        ledger.finish(entry.submission_id, outcome)
        counts[outcome] += 1
        log('pool_outcome', classification=outcome, submission_id=entry.submission_id,
            pool_job_id=job.pool_job_id, accepted=counts['accepted'], submitted=submitted)
        checkpoint()
        alarm = rejection_alarm(outcome, accepted_before)
        if alarm:
            state.save(halted=alarm)
            log('alert', severity='compatibility' if accepted_before == 0 else 'P0', reason=alarm)
            raise FatalDeviceError(alarm)
        verdicts = sum(counts[k] for k in ('accepted', 'stale', 'duplicate', 'low-difficulty', 'invalid'))
        if verdicts and counts['stale'] / verdicts > .05:
            state.save(halted='pool stale shares exceed 5%')
            log('alert', severity='health', reason='pool stale shares exceed 5%')
            raise FatalDeviceError('pool stale shares exceed 5%')
        if ((max_accepted and counts['accepted'] >= max_accepted) or
                (max_submitted and submitted >= max_submitted)):
            stop.set()

    try:
        native = await asyncio.to_thread(Native)
        pipeline = Pipeline(native, shape, pipeline_log, submission_policy=PoolSubmissionPolicy())
        probes = ProbeSchedule(float(run.get('probe_interval_seconds', 6 * 3600)))
        def schedule_admission(deadline):
            if deadline is not None:
                # Leave a minute to drain jobs before the native admission expires.
                probes.deadline = min(probes.deadline,
                    time.monotonic() + max(0, deadline - time.time() - 60))
        schedule_admission(admission_deadline)
        log('pool_startup', wallet=mask_wallet(wallet),
            shape=dataclasses.asdict(shape), memory_budget_bytes=budget,
            memory_estimate_bytes=estimate, probe_key=native.probe_key)
        while not stop.is_set():
            if supervisor.done():
                supervisor.result()
                break
            if probes.due:
                if admission_refresh is not None:
                    try:
                        admission_deadline = await asyncio.to_thread(
                            admission_refresh, force=True, cancelled=stop.is_set)
                    except InterruptedError:
                        if stop.is_set():
                            break
                        raise
                if stop.is_set():
                    break
                await asyncio.to_thread(probes.refresh, native)
                schedule_admission(admission_deadline)
            source = client.latest
            if source is None:
                await asyncio.sleep(.05)
                continue
            if pipeline.source is None or pipeline.source.template_identity != source.template_identity:
                delay = 1.0 - (time.monotonic() - last_template_build)
                if delay > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), delay)
                    except asyncio.TimeoutError:
                        pass
                    continue  # Re-read the newest notify after the throttle.
                last_template_build = time.monotonic()
            await asyncio.to_thread(pipeline.set_template, source)
            async def lane(index):
                nonlocal reserved_jobs
                while current(source) and client.latest == source and not probes.due:
                    if max_jobs:
                        if reserved_jobs >= max_jobs:
                            break
                        reserved_jobs += 1
                    record = await pipeline.run(index, bits_to_target(source.share_nbits),
                                                source.share_nbits, submit, current)
                    if max_jobs and record.cancelled and not getattr(record, 'work_logged', False):
                        reserved_jobs = max(0, reserved_jobs - 1)
            results = await asyncio.gather(*(lane(i) for i in range(shape.slots)), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if max_jobs and reserved_jobs >= max_jobs and completed_jobs >= max_jobs:
                stop.set()
        if supervisor.done():
            supervisor.result()
    except BaseException as exc:
        fatal = exc
        if isinstance(exc, FatalDeviceError):
            gate_failures = 1
            state.save(halted=str(exc) if str(exc).startswith(('P0', 'pool rejected', 'pool stale')) else 'pool device safety failure')
        raise
    finally:
        stop.set()
        if pipeline:
            pipeline.cancel()
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        if timer:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
        telemetry.flush()
        if native:
            native.close()
        elapsed = max(time.monotonic() - started, 1e-9)
        lower, upper = poisson_interval(expected)
        rejected = sum(counts[k] for k in ('duplicate', 'low-difficulty', 'invalid'))
        summary = dict(accepted=counts['accepted'], stale=counts['stale'], rejected=rejected,
            duplicate=counts['duplicate'], low_difficulty=counts['low-difficulty'], invalid=counts['invalid'],
            transport=counts['transport'], timeout=counts['timeout'], submitted=submitted,
            gate_failures=gate_failures, completed_ops=completed_ops, completed_macs=completed_ops // 2,
            completed_jobs=completed_jobs, max_jobs=max_jobs,
            observed=observed, expected=expected, compact_expected=compact_expected,
            lower=lower, upper=upper, poisson_ok=lower <= observed <= upper,
            elapsed_seconds=elapsed, ops_per_second=completed_ops / elapsed,
            pool_hashrate=completed_ops / (2 * elapsed), block_candidates=blocks,
            failed=fatal is not None)
        state.save(pool_summary=summary)
        log('pool_summary', **summary)
        if not summary['poisson_ok']:
            log('alert', severity='health', reason='share_rate_poisson', expected=expected, observed=observed)
