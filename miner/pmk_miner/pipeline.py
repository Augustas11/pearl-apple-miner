# SPDX-License-Identifier: Apache-2.0
"""Callback-driven slot ownership; the sole release owner is the job coroutine."""
from __future__ import annotations
import asyncio
import ctypes as C
from dataclasses import dataclass, field
import time
import subprocess
from .native import CALLBACK, Desc, PTR, NativeError, words
from .scheme import V3_SCHEME, scheme_for_cert_version
from .transport import redact_credentials

class FatalDeviceError(RuntimeError):
    """Fail-closed device/verifier gate error with safe structured metadata."""

    DEVICE_GATE_FUNCTION = "python.device_gate"
    VERIFIER_GATE_FUNCTION = "python.verifier_gate"
    DEVICE_GATE_CODE = -2001
    VERIFIER_GATE_CODE = -2002

    def __init__(self, message, *, gate="device", native_error=None, function=None, code=None):
        self.gate = gate
        self.native_error = native_error
        if native_error is not None:
            self.native_function = native_error.function
            self.native_code = native_error.code
            self.native_message = native_error.message
        else:
            self.native_function = function or (
                self.VERIFIER_GATE_FUNCTION if gate == "verifier" else self.DEVICE_GATE_FUNCTION
            )
            self.native_code = int(code if code is not None else (
                self.VERIFIER_GATE_CODE if gate == "verifier" else self.DEVICE_GATE_CODE
            ))
            self.native_message = str(message)
        super().__init__(str(message))

    @classmethod
    def verifier(cls, message):
        return cls(message, gate="verifier")

    @classmethod
    def verifier_at(cls, function, message):
        return cls(message, gate="verifier", function=function, code=cls.VERIFIER_GATE_CODE)

    @classmethod
    def device(cls, message, *, function=None):
        return cls(message, gate="device", function=function, code=cls.DEVICE_GATE_CODE)

    @classmethod
    def from_native(cls, exc):
        return cls(exc.message, gate="device", native_error=exc)


def _fatal_fields(exc):
    if not isinstance(exc, FatalDeviceError):
        return {}
    return {
        "gate": exc.gate,
        "native_function": exc.native_function,
        "native_code": exc.native_code,
        "error_message": redact_credentials(exc.native_message)[:256],
    }

async def drain_thread(function):
    """Repeated cancellation cannot let native work outlive its buffer owner."""
    task = asyncio.create_task(asyncio.to_thread(function))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result

class MeteredAwaitable:
    """Measure only coroutine execution slices, excluding suspended await time.

    Native calls subtract their calling-thread CPU; native build/proof worker
    threads are measured separately. Other asyncio tasks cannot be charged to
    this job while it is suspended.
    """
    def __init__(self, coroutine, native, record):
        self.coroutine, self.native, self.record = coroutine, native, record

    def __await__(self):
        iterator = self.coroutine.__await__()
        value = None
        error = None
        while True:
            started = time.thread_time()
            native_before = getattr(self.native.timing, 'native', 0)
            try:
                if error is None:
                    waiting = iterator.send(value)
                else:
                    waiting = iterator.throw(error)
            except StopIteration as done:
                return done.value
            finally:
                self.record.python_seconds += max(0, time.thread_time() - started -
                    (getattr(self.native.timing, 'native', 0) - native_before))
            try:
                value = yield waiting
                error = None
            except BaseException as exc:
                error = exc

@dataclass
class Record:
    job_id: int
    source: object
    cfg: bytes | None = None
    share_nbits: int | None = None
    state: str = 'building'
    python_seconds: float = 0.0
    started: float = field(default_factory=time.monotonic)
    history: list = field(default_factory=lambda:['building'])
    pending_finds: int = 0
    handle: object = None
    cancelled: bool = False
    stage_seconds: dict = field(default_factory=dict)
    completed_ops: int = 0
    share_count: int = 0
    block_count: int = 0
    gpu_seconds: float = 0.0
    work_logged: bool = False

    def move(self,state):
        allowed={'building':{'gpu','released'},'gpu':{'scanned'},'scanned':{'proving','released'},'proving':{'submitted','released'},'submitted':{'released'}}
        if state not in allowed.get(self.state,set()):
            raise RuntimeError(f'illegal lifecycle {self.state}->{state}')
        if state=='released' and self.pending_finds:
            raise RuntimeError('finds still own this slot')
        self.state=state; self.history.append(state)

@dataclass(frozen=True)
class Shape:
    m: int = 8192
    n: int = 8192
    k: int = 4096
    slots: int = 2

    def validate(self,ram=None):
        if self.slots not in (2,3) or not (2048<=self.k<=8192) or self.k%128:
            raise ValueError('v1 requires 2–3 slots, rank 128, k in [2048,8192] divisible by 128')
        if any(x<=0 or x>2**24 or x%64 for x in (self.m,self.n)):
            raise ValueError('SG dimensions must be positive multiples of 64 <= 2^24')
        return self.memory_estimate(ram)

    def memory_estimate(self,ram=None):
        if self.slots not in (2,3):
            raise ValueError('requires 2–3 retained slots')
        if any(type(x) is not int or x<=0 or x>2**24 for x in (self.m,self.n,self.k)):
            raise ValueError('shape dimensions must be positive integers <= 2^24')
        raw=(self.m+self.n)*self.k
        if 6*raw > 1024**3:
            raise ValueError('shape exceeds retained pmkcore oracle resource limit')
        if max(max(self.m,self.n)*self.k, (self.result_capacity+8)*104)>=2_000_000_000:
            raise ValueError('buffer exceeds 2 GB boundary')
        # Conservative simultaneous GPU recovery + raw buffers + one CPU oracle
        # per slot, including both native result streams, Python find merging/sorting,
        # Merkle trees and allocation slack (even if every tile is a find).
        self_estimate=self.slots*(12*raw+512*self.m*self.n//32)+64*1024**2
        if ram is None:
            ram=int(subprocess.check_output(['sysctl','-n','hw.memsize']))
        if self_estimate>ram//4:
            raise ValueError('combined pmk memory estimate exceeds 25% RAM')
        return self_estimate

    @property
    def result_capacity(self):
        # A tile may satisfy either bound, including every tile on easy regtest.
        # Production shapes exceed libpmk's bounded CPU overflow recovery.
        return max(64, self.m*self.n//32)

    @property
    def ops(self):
        return 2*self.m*self.n*self.k


@dataclass(frozen=True)
class VerifiedCandidate:
    kind: str
    job: object
    proof: str
    job_id: int
    target: int
    share_nbits: int | None = None
    cfg: bytes | None = None


class SoloSubmissionPolicy:
    submit_all_shares = False

    def verified_candidate(
        self,
        *,
        kind: str,
        job: object,
        proof: str,
        job_id: int,
        target: int,
        share_nbits: int | None,
        cfg: bytes | None = None,
    ) -> VerifiedCandidate:
        if kind not in ("block", "share"):
            raise FatalDeviceError.verifier("unknown verified candidate kind")
        return VerifiedCandidate(kind, job, proof, job_id, target, share_nbits, cfg)

    async def submit(self, candidate: VerifiedCandidate, submit) -> None:
        if candidate.kind == "block":
            await submit(candidate.job, candidate.proof)


class PoolSubmissionPolicy:
    """Pool mode submits every locally verified share under its source job id."""

    submit_all_shares = True

    def verified_candidate(
        self,
        *,
        kind: str,
        job: object,
        proof: str,
        job_id: int,
        target: int,
        share_nbits: int | None,
        cfg: bytes | None = None,
    ) -> VerifiedCandidate:
        if kind not in ("share", "block-candidate"):
            raise FatalDeviceError.verifier("unknown verified pool candidate kind")
        if share_nbits is None:
            raise FatalDeviceError.verifier("pool candidate missing share_nbits")
        return VerifiedCandidate(kind, job, proof, job_id, target, share_nbits, cfg)

    async def submit(self, candidate: VerifiedCandidate, submit) -> None:
        await submit(candidate.job, candidate.proof)


def collect_finds(result,recover,max_recovery_finds=320):
    if result.overflow:
        if result.block_count + result.share_count > max_recovery_finds:
            raise FatalDeviceError.device('overflow recovery exceeds bounded work limit')
        found=recover()
        if len(found)>max_recovery_finds:
            raise FatalDeviceError.device('overflow recovery returned too many candidates')
        if sum(x[2] for x in found)!=result.block_count or sum(x[3] for x in found)!=result.share_count:
            raise FatalDeviceError.device('overflow oracle disagrees with GPU counters')
        return found
    if result.block_stored!=result.block_count or result.share_stored!=result.share_count:
        raise FatalDeviceError.device('truncated result without overflow')
    merged={}
    for arr,count,index in ((result.blocks,result.block_stored,0),(result.shares,result.share_stored,1)):
        for i in range(count):
            row=arr[i]; key=(row.t_rows,row.t_cols)
            flags=merged.setdefault(key,[False,False]); flags[index]=True
    return [(r,c,*merged[(r,c)]) for r,c in sorted(merged)]


def _v4_stats_fields(result, scheme=None):
    if getattr(scheme,'cert_version',None) != 4:
        return {}
    stats = getattr(result,'stats',None)
    if stats is None:
        return {}
    def value(name, default=0):
        try:
            return int(getattr(stats,name,default))
        except (TypeError,ValueError):
            return int(default)
    fallback = value('fallback_groups')
    total = value('total_groups')
    rate = (fallback / total) if total > 0 else 0.0
    return {
        'fallback_per_job': fallback,
        'fallback_groups': fallback,
        'total_groups': total,
        'fallback_rate': rate,
        'fallback_alert': bool(value('fallback_alert')),
        'layout_failures': value('layout_failures'),
        'quantized_a': value('quantized_a'),
        'quantized_b': value('quantized_b'),
        'quant_saturated_a': value('quant_saturated_a'),
        'quant_saturated_b': value('quant_saturated_b'),
        'quant_nan_a': value('quant_nan_a'),
        'quant_nan_b': value('quant_nan_b'),
    }


def verify_gate(header,proof,config,share_nbits=None,invoke=None,scheme=V3_SCHEME,native=None):
    try:
        if native is not None and hasattr(scheme,'validate_proof_native'):
            scheme.validate_proof_native(native,header,proof,config,share_nbits)
        else:
            scheme.validate_proof(header,proof,config,share_nbits,invoke=invoke)
    except ValueError as exc:
        raise FatalDeviceError.verifier_at(
            getattr(exc,'function','Scheme.validate_proof'),str(exc)) from exc

class Pipeline:
    def __init__(self,native,shape,log,scheme=V3_SCHEME,max_gpu_seconds=0.4,submission_policy=None,desktop=None):
        self.native=native; self.shape=shape; self.log=log; self.scheme=scheme
        self.submission_policy=submission_policy or SoloSubmissionPolicy()
        self.max_gpu_seconds=max_gpu_seconds
        self.desktop=desktop
        try:
            self.config=self._build_config(scheme)
        except ValueError as exc:
            if scheme is not V3_SCHEME:
                raise FatalDeviceError.device(str(exc),function='Scheme.validate_config') from exc
            self.config=None
        self.a=native.alloc(shape.m*shape.k)
        self.b=[native.alloc(shape.n*shape.k) for _ in range(shape.slots)]
        self.template=None; self.source=None; self.records={}; self.sequence=0
        self.stopped=False; self.gpu_rate=None; self.inflight=0

    def _validate_shape_for_scheme(self,scheme):
        if scheme is V3_SCHEME:
            self.shape.validate()
        elif hasattr(scheme,'validate_shape'):
            scheme.validate_shape(self.shape)

    def _build_config(self,scheme):
        config=scheme.build_config(self.native,self.shape.m,self.shape.n,self.shape.k)
        scheme.validate_config(config)
        return config

    def set_template(self,source):
        if hasattr(source,'cert_version'):
            if getattr(self.scheme,'cert_version',None) == source.cert_version:
                next_scheme=self.scheme
            else:
                next_scheme=scheme_for_cert_version(source.cert_version)
            self._validate_shape_for_scheme(next_scheme)
            if hasattr(next_scheme,'admit_source'):
                next_scheme.admit_source(source,self.native)
            else:
                next_scheme.validate_cert_version(source.cert_version)
        else:
            next_scheme=self.scheme
            self._validate_shape_for_scheme(next_scheme)
        if self.records:
            raise RuntimeError('cannot replace A while retained jobs exist')
        if (self.source is not None and self.source.template_identity == source.template_identity
                and self.scheme is next_scheme and self.config is not None):
            self.source=source
            return
        if self.scheme is not next_scheme or self.config is None:
            try:
                config=self._build_config(next_scheme)
            except ValueError as exc:
                raise FatalDeviceError.device(str(exc),function='Scheme.validate_config') from exc
            self.scheme=next_scheme
            self.config=config
            self.gpu_rate=None
        s=self.shape
        self.template=next_scheme.build_template(self.native,source,self.config,s,self.a)
        self.source=source

    def bounds(self,target):
        try:
            return self.scheme.target_bound(target,self.config)
        except ValueError as exc:
            raise FatalDeviceError.device(str(exc),function='Scheme.target_bound') from exc

    def nbits_bounds(self,nbits,target=None):
        try:
            if not hasattr(self.scheme,'nbits_bound'):
                raise AttributeError
            return self.scheme.nbits_bound(nbits,self.config,target)
        except AttributeError:
            if target is None:
                raise FatalDeviceError.device('scheme lacks compact nbits bound support',function='Scheme.nbits_bound')
            return self.bounds(target)
        except ValueError as exc:
            raise FatalDeviceError.device(str(exc),function='Scheme.nbits_bound') from exc

    def cancel(self):
        self.stopped=True
        for record in self.records.values():
            record.cancelled=True

    def record_completed_work(self,record,result):
        record.share_count=result.share_count
        record.block_count=result.block_count
        record.gpu_seconds=max(0.0,result.gpu_end_time-result.gpu_start_time)
        record.completed_ops=self.shape.ops
        if not record.work_logged and getattr(self.submission_policy,'submit_all_shares',False):
            source=record.source
            self.log('work_completed',
                job_id=record.job_id,
                ops=record.completed_ops,
                shares=record.share_count,
                blocks=record.block_count,
                gpu_seconds=record.gpu_seconds,
                share_nbits=record.share_nbits,
                target=getattr(source,'target',None),
                block_target=getattr(source,'block_target',getattr(source,'target',None)),
                template=getattr(source,'template_identity',None))
            record.work_logged=True

    async def run(self,index,share_target,share_nbits,submit,is_current):
        self.sequence+=1
        record=Record(self.sequence,self.source,cfg=self.config,share_nbits=share_nbits)
        result = await MeteredAwaitable(
            self._run(index,record,share_target,share_nbits,submit,is_current),self.native,record)
        if not record.cancelled:
            wall=time.monotonic()-record.started
            self.log('python_overhead',job_id=record.job_id,python_seconds=record.python_seconds,
                wall_seconds=wall,python_overhead_pct=100*record.python_seconds/wall,stages=record.stage_seconds)
        return result

    async def _run(self,index,record,share_target,share_nbits,submit,is_current):
        native=self.native; s=self.shape; scheme=self.scheme
        loop=asyncio.get_running_loop(); completed=loop.create_future()
        pool_submit = bool(getattr(self.submission_policy,'submit_all_shares',False))
        block = share = None
        def mark(_job,_user):
            start=time.thread_time()
            loop.call_soon_threadsafe(completed.set_result,None)
            record.python_seconds+=time.thread_time()-start
        callback=CALLBACK(mark)
        result=None
        gpu_counted=False
        desktop_owned=False
        try:
            self.records[index]=record
            block=self.bounds(getattr(record.source,'block_target',record.source.target))
            share=(self.nbits_bounds(share_nbits,share_target)
                if pool_submit and share_nbits is not None else self.bounds(share_target))
            def prepare():
                with native.measured(record):
                    cpu_start=time.thread_time()
                    native_start=time.monotonic()
                    try:
                        return scheme.build_job(native,self.template,self.b[index],s)
                    finally:
                        record.stage_seconds['build_cpu']=time.thread_time()-cpu_start
                        record.stage_seconds['build_native']=time.monotonic()-native_start
            stage_start=time.monotonic()
            job=await drain_thread(prepare)
            record.stage_seconds['build']=time.monotonic()-stage_start
            stage_start=time.monotonic()
            if self.stopped or not is_current(record.source):
                record.cancelled=True
                return record
            if self.max_gpu_seconds is not None and self.gpu_rate and s.ops/self.gpu_rate>self.max_gpu_seconds:
                raise FatalDeviceError.device('predicted command buffer exceeds 400 ms; reduce shape')
            if hasattr(scheme,'job_descriptor'):
                desc=scheme.job_descriptor(native,record.source,job,s,self.a,self.b[index],block,share,record.job_id)
            else:
                desc=Desc(abi_version=1,m=s.m,n=s.n,k=s.k,a=self.a,bt=self.b[index],
                    a_bytes=s.m*s.k,bt_bytes=s.n*s.k,
                    a_seed=words(bytes(job.a_noise_seed)),b_seed=words(bytes(job.b_noise_seed)),
                    block_bound=words(block),share_bound=words(share),
                    block_capacity=s.result_capacity,share_capacity=s.result_capacity,
                    cert_version=scheme.kernel_cert_version(),rank=scheme.rank,job_id=record.job_id)
            if self.desktop:
                desktop_owned=await self.desktop.before_dispatch(
                    lambda: self.stopped or not is_current(record.source))
                if not desktop_owned:
                    record.cancelled=True
                    return record
            handle=PTR()
            def dispatch():
                with native.measured(record):
                    scheme.dispatch_kernel(native,C.byref(desc),callback,None,C.byref(handle))
            try:
                await drain_thread(dispatch)
            finally:
                # Cancellation can arrive while dispatch commits a buffer. Publish
                # ownership after the worker drains, before propagating cancellation.
                if handle:
                    record.handle=handle; record.move('gpu')
                    self.inflight+=1
                    gpu_counted=True
                    self.log('gpu_dispatch',job_id=record.job_id,host_time=time.monotonic(),inflight=self.inflight)
            record.stage_seconds['dispatch']=time.monotonic()-stage_start
            stage_start=time.monotonic()
            await asyncio.shield(completed)
            record.stage_seconds['gpu_wait']=time.monotonic()-stage_start
            self.inflight-=1
            gpu_counted=False
            stage_start=time.monotonic()
            result=scheme.poll_kernel(native,record.handle) if hasattr(scheme,'poll_kernel') else native.poll(record.handle)
            self.record_completed_work(record,result)
            if desktop_owned:
                self.desktop.after_gpu(result.gpu_start_time,result.gpu_end_time)
                desktop_owned=False
            record.move('scanned')
            gpu=result.gpu_end_time-result.gpu_start_time
            if gpu>0:
                self.gpu_rate=s.ops/gpu if self.gpu_rate is None else min(self.gpu_rate,s.ops/gpu)
            record.cancelled |= self.stopped or not is_current(record.source)
            record.stage_seconds['scan']=time.monotonic()-stage_start
            if record.cancelled:
                return record
            # Retain one native oracle and at most one serialized proof. A full
            # easy-target result must not create an unbounded Python proof list.
            retained = {}
            proof_seconds = 0.0
            submit_seconds = 0.0
            def open_oracle():
                with native.measured(record):
                    oracle_operand = (scheme.oracle_operand(job,self.b[index])
                        if hasattr(scheme,'oracle_operand') else self.b[index])
                    context = scheme.oracle(native,record.source,self.config,s,self.a,oracle_operand)
                    retained['context'] = context
                    retained['oracle'] = context.__enter__()
            def close_oracle():
                with native.measured(record):
                    retained['context'].__exit__(None,None,None)
            try:
                if result.block_count or result.share_count:
                    stage_start=time.monotonic()
                    await drain_thread(open_oracle)
                    def scan_finds():
                        with native.measured(record):
                            recovery_limit = desc.block_capacity + desc.share_capacity
                            if hasattr(scheme,'scan_oracle'):
                                scan = lambda: scheme.scan_oracle(native,retained['oracle'],share,block)
                            else:
                                scan = lambda: native.scan(retained['oracle'],share,block)
                            return collect_finds(result,scan,recovery_limit)
                    found=await drain_thread(scan_finds)
                    proof_seconds += time.monotonic()-stage_start
                    record.pending_finds=len(found); record.move('proving')
                    for row,col,is_block,is_share in found:
                        try:
                            if record.cancelled or self.stopped or not is_current(record.source):
                                continue
                            if pool_submit:
                                kind = 'block-candidate' if is_block else 'share'
                                self.log('candidate_classified',job_id=record.job_id,kind=kind,
                                    is_share=bool(is_share),is_block=bool(is_block),share_nbits=share_nbits)
                                if not is_share:
                                    continue
                                candidate_target = share_target
                                gate_nbits = share_nbits
                            else:
                                kind = 'block' if is_block else 'share'
                                candidate_target = record.source.target if is_block else share_target
                                gate_nbits = None if is_block else share_nbits
                            def build_one():
                                with native.measured(record):
                                    encoded=scheme.build_proof(native,retained['oracle'],row,col)
                                    try:
                                        proof=self.scheme.decode_proof(encoded,invoke=native.foreign)
                                    except (ValueError,TypeError) as exc:
                                        raise FatalDeviceError.verifier_at('Scheme.decode_proof','v3 verifier gate failed: malformed proof') from exc
                                    verify_gate(record.source.header,proof,self.config,
                                        gate_nbits,invoke=native.foreign,scheme=self.scheme,native=native)
                                    return self.submission_policy.verified_candidate(
                                        kind=kind,
                                        job=record.source,
                                        proof=encoded,
                                        job_id=record.job_id,
                                        target=candidate_target,
                                        share_nbits=gate_nbits,
                                        cfg=self.config,
                                    )
                            stage_start=time.monotonic()
                            candidate=await drain_thread(build_one)
                            proof_seconds += time.monotonic()-stage_start
                            stage_start=time.monotonic()
                            if not self.stopped and is_current(record.source):
                                await self.submission_policy.submit(candidate,submit)
                            elif is_block:
                                self.log('outcome',classification='stale',job_id=record.job_id)
                            submit_seconds += time.monotonic()-stage_start
                        finally:
                            record.pending_finds-=1
            finally:
                # drain_thread populates retained even if cancellation arrives
                # during native construction; the oracle is then freed here.
                if 'oracle' in retained:
                    stage_start=time.monotonic()
                    await drain_thread(close_oracle)
                    proof_seconds += time.monotonic()-stage_start
            record.stage_seconds['proof']=proof_seconds
            record.stage_seconds['submit']=submit_seconds
            if record.state=='proving':
                record.move('submitted')
            v4_stats = _v4_stats_fields(result,scheme)
            if v4_stats and s.k <= 4096 and v4_stats['fallback_rate'] > 0.01:
                self.log('v4_fallback_warning',job_id=record.job_id,severity='warning',
                    k=s.k,fallback_rate=v4_stats['fallback_rate'],
                    fallback_per_job=v4_stats['fallback_per_job'],total_groups=v4_stats['total_groups'])
            self.log('completed',job_id=record.job_id,ops=s.ops,shares=result.share_count,blocks=result.block_count,
                overflow=bool(result.overflow),gpu_seconds=gpu,
                gpu_start_time=result.gpu_start_time,gpu_end_time=result.gpu_end_time,
                wall_seconds=time.monotonic()-record.started,**v4_stats)
            return record
        except BaseException as exc:
            self.cancel()
            if isinstance(exc,NativeError):
                exc = FatalDeviceError.from_native(exc)
            if isinstance(exc,FatalDeviceError):
                self.log('alert',severity='P0',reason='device_or_verifier_gate',
                    job_id=record.job_id,**_fatal_fields(exc))
            raise exc
        finally:
            # Metal has no cancellation ABI: cancel eligibility immediately, drain callback,
            # then abandon proofs. Never free a buffer still owned by a command buffer.
            if record.handle:
                while not completed.done():
                    try:
                        await asyncio.shield(completed)
                    except asyncio.CancelledError:
                        continue
                if result is None:
                    try:
                        result=scheme.poll_kernel(native,record.handle) if hasattr(scheme,'poll_kernel') else native.poll(record.handle)
                        self.record_completed_work(record,result)
                    except Exception:
                        pass
                if record.state=='gpu':
                    record.move('scanned')
                record.pending_finds=0
                if hasattr(scheme,'release_kernel'):
                    scheme.release_kernel(native,record.handle)
                else:
                    native.release(record.handle)
            if 'job' in locals():
                try:
                    if hasattr(scheme,'release_job'):
                        scheme.release_job(native,job)
                except Exception:
                    pass
            if desktop_owned:
                if result is not None:
                    self.desktop.after_gpu(result.gpu_start_time,result.gpu_end_time)
                else:
                    self.desktop.abort_dispatch()
            if gpu_counted:
                self.inflight-=1
            record.move('released')
            self.records.pop(index,None)
