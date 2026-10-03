import contextlib
import ctypes as C
from types import SimpleNamespace
import pytest
from pmk_miner.pipeline import (
    Record,
    Shape,
    VerifiedCandidate,
    collect_finds,
    FatalDeviceError,
    PoolSubmissionPolicy,
    verify_gate,
)
from pmk_miner.native import Slot, Result, Desc, Template, Seeds, Tile
from pmk_miner.native import NativeError
from pmk_miner.scheme import VERIFIER_FUNCTION, V3_SCHEME, scheme_for_cert_version


def test_abi_layouts():
    assert C.sizeof(Slot)==104
    assert C.sizeof(Template)==112
    assert C.sizeof(Seeds)==96
    assert C.sizeof(Tile)==112
    assert Desc.job_id.offset==192
    assert C.sizeof(Result)==72


def test_lifecycle_release_waits_for_finds():
    r=Record(1,None)
    r.move('gpu')
    with pytest.raises(RuntimeError): r.move('released')
    r.move('scanned'); r.pending_finds=1; r.move('proving')
    with pytest.raises(RuntimeError): r.move('released')
    r.pending_finds=0; r.move('submitted'); r.move('released')
    assert r.history==['building','gpu','scanned','proving','submitted','released']


def test_empty_job_release():
    r=Record(1,None); r.move('gpu'); r.move('scanned'); r.move('released')


def test_cancelled_builder_release():
    r=Record(1,None); r.cancelled=True; r.move('released')


def test_overflow_uses_retained_oracle():
    called=[]
    r=SimpleNamespace(overflow=3,block_count=2,share_count=3)
    expected=[(0,0,True,True),(1,1,True,True),(2,2,False,True)]
    assert collect_finds(r,lambda: called.append('oracle') or expected)==expected
    assert called==['oracle']


def test_overflow_count_mismatch_is_fatal():
    r=SimpleNamespace(overflow=1,block_count=2,share_count=0)
    with pytest.raises(FatalDeviceError): collect_finds(r,lambda:[(0,0,True,False)])


def test_overflow_recovery_is_bounded_before_oracle_scan():
    r=SimpleNamespace(overflow=1,block_count=321,share_count=0)
    with pytest.raises(FatalDeviceError,match='bounded work'):
        collect_finds(r,lambda:pytest.fail('oracle recovery should not run'),320)


def test_overflow_recovery_result_count_is_bounded():
    r=SimpleNamespace(overflow=1,block_count=1,share_count=0)
    found=[(i,i,i==0,False) for i in range(321)]
    with pytest.raises(FatalDeviceError,match='too many candidates'):
        collect_finds(r,lambda:found,320)


def test_independent_slot_arrays_deduplicated():
    blocks=(Slot*1)(); shares=(Slot*2)()
    blocks[0].t_rows=8; shares[0].t_rows=8; shares[1].t_cols=2
    r=Result(block_count=1,share_count=2,block_stored=1,share_stored=2,blocks=blocks,shares=shares)
    assert collect_finds(r,lambda:pytest.fail('unexpected recovery'))==[(0,2,False,True),(8,0,True,True)]


def test_truncated_without_overflow_fatal():
    with pytest.raises(FatalDeviceError): collect_finds(Result(block_count=1),lambda:[])

@pytest.mark.parametrize('shape',[Shape(k=1024),Shape(k=2112),Shape(k=16384),Shape(m=63),Shape(n=0),Shape(slots=1),Shape(m=2**24,n=2**24)])
def test_shape_policy(shape):
    with pytest.raises(ValueError): shape.validate(24*1024**3)


def test_default_memory_budget():
    assert Shape().validate(24*1024**3)<6*1024**3


def test_gate_rejects_rank_before_upstream():
    with pytest.raises(FatalDeviceError):
        verify_gate(b'',SimpleNamespace(noise_rank=64,k=4096),b'')


def test_gate_preserves_native_verifier_rejection_message_and_function():
    import asyncio
    import pearl_mining as pm

    native=FakeNative(asyncio.new_event_loop())
    try:
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        proof=SimpleNamespace(
            noise_rank=128,
            k=4096,
            a=SimpleNamespace(row_indices=[0,8,16,24]),
            bt=SimpleNamespace(row_indices=[0,1,8,9,16,17,24,25]),
        )
        def fake_invoke(function,*args,**kwargs):
            assert function is pm.verify_plain_proof_for_cert_version
            return False, "Jackpot condition not satisfied"

        assert V3_SCHEME.verify(bytes(header.to_bytes()),proof,invoke=fake_invoke) is False
        with pytest.raises(FatalDeviceError) as raised:
            verify_gate(bytes(header.to_bytes()),proof,native.config_bytes,invoke=fake_invoke)
        assert raised.value.gate == "verifier"
        assert raised.value.native_function == VERIFIER_FUNCTION
        assert raised.value.native_code == -2002
        assert raised.value.native_message == "v3 verifier gate failed: Jackpot condition not satisfied"
    finally:
        native.loop.close()


def test_only_v3_scheme_is_supported():
    assert scheme_for_cert_version(3) is V3_SCHEME
    with pytest.raises(ValueError,match='unsupported'):
        scheme_for_cert_version(4)


def test_v3_scheme_rejects_diagnostic_k():
    with pytest.raises(ValueError,match='production k'):
        V3_SCHEME.ensure_k(65536)


@pytest.mark.parametrize(
    "share_nbits",
    [0x1A07FFF8, 0x1A086373, 0x1B014F8A, 0x1B068DB2],
)
def test_v3_scheme_uses_native_extract_difficulty_bound_for_compact_nbits(share_nbits):
    import asyncio
    import pearl_mining as pm
    native=FakeNative(asyncio.new_event_loop())
    try:
        cfg = native.config_bytes
        target = int(pm.nbits_to_difficulty(share_nbits))
        assert V3_SCHEME.nbits_bound(share_nbits,cfg,target) == int(pm.extract_difficulty_bound(share_nbits,V3_SCHEME.parse_config(cfg)))
        with pytest.raises(ValueError,match="easier than pool target"):
            V3_SCHEME.nbits_bound(share_nbits,cfg,target - 1)
    finally:
        native.loop.close()


def test_pipeline_template_cert_guard_routes_through_scheme():
    import asyncio
    from pmk_miner.pipeline import Pipeline
    native=FakeNative(asyncio.new_event_loop())
    try:
        pipeline=Pipeline(native,Shape(128,128,4096,2),lambda *a,**kw:None)
        source=SimpleNamespace(cert_version=4,template_identity='bad',header=b'',target=1)
        with pytest.raises(ValueError,match='requires cert_version=3'):
            pipeline.set_template(source)
    finally:
        native.loop.close()


class FakeNative:
    def __init__(self,loop):
        import contextlib
        import pearl_mining as pm
        from pmk_miner.native import Seeds
        self.timing=__import__("threading").local(); self.loop=loop; self.context=C.c_void_p(1); self.completed=False; self.released=False; self.dispatched=False; self.last_desc=None
        self.seed=Seeds()
        self.config_bytes=bytes(pm.MiningConfiguration(4096,128,pm.MMAType.Int7xInt7ToInt32,
            pm.PeriodicPattern.from_list([0,8,16,24]),pm.PeriodicPattern.from_list([0,1,8,9,16,17,24,25]),None).to_bytes())
        self.metal=SimpleNamespace(pmk_run_job=self.dispatch)
    def config(self,*args): return self.config_bytes
    def alloc(self,size): return C.c_void_p(8)
    def template(self,*args): return object()
    def build(self,*args): return self.seed
    def measured(self,record):
        from contextlib import nullcontext
        return nullcontext()
    def foreign(self,fn,*args,**kwargs): return fn(*args,**kwargs)
    def call(self,fn,*args): return fn(*args)
    def dispatch(self,context,desc,callback,user,out):
        self.dispatched=True; self.last_desc=desc._obj; out._obj.value=123
        def finish():
            self.completed=True; callback(123,None)
        self.loop.call_soon_threadsafe(self.loop.call_later,.03,finish)
    def poll(self,handle):
        assert self.completed
        return Result(abi_version=1,gpu_start_time=1,gpu_end_time=1.01)
    def release(self,handle):
        assert self.completed, 'released before callback'
        self.released=True


class OpaqueNative(FakeNative):
    def __init__(self,loop):
        super().__init__(loop)
        self.config_bytes=b"opaque-non-v3-config"
        self.allocations=[]
        self.metal=SimpleNamespace(pmk_run_job_v9=self.dispatch)
    def config(self,*args):
        pytest.fail("opaque scheme must own config bytes without native v3 builder")
    def alloc(self,size):
        self.allocations.append(size)
        return C.c_void_p(8)


class OpaqueScheme:
    name = "opaque"
    cert_version = 9
    rank = 7
    config_bytes = b"opaque-non-v3-config"

    def __init__(self,fail_config=False,fail_proof=False):
        self.fail_config = fail_config
        self.fail_proof = fail_proof
        self.proofs = []

    def build_config(self,native,m,n,k):
        return self.config_bytes
    def validate_config(self,config):
        assert config == self.config_bytes
        if self.fail_config:
            raise ValueError("opaque config rejected")
        return {"opaque": True}
    def target_bound(self,target,config):
        assert config == self.config_bytes
        return target + 9
    def build_template(self,native,source,config,shape,a):
        assert config == self.config_bytes
        return object()
    def build_job(self,native,template,bt,shape):
        return native.seed
    @contextlib.contextmanager
    def oracle(self,native,source,config,shape,a,bt):
        assert config == self.config_bytes
        yield 99
    def build_proof(self,native,oracle,row,col):
        return f"opaque-proof:{row}:{col}"
    def kernel_entry(self,native):
        return native.metal.pmk_run_job_v9
    def dispatch_kernel(self,native,desc,callback,user,handle_out):
        native.call(self.kernel_entry(native),native.context,desc,callback,user,handle_out)
    def kernel_cert_version(self):
        return self.cert_version
    def validate_cert_version(self,cert_version):
        assert cert_version == self.cert_version
    def decode_proof(self,encoded,invoke=None):
        return {"encoded": encoded}
    def validate_proof(self,header,proof,config,share_nbits=None,invoke=None):
        assert config == self.config_bytes
        self.proofs.append((proof,share_nbits))
        if self.fail_proof:
            raise ValueError("opaque proof rejected")


def opaque_source(header: bytes, target: int, cert_version: int = 9):
    return SimpleNamespace(
        header=header,
        target=target,
        bits=0x177fd82e,
        cert_version=cert_version,
        template_identity=("opaque", header, target, cert_version),
    )


def test_cancellation_drains_gpu_before_release():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.transport import GatewayJob
    from pmk_miner.monitor import bits_to_target
    async def scenario():
        native=FakeNative(asyncio.get_running_loop())
        pipeline=Pipeline(native,Shape(128,128,4096,2),lambda *a,**kw:None)
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=GatewayJob(bytes(header.to_bytes()),bits_to_target(header.nbits),3)
        pipeline.set_template(source)
        async def submit(*args): pytest.fail('cancelled work submitted')
        task=asyncio.create_task(pipeline.run(0,source.target,source.bits,submit,lambda _:True))
        while not native.dispatched: await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert native.completed and native.released and not pipeline.records
    asyncio.run(scenario())


def test_cert_guard_cancellation_abandons_pending_work():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.transport import GatewayJob,FatalCertVersionError
    from pmk_miner.monitor import bits_to_target
    async def scenario():
        native=FakeNative(asyncio.get_running_loop())
        pipeline=Pipeline(native,Shape(128,128,4096,2),lambda *a,**kw:None)
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=GatewayJob(bytes(header.to_bytes()),bits_to_target(header.nbits),3)
        pipeline.set_template(source)
        async def submit(*args): pytest.fail('invalid cert work submitted')
        task=asyncio.create_task(pipeline.run(0,source.target,source.bits,submit,lambda _:True))
        while not native.dispatched: await asyncio.sleep(.001)
        with pytest.raises(FatalCertVersionError):
            try: GatewayJob(source.header,source.target,4)
            except FatalCertVersionError:
                pipeline.cancel(); raise
        record=await task
        assert record.cancelled and record.state=='released' and native.released
    asyncio.run(scenario())


def test_cancelled_native_worker_drains_before_return():
    import asyncio
    import threading
    from pmk_miner.pipeline import drain_thread
    async def scenario():
        started=threading.Event(); release=threading.Event(); finished=threading.Event()
        def native_work():
            started.set(); release.wait(); finished.set()
        task=asyncio.create_task(drain_thread(native_work))
        while not started.is_set(): await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.005)
        assert not task.done() and not finished.is_set()
        task.cancel()
        await asyncio.sleep(.005)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert finished.is_set()
    asyncio.run(scenario())


def test_same_template_after_disconnect_does_not_rehash_A():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.transport import GatewayJob
    from pmk_miner.monitor import bits_to_target
    async def scenario():
        native=FakeNative(asyncio.get_running_loop())
        calls=[]
        native.template=lambda *a:calls.append(a) or object()
        pipeline=Pipeline(native,Shape(128,128,4096,2),lambda *a,**kw:None)
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        job=GatewayJob(bytes(header.to_bytes()),bits_to_target(header.nbits),3)
        pipeline.set_template(job); pipeline.set_template(job)
        assert len(calls)==1
    asyncio.run(scenario())


def test_large_ram_does_not_bypass_native_oracle_limit():
    with pytest.raises(ValueError,match='oracle resource'):
        Shape(32768,32768,4096,2).validate(256*1024**3)


def test_proofs_stream_one_at_a_time_and_gate_precedes_submit():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.monitor import bits_to_target
    async def scenario():
        events=[]
        class RecordingScheme(OpaqueScheme):
            def build_config(self,native,m,n,k):
                events.append('scheme_config')
                return super().build_config(native,m,n,k)
            def validate_config(self,config):
                events.append('scheme_validate_config')
                return super().validate_config(config)
            def target_bound(self,target,config):
                events.append(f'scheme_bound{target}')
                return super().target_bound(target,config)
            def build_template(self,native,source,config,shape,a):
                events.append('scheme_template')
                return super().build_template(native,source,config,shape,a)
            def build_job(self,native,template,bt,shape):
                events.append('scheme_job')
                return super().build_job(native,template,bt,shape)
            def oracle(self,native,source,config,shape,a,bt):
                events.append('scheme_oracle')
                return super().oracle(native,source,config,shape,a,bt)
            def build_proof(self,native,oracle,row,col):
                events.append('scheme_proof')
                encoded=super().build_proof(native,oracle,row,col)
                events.append(f'build{row}')
                return encoded
            def kernel_cert_version(self): return 9
            def validate_cert_version(self,cert_version):
                events.append('scheme_cert')
                super().validate_cert_version(cert_version)
            def dispatch_kernel(self,native,desc,callback,user,handle_out):
                events.append('scheme_kernel')
                super().dispatch_kernel(native,desc,callback,user,handle_out)
            def decode_proof(self,encoded,invoke=None):
                events.append('scheme_decode')
                return super().decode_proof(encoded,invoke=invoke)
            def validate_proof(self,header,proof,config,share_nbits=None,invoke=None):
                events.append(f"gate{proof['encoded']}:{share_nbits}")
                return super().validate_proof(header,proof,config,share_nbits,invoke=invoke)
        scheme=RecordingScheme()
        native=OpaqueNative(asyncio.get_running_loop())
        from pmk_miner.pipeline import Pipeline
        pipeline=Pipeline(native,Shape(128,128,4096,2),lambda *a,**kw:None,scheme=scheme)
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=opaque_source(bytes(header.to_bytes()),bits_to_target(header.nbits))
        pipeline.set_template(source)
        slots=(Slot*3)()
        for i in range(3): slots[i].t_rows=i
        native.poll=lambda handle: Result(abi_version=1,block_count=3,block_stored=3,blocks=slots,gpu_start_time=1,gpu_end_time=1.01)
        async def submit(job,encoded): events.append(f'submit{encoded}')
        record=await pipeline.run(0,source.target,source.bits,submit,lambda _:True)
        assert events==[
            'scheme_config','scheme_validate_config','scheme_cert','scheme_template',
            f'scheme_bound{source.target}',f'scheme_bound{source.target}',
            'scheme_job','scheme_kernel',
            'scheme_oracle',
            'scheme_proof','build0','scheme_decode','gateopaque-proof:0:0:None','submitopaque-proof:0:0',
            'scheme_proof','build1','scheme_decode','gateopaque-proof:1:0:None','submitopaque-proof:1:0',
            'scheme_proof','build2','scheme_decode','gateopaque-proof:2:0:None','submitopaque-proof:2:0',
        ]
        assert native.last_desc.cert_version==9
        assert pipeline.config==b"opaque-non-v3-config"
        assert scheme.proofs==[
            ({"encoded":"opaque-proof:0:0"},None),
            ({"encoded":"opaque-proof:1:0"},None),
            ({"encoded":"opaque-proof:2:0"},None),
        ]
        assert native.released and record.pending_finds==0
    asyncio.run(scenario())


def test_submission_policy_receives_verified_block_and_share_candidates():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.monitor import bits_to_target

    class CapturePolicy:
        def __init__(self):
            self.verified = []
            self.submitted = []
        def verified_candidate(self, **kwargs):
            candidate = VerifiedCandidate(**kwargs)
            self.verified.append(candidate)
            return candidate
        async def submit(self, candidate, submit):
            self.submitted.append(candidate)
            if candidate.kind == "block":
                await submit(candidate.job, candidate.proof)

    async def scenario():
        native=OpaqueNative(asyncio.get_running_loop())
        policy=CapturePolicy()
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
        lambda event,**kw:None,
            scheme=OpaqueScheme(),
            submission_policy=policy,
        )
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=opaque_source(bytes(header.to_bytes()),bits_to_target(header.nbits))
        pipeline.set_template(source)
        block_slots=(Slot*1)()
        share_slots=(Slot*2)()
        block_slots[0].t_rows=0
        share_slots[0].t_rows=0
        share_slots[0].t_rows=8
        share_slots[1].t_rows=8
        native.poll=lambda handle: Result(
            abi_version=1,
            block_count=1,
            share_count=2,
            block_stored=1,
            share_stored=2,
            blocks=block_slots,
            shares=share_slots,
            gpu_start_time=1,
            gpu_end_time=1.01,
        )
        submitted=[]
        async def submit(job,encoded): submitted.append((job,encoded))
        await pipeline.run(0,source.target // 2,source.bits - 1,submit,lambda _:True)
        assert [(c.kind,c.job_id,c.target,c.share_nbits) for c in policy.verified] == [
            ("block",1,source.target,None),
            ("share",1,source.target // 2,source.bits - 1),
        ]
        assert [(c.kind,c.proof) for c in policy.submitted] == [
            ("block","opaque-proof:0:0"),
            ("share","opaque-proof:8:0"),
        ]
        assert policy.verified[0].job is source
        assert policy.verified[1].job is source
        assert submitted == [(source,"opaque-proof:0:0")]
    asyncio.run(scenario())



def test_same_identity_template_refreshes_pool_source_without_rebuilding_a():
    import asyncio
    from pmk_miner.pipeline import Pipeline

    events=[]
    class CountingScheme(OpaqueScheme):
        def build_template(self,native,source,config,shape,a):
            events.append(("template", source.pool_job_id, source.target, source.session_id))
            return super().build_template(native,source,config,shape,a)

    native=OpaqueNative(asyncio.new_event_loop())
    try:
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda *a,**kw:None,
            scheme=CountingScheme(),
            submission_policy=PoolSubmissionPolicy(),
        )
        identity="aa" * 76
        first=SimpleNamespace(
            header=bytes.fromhex(identity), target=1000, block_target=2000, bits=0x177fd82e,
            cert_version=9, template_identity=identity, pool_job_id="job-a",
            share_nbits=0x1A07FFF8, session_id="session-a",
        )
        second=SimpleNamespace(
            header=first.header, target=900, block_target=2000, bits=0x177fd82e,
            cert_version=9, template_identity=identity, pool_job_id="job-b",
            share_nbits=0x1A07FFF7, session_id="session-b",
        )
        pipeline.set_template(first)
        old_template=pipeline.template
        pipeline.set_template(second)
        assert pipeline.source is second
        assert pipeline.template is old_template
        assert events == [("template", "job-a", 1000, "session-a")]
    finally:
        native.loop.close()


def test_pool_nbits_bound_failure_releases_record_without_dispatch():
    import asyncio
    from pmk_miner.pipeline import Pipeline

    async def scenario():
        logs=[]
        class FailingBoundScheme(OpaqueScheme):
            def nbits_bound(self,nbits,config,target=None):
                raise ValueError("native bound unavailable token=supersecret wallet=prl1qqqqqqqqqqqqqq")

        native=OpaqueNative(asyncio.get_running_loop())
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda event,**kw:logs.append((event,kw)),
            scheme=FailingBoundScheme(),
            submission_policy=PoolSubmissionPolicy(),
        )
        source=SimpleNamespace(
            header=b"h" * 76, target=1000, block_target=2000, bits=0x177fd82e,
            cert_version=9, template_identity="h" * 152, pool_job_id="job-a",
            share_nbits=0x1A07FFF8, session_id="session-a",
        )
        pipeline.set_template(source)
        submitted=[]
        async def submit(job,encoded): submitted.append((job,encoded))
        with pytest.raises(FatalDeviceError,match="native bound unavailable"):
            await pipeline.run(0,source.target,source.share_nbits,submit,lambda _:True)
        assert submitted == []
        assert pipeline.stopped
        assert not native.dispatched
        assert not pipeline.records
        alerts=[kw for event,kw in logs if event == "alert"]
        assert alerts == [{
            "severity": "P0",
            "reason": "device_or_verifier_gate",
            "job_id": 1,
            "gate": "device",
            "native_function": "Scheme.nbits_bound",
            "native_code": -2001,
            "error_message": "native bound unavailable token=<redacted> wallet=<redacted>",
        }]
        assert "supersecret" not in repr(alerts)
        assert "prl1qqqq" not in repr(alerts)
    asyncio.run(scenario())


def test_native_dispatch_failure_logs_device_gate_metadata():
    import asyncio
    from pmk_miner.pipeline import Pipeline

    async def scenario():
        logs=[]
        class FailingDispatchScheme(OpaqueScheme):
            def nbits_bound(self,nbits,config,target=None):
                return 777
            def dispatch_kernel(self,native,desc,callback,user,handle_out):
                raise NativeError("pmk_run_job", -104, "command buffer status=4: GPU fault")

        native=OpaqueNative(asyncio.get_running_loop())
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda event,**kw:logs.append((event,kw)),
            scheme=FailingDispatchScheme(),
            submission_policy=PoolSubmissionPolicy(),
        )
        source=SimpleNamespace(
            header=b"h" * 76, target=1000, block_target=2000, bits=0x177fd82e,
            cert_version=9, template_identity="h" * 152, pool_job_id="job-a",
            share_nbits=0x1A07FFF8, session_id="session-a",
        )
        pipeline.set_template(source)
        async def submit(job,encoded): pytest.fail("failed dispatch must not submit")
        with pytest.raises(FatalDeviceError) as raised:
            await pipeline.run(0,source.target,source.share_nbits,submit,lambda _:True)
        assert raised.value.gate == "device"
        assert raised.value.native_function == "pmk_run_job"
        assert raised.value.native_code == -104
        alerts=[kw for event,kw in logs if event == "alert"]
        assert alerts == [{
            "severity": "P0",
            "reason": "device_or_verifier_gate",
            "job_id": 1,
            "gate": "device",
            "native_function": "pmk_run_job",
            "native_code": -104,
            "error_message": "command buffer status=4: GPU fault",
        }]
    asyncio.run(scenario())


def test_malformed_pool_proof_decode_is_local_gate_failure_before_submit():
    import asyncio
    from pmk_miner.pipeline import Pipeline

    async def scenario():
        logs=[]
        class MalformedProofScheme(OpaqueScheme):
            def nbits_bound(self,nbits,config,target=None):
                return 777
            def decode_proof(self,encoded,invoke=None):
                raise ValueError("bad bincode")
            def validate_proof(self,header,proof,config,share_nbits=None,invoke=None):
                pytest.fail("malformed proof must fail before verifier call")

        native=OpaqueNative(asyncio.get_running_loop())
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda event,**kw:logs.append((event,kw)),
            scheme=MalformedProofScheme(),
            submission_policy=PoolSubmissionPolicy(),
        )
        source=SimpleNamespace(
            header=b"h" * 76, target=1000, block_target=2000, bits=0x177fd82e,
            cert_version=9, template_identity="h" * 152, pool_job_id="job-a",
            share_nbits=0x1A07FFF8, session_id="session-a",
        )
        pipeline.set_template(source)
        slots=(Slot*1)()
        native.poll=lambda handle: Result(
            abi_version=1,
            share_count=1,
            share_stored=1,
            shares=slots,
            gpu_start_time=1,
            gpu_end_time=1.01,
        )
        submitted=[]
        async def submit(job,encoded): submitted.append((job,encoded))
        with pytest.raises(FatalDeviceError,match="v3 verifier gate failed: malformed proof"):
            await pipeline.run(0,source.target,source.share_nbits,submit,lambda _:True)
        assert submitted == []
        assert pipeline.stopped
        assert native.released
        assert not pipeline.records
        assert [event for event,_ in logs].count("work_completed") == 1
        assert any(event == "alert" and kw["reason"] == "device_or_verifier_gate" for event,kw in logs)
    asyncio.run(scenario())

def test_pool_submission_policy_gates_and_submits_block_candidates_as_shares():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.monitor import bits_to_target

    async def scenario():
        events=[]
        class PoolScheme(OpaqueScheme):
            def target_bound(self,target,config):
                events.append(f"block_bound:{target}")
                return target + 9
            def nbits_bound(self,nbits,config,target=None):
                events.append(f"share_nbits_bound:{nbits}")
                return 777
            def validate_proof(self,header,proof,config,share_nbits=None,invoke=None):
                events.append(f"gate:{proof['encoded']}:{share_nbits}")
                return super().validate_proof(header,proof,config,share_nbits,invoke=invoke)

        native=OpaqueNative(asyncio.get_running_loop())
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda *a,**kw:None,
            scheme=PoolScheme(),
            submission_policy=PoolSubmissionPolicy(),
        )
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=SimpleNamespace(
            header=bytes(header.to_bytes()),
            target=999,
            block_target=bits_to_target(header.nbits),
            bits=header.nbits,
            cert_version=9,
            template_identity=bytes(header.to_bytes()).hex(),
            pool_job_id="00000001_1000",
            share_nbits=0x1A07FFF8,
            session_id="session-a",
        )
        pipeline.set_template(source)
        block_slots=(Slot*1)()
        share_slots=(Slot*2)()
        block_slots[0].t_rows=0
        share_slots[0].t_rows=0
        share_slots[1].t_rows=8
        native.poll=lambda handle: Result(
            abi_version=1,
            block_count=1,
            share_count=2,
            block_stored=1,
            share_stored=2,
            blocks=block_slots,
            shares=share_slots,
            gpu_start_time=1,
            gpu_end_time=1.01,
        )
        submitted=[]
        async def submit(job,encoded): submitted.append((job,encoded))
        record=await pipeline.run(0,source.target,source.share_nbits,submit,lambda _:True)
        assert events[:2] == [f"block_bound:{source.block_target}", f"share_nbits_bound:{source.share_nbits}"]
        assert events[-2:] == [
            f"gate:opaque-proof:0:0:{source.share_nbits}",
            f"gate:opaque-proof:8:0:{source.share_nbits}",
        ]
        assert submitted == [(source,"opaque-proof:0:0"), (source,"opaque-proof:8:0")]
        assert record.completed_ops == Shape(128,128,4096,2).ops
        assert record.block_count == 1 and record.share_count == 2 and record.gpu_seconds > 0
    asyncio.run(scenario())


def test_pool_block_only_find_is_classified_but_not_submitted_as_share():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.monitor import bits_to_target

    async def scenario():
        logs=[]
        class PoolScheme(OpaqueScheme):
            def nbits_bound(self,nbits,config,target=None):
                return 777
            def validate_proof(self,header,proof,config,share_nbits=None,invoke=None):
                pytest.fail("block-only pool find must not be share-gated")

        native=OpaqueNative(asyncio.get_running_loop())
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda event,**kw:logs.append((event,kw)),
            scheme=PoolScheme(),
            submission_policy=PoolSubmissionPolicy(),
        )
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=SimpleNamespace(
            header=bytes(header.to_bytes()),
            target=999,
            block_target=bits_to_target(header.nbits),
            bits=header.nbits,
            cert_version=9,
            template_identity=bytes(header.to_bytes()).hex(),
            pool_job_id="00000001_1000",
            share_nbits=0x1A07FFF8,
            session_id="session-a",
        )
        pipeline.set_template(source)
        block_slots=(Slot*1)()
        block_slots[0].t_rows=0
        submitted=[]
        native.poll=lambda handle: Result(
            abi_version=1,
            block_count=1,
            share_count=0,
            block_stored=1,
            share_stored=0,
            blocks=block_slots,
            gpu_start_time=1,
            gpu_end_time=1.01,
        )
        async def submit(job,encoded): submitted.append((job,encoded))
        await pipeline.run(0,source.target,source.share_nbits,submit,lambda _:True)
        assert submitted == []
        classifications=[kw for event,kw in logs if event=="candidate_classified"]
        assert classifications == [{
            "job_id": 1,
            "kind": "block-candidate",
            "is_share": False,
            "is_block": True,
            "share_nbits": source.share_nbits,
        }]
    asyncio.run(scenario())


def test_scheme_config_failure_stops_before_allocation():
    import asyncio
    from pmk_miner.pipeline import Pipeline

    native=OpaqueNative(asyncio.new_event_loop())
    try:
        with pytest.raises(FatalDeviceError,match="opaque config rejected"):
            Pipeline(
                native,
                Shape(128,128,4096,2),
                lambda *a,**kw:None,
                scheme=OpaqueScheme(fail_config=True),
            )
        assert native.allocations==[]
        assert not native.dispatched
    finally:
        native.loop.close()


def test_scheme_proof_failure_cancels_and_does_not_submit():
    import asyncio
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.monitor import bits_to_target

    async def scenario():
        scheme=OpaqueScheme(fail_proof=True)
        native=OpaqueNative(asyncio.get_running_loop())
        logs=[]
        pipeline=Pipeline(
            native,
            Shape(128,128,4096,2),
            lambda event,**kw:logs.append((event,kw)),
            scheme=scheme,
            submission_policy=PoolSubmissionPolicy(),
        )
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=opaque_source(bytes(header.to_bytes()),bits_to_target(header.nbits))
        pipeline.set_template(source)
        slots=(Slot*1)()
        native.poll=lambda handle: Result(
            abi_version=1,
            share_count=1,
            share_stored=1,
            shares=slots,
            gpu_start_time=1,
            gpu_end_time=1.01,
        )
        submitted=[]
        async def submit(job,encoded): submitted.append((job,encoded))
        with pytest.raises(FatalDeviceError,match="opaque proof rejected"):
            await pipeline.run(0,source.target,source.bits,submit,lambda _:True)
        assert submitted==[]
        assert pipeline.stopped
        assert native.released
        assert not pipeline.records
        assert scheme.proofs==[({"encoded":"opaque-proof:0:0"},source.bits)]
        work=[kw for event,kw in logs if event=="work_completed"]
        assert len(work)==1
        assert work[0]["ops"] == Shape(128,128,4096,2).ops
        assert work[0]["blocks"] == 0 and work[0]["shares"] == 1
    asyncio.run(scenario())


def test_malformed_merkle_proof_rejected_before_submit():
    import asyncio
    import contextlib
    import pearl_mining as pm
    from pmk_miner.pipeline import Pipeline
    from pmk_miner.transport import GatewayJob
    from pmk_miner.monitor import bits_to_target

    rows=[0,8,16,24]
    cols=[0,1,8,9,16,17,24,25]
    leaf=b'\x00'*1024
    bad_merkle=pm.MerkleProof([leaf],[0],b'\x11'*32,[],1)
    matrix_a=pm.MatrixMerkleProof(bad_merkle,rows)
    matrix_b=pm.MatrixMerkleProof(bad_merkle,cols)
    malformed=pm.PlainProof(128,128,4096,128,matrix_a,matrix_b,None)

    async def scenario():
        native=FakeNative(asyncio.get_running_loop())
        pipeline=Pipeline(native,Shape(128,128,4096,2),lambda *a,**kw:None)
        header=pm.IncompleteBlockHeader(1,bytes(32),bytes(32),1,0x177fd82e)
        source=GatewayJob(bytes(header.to_bytes()),bits_to_target(header.nbits),3)
        pipeline.set_template(source)
        slot=(Slot*1)()
        native.poll=lambda handle: Result(abi_version=1,block_count=1,block_stored=1,blocks=slot,gpu_start_time=1,gpu_end_time=1.01)
        @contextlib.contextmanager
        def oracle(*args):
            yield 9
        native.oracle=oracle
        native.proof=lambda handle,row,col: malformed.to_base64()
        submitted=[]
        async def submit(job,encoded): submitted.append(encoded)
        with pytest.raises(FatalDeviceError,match='verifier gate'):
            await pipeline.run(0,source.target,source.bits,submit,lambda _:True)
        assert submitted==[]
        assert native.last_desc.cert_version==3
        assert native.released
    asyncio.run(scenario())


@pytest.mark.parametrize("slots", [2, 3])
def test_production_result_capacity_and_memory(slots):
    shape = Shape(slots=slots)
    assert shape.result_capacity == 2_097_152
    assert (shape.result_capacity + 8) * C.sizeof(Slot) < 2_000_000_000
    assert shape.validate(24 * 1024**3) <= 24 * 1024**3 // 4

def test_result_buffer_limit_checked_before_allocation():
    with pytest.raises(ValueError, match="2 GB"):
        Shape(m=32768,n=32768,k=2048).validate(1024**4)


def test_production_dispatch_has_capacity_for_every_find():
    import asyncio
    from pmk_miner.pipeline import Pipeline
    async def scenario():
        native = FakeNative(asyncio.get_running_loop())
        pipeline = Pipeline(native, Shape(), lambda *a, **kw: None)
        pipeline.source = SimpleNamespace(target=0)
        pipeline.template = object()
        await pipeline.run(0, 0, None, None, lambda _: True)
        assert native.last_desc.block_capacity == 2_097_152
        assert native.last_desc.share_capacity == 2_097_152
    asyncio.run(scenario())


def test_cancellation_during_dispatch_retains_gpu_ownership():
    import asyncio
    import threading
    from pmk_miner.pipeline import Pipeline
    async def scenario():
        started, unblock = threading.Event(), threading.Event()
        native = FakeNative(asyncio.get_running_loop())
        original = native.metal.pmk_run_job
        def dispatch(*args):
            started.set()
            assert unblock.wait(5)
            original(*args)
        native.metal.pmk_run_job = dispatch
        pipeline = Pipeline(native, Shape(128,128,4096,2), lambda *a, **kw: None)
        pipeline.source = SimpleNamespace(target=0)
        pipeline.template = object()
        task = asyncio.create_task(pipeline.run(0, 0, None, None, lambda _: True))
        while not started.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done() and not native.released
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert native.completed and native.released
        assert pipeline.inflight == 0 and not pipeline.records
    asyncio.run(scenario())


def test_battery_transition_finishes_inflight_then_pauses_next_dispatch():
    import asyncio
    from pmk_miner.desktop import DesktopControls
    from pmk_miner.pipeline import Pipeline
    async def scenario():
        source = ['ac']
        desktop = DesktopControls(lambda: source[0], intensity=50, poll_seconds=0)
        native = FakeNative(asyncio.get_running_loop())
        pipeline = Pipeline(native, Shape(128,128,4096,2), lambda *a, **kw: None,
                            desktop=desktop)
        pipeline.source = SimpleNamespace(target=0)
        pipeline.template = object()
        first = asyncio.create_task(pipeline.run(0,0,None,None,lambda _:True))
        while not native.dispatched:
            await asyncio.sleep(.001)
        source[0] = 'battery'
        desktop.refresh()
        record = await first
        assert not record.cancelled and native.released and record.completed_ops > 0
        native.dispatched = False
        second = asyncio.create_task(pipeline.run(0,0,None,None,lambda _:True))
        await asyncio.sleep(.05)
        assert not native.dispatched and not second.done()
        source[0] = 'ac'
        record = await asyncio.wait_for(second,1)
        assert native.dispatched and not record.cancelled
        assert not desktop._gate.locked()
    asyncio.run(scenario())


def test_intensity_gate_is_released_when_native_dispatch_fails():
    import asyncio
    from pmk_miner.desktop import DesktopControls
    from pmk_miner.pipeline import Pipeline
    async def scenario():
        desktop = DesktopControls(lambda:'ac', intensity=50)
        native = FakeNative(asyncio.get_running_loop())
        def fail(*args): raise RuntimeError('dispatch failed')
        native.metal.pmk_run_job = fail
        pipeline = Pipeline(native, Shape(128,128,4096,2), lambda *a, **kw:None,
                            desktop=desktop)
        pipeline.source = SimpleNamespace(target=0)
        pipeline.template = object()
        with pytest.raises(RuntimeError,match='dispatch failed'):
            await pipeline.run(0,0,None,None,lambda _:True)
        assert not desktop._gate.locked() and not pipeline.records
    asyncio.run(scenario())
