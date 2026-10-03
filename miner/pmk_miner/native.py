"""Typed C ABIs. Only metadata and proof bytes cross into Python."""
from __future__ import annotations
import base64
import ctypes as C
import pathlib
import time
from contextlib import contextmanager
import threading

U8 = C.c_uint8
U32 = C.c_uint32
U64 = C.c_uint64
PTR = C.c_void_p
P = C.POINTER
ROOT = pathlib.Path(__file__).resolve().parents[2]

class Template(C.Structure):
    _fields_ = [(x, U8 * 32) for x in ('job_key', 'raw_root_a', 'salted_root_a')] + [(x, U32) for x in ('m','n','k','reserved')]
class Seeds(C.Structure):
    _fields_ = [(x, U8 * 32) for x in ('raw_root_b','b_noise_seed','a_noise_seed')]
class Tile(C.Structure):
    _fields_ = [('t_rows',U32),('t_cols',U32),('transcript',U32*16),('hash',U8*32),('is_share',U32),('is_block',U32)]
class Slot(C.Structure):
    _fields_ = [('t_rows',U32),('t_cols',U32),('transcript',U32*16),('hash',U32*8)]
class Desc(C.Structure):
    _fields_ = [(x,U32) for x in ('abi_version','m','n','k')] + [('a',PTR),('bt',PTR),('a_bytes',U64),('bt_bytes',U64)] + [(x,U32*8) for x in ('a_seed','b_seed','block_bound','share_bound')] + [(x,U32) for x in ('block_capacity','share_capacity','cert_version','rank')] + [('job_id',U64)]
class Result(C.Structure):
    _fields_ = [('abi_version',U32),('status',C.c_int32),('job_id',U64)] + [(x,U32) for x in ('block_count','share_count','block_stored','share_stored','overflow','recovered')] + [('blocks',P(Slot)),('shares',P(Slot)),('gpu_start_time',C.c_double),('gpu_end_time',C.c_double)]
CALLBACK = C.CFUNCTYPE(None,PTR,PTR)

def buf(data):
    return (U8 * len(data)).from_buffer_copy(data)
def words(value):
    return (U32*8).from_buffer_copy(value if isinstance(value,bytes) else value.to_bytes(32,'little'))

class NativeError(RuntimeError):
    """A failed native ABI call with safe, structured diagnostic metadata."""

    def __init__(self, function, code, message):
        self.function = str(function)
        self.code = int(code)
        self.message = str(message)
        super().__init__(f'{self.function} failed ({self.code}): {self.message}')


_LIBPMK_ERRORS = {
    1: 'operation pending',
    -101: 'invalid argument or descriptor',
    -102: 'resource budget or allocation limit exceeded',
    -103: 'startup or health probe failed',
    -104: 'GPU execution failed',
    -105: 'operation busy',
}
_CONTEXT_DIAGNOSTIC_FUNCTIONS = {'pmk_buffer_alloc', 'pmk_run_job', 'pmk_run_job_diagnostic'}

class Native:
    def __init__(self, core=None, metal=None):
        self.core = C.CDLL(str(core or ROOT/'pmkcore/target/release/libpmkcore.dylib'))
        self.metal = C.CDLL(str(metal or ROOT/'libpmk/.build/release/libpmk.dylib'))
        self.timing = threading.local()
        declarations = {
            'pmkcore_build_config': [U32]*4+[PTR],
            'pmkcore_template_init':[PTR,PTR,U32,U32,PTR,U64,U8,P(Template)],
            'pmkcore_build_job':[P(Template),PTR,U64,P(Seeds)],
            'pmkcore_oracle_job_create':[PTR,PTR,U32,U32,PTR,U64,PTR,U64,P(PTR)],
            'pmkcore_oracle_scan':[PTR,PTR,PTR,P(Tile),U64,P(U64)],
            'pmkcore_oracle_build_plain_proof':[PTR,U32,U32,PTR,U64,P(U64)],
        }
        for name,args in declarations.items():
            fn=getattr(self.core,name); fn.argtypes=args; fn.restype=C.c_int32
        self.core.pmkcore_strerror.argtypes=[C.c_int32]
        self.core.pmkcore_strerror.restype=C.c_char_p
        self.core.pmkcore_oracle_job_free.argtypes=[PTR]
        self.core.pmkcore_oracle_job_free.restype=None
        declarations = {
            'pmk_init':[P(PTR),PTR,U64], 'pmk_probe':[PTR,PTR,U64],
            'pmk_probe_refresh':[PTR,PTR,U64],
            'pmk_buffer_alloc':[PTR,U64,P(PTR)], 'pmk_buffer_release':[PTR,PTR],
            'pmk_run_job':[PTR,P(Desc),CALLBACK,PTR,P(PTR)],
            'pmk_poll':[PTR,P(Result)], 'pmk_context_error':[PTR,PTR,U64],
            'pmk_job_error':[PTR,PTR,U64], 'pmk_job_wait_callback':[PTR],
            'pmk_job_release':[PTR],
        }
        for name,args in declarations.items():
            fn=getattr(self.metal,name); fn.argtypes=args; fn.restype=C.c_int32
        self.metal.pmk_destroy.argtypes=[PTR]; self.metal.pmk_destroy.restype=None
        self.context=PTR(); self.buffers=[]
        error=C.create_string_buffer(4096)
        self.call(self.metal.pmk_init,C.byref(self.context),error,len(error),error_buffer=error)
        key=C.create_string_buffer(128)
        self.call(self.metal.pmk_probe,self.context,key,len(key))
        self.probe_key=key.value.decode()

    def foreign(self, fn, *args, **kwargs):
        start=time.thread_time()
        try:
            return fn(*args, **kwargs)
        finally:
            self.timing.native=getattr(self.timing,'native',0)+time.thread_time()-start

    def error_message(self, function, code, error_buffer=None):
        if error_buffer is not None and error_buffer.value:
            return error_buffer.value.decode('utf-8',errors='replace')
        if function.startswith('pmkcore_'):
            message=self.foreign(self.core.pmkcore_strerror,code)
            if message:
                return message.decode('utf-8',errors='replace')
        return _LIBPMK_ERRORS.get(code,'unknown native error')

    def context_error(self):
        metal=getattr(self,'metal',None)
        if metal is None or not hasattr(metal,'pmk_context_error') or not getattr(self,'context',None):
            return ''
        error=C.create_string_buffer(1024)
        rc=self.foreign(metal.pmk_context_error,self.context,error,len(error))
        return error.value.decode('utf-8',errors='replace') if rc == 0 else ''

    def job_error(self, handle):
        metal=getattr(self,'metal',None)
        if metal is None or not hasattr(metal,'pmk_job_error'):
            return ''
        error=C.create_string_buffer(1024)
        rc=self.foreign(metal.pmk_job_error,handle,error,len(error))
        return error.value.decode('utf-8',errors='replace') if rc == 0 else ''

    def call(self, fn, *args, error_buffer=None):
        rc=self.foreign(fn,*args)
        if rc != 0:
            function=fn.__name__
            message=self.error_message(function,rc,error_buffer)
            if function in _CONTEXT_DIAGNOSTIC_FUNCTIONS and not (error_buffer is not None and error_buffer.value):
                message=self.context_error() or message
            raise NativeError(function,rc,message)
        return rc

    @contextmanager
    def measured(self, record):
        start=time.thread_time(); native=getattr(self.timing,'native',0)
        try:
            yield
        finally:
            record.python_seconds += max(0,time.thread_time()-start-(getattr(self.timing,'native',0)-native))

    def config(self,pattern,m,n,k):
        out=(U8*52)(); self.call(self.core.pmkcore_build_config,pattern,k,m,n,out)
        return bytes(out)

    def alloc(self,size):
        ptr=PTR(); self.call(self.metal.pmk_buffer_alloc,self.context,size,C.byref(ptr))
        self.buffers.append(ptr); return ptr

    def refresh_probe(self):
        error=C.create_string_buffer(4096)
        self.call(self.metal.pmk_probe_refresh,self.context,error,len(error),error_buffer=error)
        key=C.create_string_buffer(128)
        self.call(self.metal.pmk_probe,self.context,key,len(key))
        self.probe_key=key.value.decode()
        return self.probe_key

    def template(self,header,config,m,n,a,k):
        out=Template()
        self.call(self.core.pmkcore_template_init,buf(header),buf(config),m,n,a,m*k,1,C.byref(out))
        return out

    def build(self,template,bt,n,k):
        out=Seeds(); self.call(self.core.pmkcore_build_job,C.byref(template),bt,n*k,C.byref(out))
        return out

    @contextmanager
    def oracle(self,header,config,m,n,k,a,bt):
        handle=PTR()
        self.call(self.core.pmkcore_oracle_job_create,buf(header),buf(config),m,n,a,m*k,bt,n*k,C.byref(handle))
        try:
            yield handle
        finally:
            self.foreign(self.core.pmkcore_oracle_job_free,handle)

    def scan(self,oracle,share,block):
        count=U64(); sb=buf(share.to_bytes(32,'little')); bb=buf(block.to_bytes(32,'little'))
        self.call(self.core.pmkcore_oracle_scan,oracle,sb,bb,None,0,C.byref(count))
        tiles=(Tile*count.value)()
        self.call(self.core.pmkcore_oracle_scan,oracle,sb,bb,tiles,len(tiles),C.byref(count))
        return [(t.t_rows,t.t_cols,bool(t.is_block),bool(t.is_share)) for t in tiles if t.is_block or t.is_share]

    def proof(self,oracle,row,col):
        count=U64()
        fn=self.core.pmkcore_oracle_build_plain_proof
        self.call(fn,oracle,row,col,None,0,C.byref(count))
        out=(U8*count.value)(); self.call(fn,oracle,row,col,out,len(out),C.byref(count))
        return base64.b64encode(bytes(out)).decode()

    def poll(self,handle):
        result=Result(); self.call(self.metal.pmk_poll,handle,C.byref(result))
        if result.status:
            raise NativeError('pmk_poll',result.status,
                              self.job_error(handle) or self.error_message('pmk_poll',result.status))
        if result.abi_version != 1:
            raise NativeError('pmk_poll',-101,'result ABI version mismatch')
        return result

    def release(self,handle):
        # The asyncio future may run as soon as the callback publishes its
        # completion, before the ctypes callback trampoline has returned.
        self.call(self.metal.pmk_job_wait_callback,handle)
        self.call(self.metal.pmk_job_release,handle)

    def close(self):
        for ptr in self.buffers:
            self.call(self.metal.pmk_buffer_release,self.context,ptr)
        self.buffers.clear(); self.metal.pmk_destroy(self.context)
