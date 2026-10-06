# SPDX-License-Identifier: Apache-2.0
"""Typed C ABIs. Only metadata and proof bytes cross into Python."""
from __future__ import annotations
import base64
import ctypes as C
import hashlib
import json
import os
import pathlib
import time
from contextlib import contextmanager
import threading

U8 = C.c_uint8
U16 = C.c_uint16
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

class V4OperandDesc(C.Structure):
    _fields_ = [
        ('clean_values', P(C.c_int8)), ('clean_value_count', U64),
        ('noise_e_codes', P(U8)), ('noise_e_count', U64),
        ('noise_f_codes', P(U8)), ('noise_f_count', U64),
        ('alpha_bf16', P(U16)), ('beta_bf16', P(U16)), ('scale_count', U64),
    ]
class V4Desc(C.Structure):
    _fields_ = [(x,U32) for x in ('abi_version','m','n','k','r')] + [
        ('a', V4OperandDesc), ('bt', V4OperandDesc),
        ('jackpot_key', U32*8), ('block_bound', U32*8), ('share_bound', U32*8),
        ('block_capacity', U32), ('share_capacity', U32), ('job_id', U64),
    ]
class V4Stats(C.Structure):
    _fields_ = [('abi_version',U32),('flags',U32)] + [
        (x,U64) for x in ('fallback_groups','total_groups','quantized_a','quantized_b',
                          'quant_saturated_a','quant_saturated_b','quant_nan_a','quant_nan_b')
    ] + [('layout_failures',U32),('fallback_alert',U32)]
class V4Result(C.Structure):
    _fields_ = [('abi_version',U32),('status',C.c_int32),('job_id',U64)] + [
        (x,U32) for x in ('block_count','share_count','block_stored','share_stored',
                          'overflow','recovered')
    ] + [('blocks',P(Slot)),('shares',P(Slot)),('stats',V4Stats),
         ('c_bits',P(U32)),('c_count',U64),('gpu_start_time',C.c_double),('gpu_end_time',C.c_double)]
class V4GpuJobDesc(C.Structure):
    _fields_ = [(x,U32) for x in ('m','n','k','rank','tile_rows','tile_cols','row_period','col_period')] + [
        ('a_values', P(C.c_int8)), ('a_scales', P(U16)), ('bt_values', P(C.c_int8)), ('bt_scales', P(U16)),
        ('a_noised', P(U8)), ('bt_noised', P(U8)),
        ('a_noise_e', P(U8)), ('a_noise_f', P(U8)), ('bt_noise_e', P(U8)), ('bt_noise_f', P(U8)),
        ('a_alpha', P(U16)), ('a_beta', P(U16)), ('a_l2', P(U16)),
        ('bt_alpha', P(U16)), ('bt_beta', P(U16)), ('bt_l2', P(U16)),
    ] + [(x,U8*32) for x in ('key_a','key_b','hash_a','hash_b','noise_seed_a','noise_seed_b','jackpot_key')]
class V4TileResult(C.Structure):
    _fields_ = [('t_rows',U32),('t_cols',U32),('message',U8*64),('hash',U8*32),
                ('policy_pass',U32),('is_share',U32),('is_block',U32)]

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
_CONTEXT_DIAGNOSTIC_FUNCTIONS = {'pmk_buffer_alloc', 'pmk_run_job', 'pmk_run_job_na', 'pmk_run_job_diagnostic'}

class Native:
    def __init__(self, core=None, metal=None, kernel=None):
        if kernel is not None:
            os.environ["PMK_KERNEL"] = str(kernel)
        self.requested_v3_kernel = kernel
        self.core_path = pathlib.Path(core or ROOT/'pmkcore/target/release/libpmkcore.dylib')
        self.metal_path = pathlib.Path(metal or ROOT/'libpmk/.build/release/libpmk.dylib')
        self.core = C.CDLL(str(self.core_path))
        self.metal = C.CDLL(str(self.metal_path))
        self.v4_core = None
        self.v4_context = None
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
        self._declare_optional_v3_symbols()
        self.metal.pmk_destroy.argtypes=[PTR]; self.metal.pmk_destroy.restype=None
        self.context=PTR(); self.buffers=[]
        error=C.create_string_buffer(4096)
        self.call(self.metal.pmk_init,C.byref(self.context),error,len(error),error_buffer=error)
        key=C.create_string_buffer(128)
        self.call(self.metal.pmk_probe,self.context,key,len(key))
        self.probe_key=key.value.decode()

    def _declare_optional(self, name, args):
        try:
            fn = getattr(self.metal, name)
        except AttributeError:
            return None
        fn.argtypes = args
        fn.restype = C.c_int32
        return fn

    def _declare_optional_v3_symbols(self):
        self._declare_optional('pmk_run_job_na',[PTR,P(Desc),CALLBACK,PTR,P(PTR)])
        self._declare_optional('pmk_v3_kernel_metadata',[PTR,PTR,U64])
        self._declare_optional('pmk_kernel_metadata',[PTR,PTR,U64])

    def v3_kernel_metadata(self):
        for name in ('pmk_v3_kernel_metadata','pmk_kernel_metadata'):
            fn = getattr(self.metal,name,None)
            if fn is None:
                continue
            out = C.create_string_buffer(8192)
            self.call(fn,self.context,out,len(out))
            try:
                return json.loads(out.value.decode('utf-8'))
            except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                raise ValueError('malformed native v3 kernel metadata') from exc
        return None

    def validate_v3_kernel(self, scheme):
        expected = getattr(scheme,'kernel_id','sg')
        if expected == 'na' and not hasattr(self.metal,'pmk_run_job_na'):
            raise ValueError('K3-NA requested but loaded libpmk exposes no pmk_run_job_na')
        metadata = self.v3_kernel_metadata()
        if metadata is None:
            if expected == 'na':
                raise ValueError('K3-NA requested but loaded libpmk exposes no kernel metadata')
            return None
        from .kernel import device_class_generation, metadata_kernel
        actual = metadata_kernel(metadata)
        if actual is not None and actual != expected:
            raise ValueError(f"native v3 kernel mismatch: expected {expected}, got {actual}")
        device_class = str(metadata.get('device_class',''))
        generation = device_class_generation(device_class)
        if expected == 'na' and generation is not None and generation < 10:
            raise ValueError(f"K3-NA requires Apple10+, got {device_class}")
        return metadata

    def _check_v4_library_admission(self, admission):
        if not isinstance(admission, dict):
            raise ValueError("v4 G3 admission metadata required for native v4 init")
        expected = admission.get("library_sha256")
        try:
            actual = hashlib.sha256(self.metal_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("v4 G3 admission library_sha256 unreadable") from exc
        if actual != expected:
            raise ValueError("v4 G3 admission library_sha256 mismatch")
        return actual

    def _init_v4_core(self):
        rc=self.foreign(self.v4_core.pmkcore_v4_init,0)
        if rc not in (0, -9):
            message=self.error_message('pmkcore_v4_init',rc)
            raise NativeError('pmkcore_v4_init',rc,message)

    def ensure_v4(self, admission=None):
        if admission is not None:
            self.v4_admission_library_sha256 = self._check_v4_library_admission(admission)
        elif self.v4_core is None:
            raise ValueError("v4 G3 admission metadata required for native v4 init")
        if self.v4_core is None:
            self.v4_core = C.CDLL(str(ROOT/'pmkcore/v4/target/release/libpmkcore_v4.dylib'))
            declarations = {
                'pmkcore_v4_init':[U32],
                'pmkcore_v4_job_create_grid_b200':[PTR,PTR,PTR,U64,U32,U32,U32,P(PTR)],
                'pmkcore_v4_gpu_descriptor':[PTR,P(V4GpuJobDesc)],
                'pmkcore_v4_prepare_oracle_noised':[PTR],
                'pmkcore_v4_scan_cpu_oracle':[PTR,PTR,PTR,P(V4TileResult),U64,P(U64)],
                'pmkcore_v4_bound_for_nbits':[PTR,U32,PTR],
                'pmkcore_v4_build_plain_proof':[PTR,U32,U32,PTR,U64,P(U64)],
                'pmkcore_v4_verify_plain_proof':[PTR,PTR,U64,PTR,P(U8)],
            }
            for name,args in declarations.items():
                fn=getattr(self.v4_core,name); fn.argtypes=args; fn.restype=C.c_int32
            self.v4_core.pmkcore_v4_strerror.argtypes=[C.c_int32]
            self.v4_core.pmkcore_v4_strerror.restype=C.c_char_p
            self.v4_core.pmkcore_v4_job_free.argtypes=[PTR]
            self.v4_core.pmkcore_v4_job_free.restype=None
            self._init_v4_core()
            v4_declarations = {
                'pmk_v4_init':[P(PTR),PTR,U64],
                'pmk_v4_probe':[PTR,PTR,U64],
                'pmk_v4_run_job':[PTR,P(V4Desc),CALLBACK,PTR,P(PTR)],
                'pmk_v4_poll':[PTR,P(V4Result)],
                'pmk_v4_context_error':[PTR,PTR,U64],
                'pmk_v4_job_error':[PTR,PTR,U64],
                'pmk_v4_job_wait_callback':[PTR],
                'pmk_v4_job_release':[PTR],
            }
            for name,args in v4_declarations.items():
                fn=getattr(self.metal,name); fn.argtypes=args; fn.restype=C.c_int32
            self.metal.pmk_v4_destroy.argtypes=[PTR]
            self.metal.pmk_v4_destroy.restype=None
            self.v4_context=PTR()
            error=C.create_string_buffer(4096)
            try:
                self.call(self.metal.pmk_v4_init,C.byref(self.v4_context),error,len(error),error_buffer=error)
            except BaseException:
                self.v4_context=None
                self.v4_core=None
                raise
        return self.v4_core

    def foreign(self, fn, *args, **kwargs):
        start=time.thread_time()
        try:
            return fn(*args, **kwargs)
        finally:
            self.timing.native=getattr(self.timing,'native',0)+time.thread_time()-start

    def error_message(self, function, code, error_buffer=None):
        if error_buffer is not None and error_buffer.value:
            return error_buffer.value.decode('utf-8',errors='replace')
        if function.startswith('pmkcore_v4_') and self.v4_core is not None:
            message=self.foreign(self.v4_core.pmkcore_v4_strerror,code)
            if message:
                return message.decode('utf-8',errors='replace')
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

    def v4_job_error(self, handle):
        metal=getattr(self,'metal',None)
        if metal is None or not hasattr(metal,'pmk_v4_job_error'):
            return ''
        error=C.create_string_buffer(1024)
        rc=self.foreign(metal.pmk_v4_job_error,handle,error,len(error))
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

    def v4_create_job(self, header, ancestor_header, ancestor_chain, m, n, k):
        core=self.ensure_v4()
        handle=PTR()
        chain = buf(ancestor_chain) if ancestor_chain else None
        self.call(core.pmkcore_v4_job_create_grid_b200,buf(header),buf(ancestor_header),
                  chain,len(ancestor_chain),m,n,k,C.byref(handle))
        return handle

    def v4_gpu_descriptor(self, job):
        out=V4GpuJobDesc()
        self.call(self.v4_core.pmkcore_v4_gpu_descriptor,job,C.byref(out))
        return out

    def v4_prepare_oracle(self, job):
        self.call(self.v4_core.pmkcore_v4_prepare_oracle_noised,job)

    def v4_scan(self, job, share, block):
        count=U64(); sb=buf(share.to_bytes(32,'little')); bb=buf(block.to_bytes(32,'little'))
        self.call(self.v4_core.pmkcore_v4_scan_cpu_oracle,job,sb,bb,None,0,C.byref(count))
        tiles=(V4TileResult*count.value)()
        self.call(self.v4_core.pmkcore_v4_scan_cpu_oracle,job,sb,bb,tiles,len(tiles),C.byref(count))
        return [(t.t_rows,t.t_cols,bool(t.is_block),bool(t.is_share)) for t in tiles if t.is_block or t.is_share]

    def v4_proof(self, job, row, col):
        count=U64()
        fn=self.v4_core.pmkcore_v4_build_plain_proof
        self.call(fn,job,row,col,None,0,C.byref(count))
        out=(U8*count.value)(); self.call(fn,job,row,col,out,len(out),C.byref(count))
        return base64.b64encode(bytes(out)).decode()

    def v4_verify_plain_proof(self, header, proof, share_nbits=None):
        accepted=U8()
        nbits = None if share_nbits is None else (U8*4).from_buffer_copy(share_nbits.to_bytes(4,'little'))
        self.call(self.v4_core.pmkcore_v4_verify_plain_proof,buf(header),buf(proof),len(proof),nbits,C.byref(accepted))
        return bool(accepted.value)

    def v4_dispatch_desc(self, desc, callback, user, handle_out):
        self.call(self.metal.pmk_v4_run_job,self.v4_context,desc,callback,user,handle_out)

    def v4_poll(self, handle):
        result=V4Result(); self.call(self.metal.pmk_v4_poll,handle,C.byref(result))
        if result.status:
            raise NativeError('pmk_v4_poll',result.status,
                              self.v4_job_error(handle) or self.error_message('pmk_v4_poll',result.status))
        if result.abi_version != 1:
            raise NativeError('pmk_v4_poll',-101,'v4 result ABI version mismatch')
        return result

    def v4_release(self, handle):
        self.call(self.metal.pmk_v4_job_wait_callback,handle)
        self.call(self.metal.pmk_v4_job_release,handle)

    def v4_free_job(self, job):
        self.foreign(self.v4_core.pmkcore_v4_job_free,job)

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
        self.buffers.clear()
        if getattr(self,'v4_context',None):
            self.metal.pmk_v4_destroy(self.v4_context)
            self.v4_context=None
        self.metal.pmk_destroy(self.context)
