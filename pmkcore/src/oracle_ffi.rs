//! B1 C ABI wrappers for the Pearl oracle.

use crate::oracle::{self, OracleJob, Pattern, TileResult};
use crate::{PmkError, CONFIG_LEN, HEADER_LEN};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

pub enum PmkOracleJob {}

use crate::oracle::MAX_ORACLE_BYTES;

fn ffi_code(f: impl FnOnce() -> Result<(), PmkError>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => e as i32,
        Err(_) => PmkError::Panic as i32,
    }
}

fn pattern_from_id(pattern: u32) -> Result<Pattern, PmkError> {
    match pattern {
        0 => Ok(Pattern::Na),
        1 => Ok(Pattern::Sg),
        _ => Err(PmkError::Policy),
    }
}

fn len_to_usize(len: u64) -> Result<usize, PmkError> {
    let len = usize::try_from(len).map_err(|_| PmkError::ResourceLimit)?;
    if len > isize::MAX as usize {
        return Err(PmkError::ResourceLimit);
    }
    Ok(len)
}

fn bounded_input_len(len: u64) -> Result<usize, PmkError> {
    let len = len_to_usize(len)?;
    if len > MAX_ORACLE_BYTES {
        return Err(PmkError::ResourceLimit);
    }
    Ok(len)
}

fn bounded_output_cap(cap: u64, elem_size: usize) -> Result<usize, PmkError> {
    let cap = len_to_usize(cap)?;
    cap.checked_mul(elem_size)
        .filter(|&bytes| bytes <= MAX_ORACLE_BYTES)
        .ok_or(PmkError::ResourceLimit)?;
    Ok(cap)
}

fn checked_result_bytes(count: usize) -> Result<(), PmkError> {
    count
        .checked_mul(std::mem::size_of::<TileResult>())
        .filter(|&bytes| bytes <= MAX_ORACLE_BYTES)
        .map(|_| ())
        .ok_or(PmkError::ResourceLimit)
}

unsafe fn bytes_from_raw<'a>(ptr: *const u8, len: u64) -> Result<&'a [u8], PmkError> {
    if ptr.is_null() {
        return Err(PmkError::NullPointer);
    }
    Ok(std::slice::from_raw_parts(ptr, bounded_input_len(len)?))
}

unsafe fn array_from_raw<'a, const N: usize>(ptr: *const u8) -> Result<&'a [u8; N], PmkError> {
    if ptr.is_null() {
        return Err(PmkError::NullPointer);
    }
    Ok(&*(ptr as *const [u8; N]))
}

unsafe fn write_bytes(
    bytes: &[u8],
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> Result<(), PmkError> {
    if out_len.is_null() {
        return Err(PmkError::NullPointer);
    }
    *out_len = bytes.len() as u64;
    if out.is_null() {
        return if out_cap == 0 {
            Ok(())
        } else {
            Err(PmkError::NullPointer)
        };
    }
    if bounded_output_cap(out_cap, 1)? < bytes.len() {
        return Err(PmkError::BufferTooSmall);
    }
    ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
    Ok(())
}

unsafe fn job_ref<'a>(job: *const PmkOracleJob) -> Result<&'a OracleJob, PmkError> {
    if job.is_null() {
        return Err(PmkError::NullPointer);
    }
    Ok(&*(job as *const OracleJob))
}

/// Build a SPEC v0.3 v1 MiningConfiguration.
///
/// `pattern` is 0 for NA and 1 for SG.
/// # Safety
/// `out_config` must point to 52 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_build_config(
    pattern: u32,
    k: u32,
    m: u32,
    n: u32,
    out_config: *mut u8,
) -> i32 {
    ffi_code(|| {
        if out_config.is_null() {
            return Err(PmkError::NullPointer);
        }
        let cfg = oracle::build_config(pattern_from_id(pattern)?, k, m, n)?;
        let bytes = cfg.to_bytes();
        ptr::copy_nonoverlapping(bytes.as_ptr(), out_config, CONFIG_LEN);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn pmkcore_build_config_diagnostic(
    pattern: u32,
    k: u32,
    m: u32,
    n: u32,
    out_config: *mut u8,
) -> i32 {
    ffi_code(|| {
        if out_config.is_null() {
            return Err(PmkError::NullPointer);
        }
        let cfg = oracle::build_config_diagnostic(pattern_from_id(pattern)?, k, m, n)?;
        let bytes = cfg.to_bytes();
        ptr::copy_nonoverlapping(bytes.as_ptr(), out_config, CONFIG_LEN);
        Ok(())
    })
}

/// Create an owned oracle job handle from caller-owned header/config/A/B^T bytes.
/// # Safety
/// Header/config point to 76/52 readable bytes; A/B point to their stated readable lengths.
/// `out_job` points to a writable handle slot disjoint from the input buffers.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_oracle_job_create(
    header: *const u8,
    config: *const u8,
    m: u32,
    n: u32,
    a: *const u8,
    a_len: u64,
    bt: *const u8,
    bt_len: u64,
    out_job: *mut *mut PmkOracleJob,
) -> i32 {
    ffi_code(|| {
        if out_job.is_null() {
            return Err(PmkError::NullPointer);
        }
        *out_job = ptr::null_mut();
        let header = array_from_raw::<HEADER_LEN>(header)?;
        let config = array_from_raw::<CONFIG_LEN>(config)?;
        let (expected_a, expected_bt) = oracle::matrix_lens(config, m, n)?;
        if bounded_input_len(a_len)? != expected_a || bounded_input_len(bt_len)? != expected_bt {
            return Err(PmkError::BadShape);
        }
        let a = bytes_from_raw(a, a_len)?;
        let bt = bytes_from_raw(bt, bt_len)?;
        let job = OracleJob::new(header, config, m, n, a, bt)?;
        *out_job = Box::into_raw(Box::new(job)) as *mut PmkOracleJob;
        Ok(())
    })
}

/// Free an oracle job handle returned by pmkcore_oracle_job_create.
/// # Safety
/// `job` is null or a live handle returned by create, never previously freed. No concurrent calls.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_oracle_job_free(job: *mut PmkOracleJob) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !job.is_null() {
            drop(Box::from_raw(job as *mut OracleJob));
        }
    }));
}

/// Evaluate one Pearl tile and write its transcript/hash/classification.
/// # Safety
/// `job` is live; bounds point to 32 readable bytes each; `out` points to a writable,
/// aligned TileResult disjoint from all inputs. The handle must not be freed during this call.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_oracle_tile(
    job: *const PmkOracleJob,
    t_rows: u32,
    t_cols: u32,
    share_bound: *const u8,
    block_bound: *const u8,
    out: *mut TileResult,
) -> i32 {
    ffi_code(|| {
        if out.is_null() {
            return Err(PmkError::NullPointer);
        }
        let job = job_ref(job)?;
        let share_bound = *array_from_raw::<32>(share_bound)?;
        let block_bound = *array_from_raw::<32>(block_bound)?;
        *out = job.tile(t_rows, t_cols, share_bound, block_bound)?;
        Ok(())
    })
}

/// Scan all valid Pearl tiles. Pass `out=NULL, out_cap=0` to query `out_len`.
/// # Safety
/// `job` is live; bounds point to 32 readable bytes each; `out_len` is writable.
/// A nonnull `out` points to `out_cap` aligned writable TileResults. Output ranges
/// are disjoint from all other arguments. The handle must not be freed during this call.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_oracle_scan(
    job: *const PmkOracleJob,
    share_bound: *const u8,
    block_bound: *const u8,
    out: *mut TileResult,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    ffi_code(|| {
        if out_len.is_null() {
            return Err(PmkError::NullPointer);
        }
        let job = job_ref(job)?;
        let share_bound = *array_from_raw::<32>(share_bound)?;
        let block_bound = *array_from_raw::<32>(block_bound)?;
        let count = job.tile_count();
        checked_result_bytes(count)?;
        *out_len = count as u64;
        if out.is_null() {
            return if out_cap == 0 {
                Ok(())
            } else {
                Err(PmkError::NullPointer)
            };
        }
        if bounded_output_cap(out_cap, std::mem::size_of::<TileResult>())? < count {
            return Err(PmkError::BufferTooSmall);
        }
        let results = job.scan(share_bound, block_bound);
        debug_assert_eq!(results.len(), count);
        ptr::copy_nonoverlapping(results.as_ptr(), out, results.len());
        Ok(())
    })
}

/// Serialize a PlainProof for one tile. Pass `out=NULL, out_cap=0` to query `out_len`.
/// # Safety
/// `job` is live; `out_len` is writable; nonnull `out` points to `out_cap` writable bytes.
/// Output ranges are disjoint. The handle must not be freed during this call.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_oracle_build_plain_proof(
    job: *const PmkOracleJob,
    t_rows: u32,
    t_cols: u32,
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    ffi_code(|| {
        let job = job_ref(job)?;
        let proof = job.build_plain_proof(t_rows, t_cols)?;
        write_bytes(&proof, out, out_cap, out_len)
    })
}

/// Serialize pinned oracle intermediates/test-vector data. Pass `out=NULL, out_cap=0` to query `out_len`.
/// # Safety
/// `job` is live; `out_len` is writable; nonnull `out` points to `out_cap` writable bytes.
/// Output ranges are disjoint. The handle must not be freed during this call.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_oracle_export_vectors(
    job: *const PmkOracleJob,
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    ffi_code(|| {
        let job = job_ref(job)?;
        let expected_len = job.export_vectors_len()?;
        if out_len.is_null() {
            return Err(PmkError::NullPointer);
        }
        *out_len = expected_len as u64;
        if out.is_null() {
            return if out_cap == 0 {
                Ok(())
            } else {
                Err(PmkError::NullPointer)
            };
        }
        if bounded_output_cap(out_cap, 1)? < expected_len {
            return Err(PmkError::BufferTooSmall);
        }
        let vectors = job.export_vectors();
        debug_assert_eq!(vectors.len(), expected_len);
        write_bytes(&vectors, out, out_cap, out_len)
    })
}
