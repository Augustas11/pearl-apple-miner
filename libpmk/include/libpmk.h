/* SPDX-License-Identifier: Apache-2.0 */
#ifndef LIBPMK_H
#define LIBPMK_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define PMK_ABI_VERSION 1
enum { PMK_SUCCESS=0, PMK_PENDING=1, PMK_INVALID=-101, PMK_RESOURCE=-102,
       PMK_PROBE_FAILED=-103, PMK_GPU_FAILED=-104, PMK_BUSY=-105 };
typedef void *pmk_context;
typedef void *pmk_job;
typedef void *pmk_activity;
typedef enum {
    PMK_POWER_SOURCE_UNKNOWN = 0,
    PMK_POWER_SOURCE_AC = 1,
    PMK_POWER_SOURCE_BATTERY = 2,
    PMK_POWER_SOURCE_DESKTOP = 3
} pmk_power_source_kind;
typedef enum {
    PMK_THERMAL_NOMINAL = 0,
    PMK_THERMAL_FAIR = 1,
    PMK_THERMAL_SERIOUS = 2,
    PMK_THERMAL_CRITICAL = 3,
    PMK_THERMAL_UNKNOWN = 4
} pmk_thermal_state_kind;
typedef struct { uint32_t t_rows, t_cols, transcript[16], hash[8]; } pmk_slot;
typedef struct {
    uint32_t abi_version, m, n, k;
    const int8_t *a, *bt;
    uint64_t a_bytes, bt_bytes;
    uint32_t a_seed[8], b_seed[8], block_bound[8], share_bound[8];
    uint32_t block_capacity, share_capacity;
    uint32_t cert_version, rank;
    uint64_t job_id;
} pmk_job_desc;
typedef struct {
    uint32_t abi_version;
    int32_t status;
    uint64_t job_id;
    uint32_t block_count, share_count;
    uint32_t block_stored, share_stored;
    uint32_t overflow, recovered;
    const pmk_slot *blocks, *shares;
    double gpu_start_time, gpu_end_time;
} pmk_result;
typedef void (*pmk_completion)(pmk_job job, void *user);
/* All handles must originate here and remain valid during a call. No concurrent release.
 * Shared raw inputs are immutable until job release. Result pointers live until release.
 * A callback may poll and release its job; coordinate release with polling threads.
 * All uint32 seed/bound words are little-endian; bounds are inclusive U256 LE. */
int32_t pmk_init(pmk_context *out, char *error, uint64_t error_capacity);
int32_t pmk_init_diagnostic(pmk_context *out, char *error, uint64_t error_capacity);
int32_t pmk_probe(pmk_context ctx, char *cache_key, uint64_t capacity);
int32_t pmk_probe_refresh(pmk_context ctx, char *error, uint64_t error_capacity);
void pmk_destroy(pmk_context ctx);
/* desc.a/bt must be base pointers allocated on this context; signal range [-64,64]. */
int32_t pmk_buffer_alloc(pmk_context ctx, uint64_t bytes, void **out);
int32_t pmk_buffer_release(pmk_context ctx, void *buffer);
int32_t pmk_run_job(pmk_context ctx, const pmk_job_desc *desc,
                    pmk_completion callback, void *user, pmk_job *out);
/* Explicit K3-NA entry point. Returns PMK_INVALID unless this context selected K3-NA. */
int32_t pmk_run_job_na(pmk_context ctx, const pmk_job_desc *desc,
                       pmk_completion callback, void *user, pmk_job *out);
/* Explicit diagnostic entry point for non-production vectors, including k=65536. */
int32_t pmk_run_job_diagnostic(pmk_context ctx, const pmk_job_desc *desc,
                               pmk_completion callback, void *user, pmk_job *out);
/* JSON metadata for cert-v3 kernel selection: kernel, pattern_id, device_class, cache_key. */
int32_t pmk_v3_kernel_metadata(pmk_context ctx, char *metadata, uint64_t metadata_capacity);
int32_t pmk_kernel_metadata(pmk_context ctx, char *metadata, uint64_t metadata_capacity);
int32_t pmk_poll(pmk_job job, pmk_result *out);
int32_t pmk_context_error(pmk_context ctx, char *error, uint64_t error_capacity);
int32_t pmk_job_error(pmk_job job, char *error, uint64_t error_capacity);
/* External release owners call this before release to wait until the completion
 * callback has fully returned. Calling it from that callback returns PMK_BUSY. */
int32_t pmk_job_wait_callback(pmk_job job);
int32_t pmk_job_release(pmk_job job);
int32_t pmk_valid_signal_bytes(const int8_t *bytes, uint64_t count);
/* Prevent App Nap and idle system sleep while allowing the display to sleep.
 * The returned activity must be ended exactly once; ending NULL is a no-op. */
pmk_activity pmk_activity_begin(const char *reason);
void pmk_activity_end(pmk_activity activity);
/* Returns a pmk_power_source_kind value. AC means a battery-backed Mac on
 * external power; DESKTOP means AC power with no internal battery. */
int32_t pmk_power_source(void);
/* Returns a pmk_thermal_state_kind value from ProcessInfo.thermalState. */
int32_t pmk_thermal_state(void);
#ifdef __cplusplus
}
#endif
#ifndef PMK_V4_H
#include "pmk_v4.h"
#endif
#endif
