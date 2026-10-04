/* SPDX-License-Identifier: Apache-2.0 */
/* Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948. */

#ifndef PMK_V4_H
#define PMK_V4_H
#include <stdint.h>
#include "libpmk.h"
#ifdef __cplusplus
extern "C" {
#endif

#define PMK_V4_ABI_VERSION 1

typedef void *pmk_v4_context;
typedef void *pmk_v4_job;

typedef struct {
    const int8_t *clean_values;       /* rows x k, row-major, fixed unit scale */
    uint64_t clean_value_count;
    const uint8_t *noise_e_codes;    /* rows x r, row-major E4M3 */
    uint64_t noise_e_count;
    const uint8_t *noise_f_codes;    /* k x r, row-major E4M3 */
    uint64_t noise_f_count;
    const uint16_t *alpha_bf16;      /* rows */
    const uint16_t *beta_bf16;       /* rows */
    uint64_t scale_count;
} pmk_v4_operand_desc;

typedef struct {
    uint32_t abi_version, m, n, k, r;
    pmk_v4_operand_desc a;
    pmk_v4_operand_desc bt;
    uint32_t jackpot_key[8], block_bound[8], share_bound[8];
    uint32_t block_capacity, share_capacity;
    uint64_t job_id;
} pmk_v4_job_desc;

typedef struct {
    uint32_t abi_version, m, n, k;
    const uint8_t *a_codes;          /* m x k, row-major E4M3 */
    const uint8_t *bt_codes;         /* n x k, row-major E4M3 */
    uint64_t a_code_count, bt_code_count;
    uint32_t jackpot_key[8], block_bound[8], share_bound[8];
    uint32_t block_capacity, share_capacity;
    uint64_t job_id;
} pmk_v4_codes_desc;

typedef struct {
    uint32_t abi_version, flags;
    uint64_t fallback_groups, total_groups;
    uint64_t quantized_a, quantized_b;
    uint64_t quant_saturated_a, quant_saturated_b;
    uint64_t quant_nan_a, quant_nan_b;
    uint32_t layout_failures, fallback_alert;
} pmk_v4_stats;

typedef struct {
    uint32_t abi_version;
    int32_t status;
    uint64_t job_id;
    uint32_t block_count, share_count;
    uint32_t block_stored, share_stored;
    uint32_t overflow, recovered;
    const pmk_slot *blocks, *shares;
    pmk_v4_stats stats;
    const uint32_t *c_bits;          /* optional diagnostic C readback, m*n words */
    uint64_t c_count;
    double gpu_start_time, gpu_end_time;
} pmk_v4_result;

typedef struct {
    uint32_t abi_version, reserved;
    const uint8_t *a_codes;
    const uint8_t *bt_codes;
    uint64_t a_code_count, bt_code_count;
} pmk_v4_quantized_codes;

typedef void (*pmk_v4_completion)(pmk_v4_job job, void *user);

int32_t pmk_v4_init(pmk_v4_context *out, char *error, uint64_t error_capacity);
int32_t pmk_v4_init_diagnostic(pmk_v4_context *out, char *error, uint64_t error_capacity);
int32_t pmk_v4_probe(pmk_v4_context ctx, char *cache_key, uint64_t capacity);
int32_t pmk_v4_admission_metadata(pmk_v4_context ctx, char *json, uint64_t capacity);
int32_t pmk_v4_fp8_selftest(pmk_v4_context ctx, char *error, uint64_t error_capacity);
int32_t pmk_v4_write_admission_record(pmk_v4_context ctx, const char *path,
                                      uint64_t exact_cells, char *error, uint64_t error_capacity);
void pmk_v4_destroy(pmk_v4_context ctx);

int32_t pmk_v4_run_job(pmk_v4_context ctx, const pmk_v4_job_desc *desc,
                       pmk_v4_completion callback, void *user, pmk_v4_job *out);
int32_t pmk_v4_run_codes_diagnostic(pmk_v4_context ctx, const pmk_v4_codes_desc *desc,
                                    pmk_v4_completion callback, void *user, pmk_v4_job *out);
int32_t pmk_v4_poll(pmk_v4_job job, pmk_v4_result *out);
int32_t pmk_v4_job_quantized_codes(pmk_v4_job job, pmk_v4_quantized_codes *out);
int32_t pmk_v4_context_error(pmk_v4_context ctx, char *error, uint64_t error_capacity);
int32_t pmk_v4_job_error(pmk_v4_job job, char *error, uint64_t error_capacity);
int32_t pmk_v4_job_wait_callback(pmk_v4_job job);
int32_t pmk_v4_job_release(pmk_v4_job job);

#ifdef __cplusplus
}
#endif
#endif
