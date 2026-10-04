/* SPDX-License-Identifier: Apache-2.0 */
/* Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948. */
/* pmkcore_v4 C ABI -- Pearl certificate v4 FP8 core.
 * Standalone from the v3 pmkcore ABI: link libpmkcore_v4 separately.
 * All buffers exposed through Pmk4GpuJobDesc are borrowed from Pmk4Job and stay
 * valid until pmkcore_v4_job_free. Return 0 on success, negative PMK4_E_* on error. */
#ifndef PMKCORE_V4_H
#define PMKCORE_V4_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  PMK4_OK = 0,
  PMK4_E_NULL = -1,
  PMK4_E_HEADER = -2,
  PMK4_E_CONFIG = -3,
  PMK4_E_POLICY = -4,
  PMK4_E_SHAPE = -5,
  PMK4_E_BUFFER = -6,
  PMK4_E_RANGE = -7,
  PMK4_E_ENTROPY = -8,
  PMK4_E_THREADPOOL = -9,
  PMK4_E_ILLEGAL_OFFSET = -10,
  PMK4_E_PROOF = -11,
  PMK4_E_PANIC = -12,
  PMK4_E_RESOURCE_LIMIT = -13,
  PMK4_E_VERIFY = -14,
};

typedef struct Pmk4Job Pmk4Job;

typedef struct {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t rank;
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t row_period;
  uint32_t col_period;

  /* Clean committed operands, row-major A[m][k] and B^T[n][k].
   * Current grid jobs use +/-64 values and unit BF16 scales. */
  const int8_t *a_values;
  const uint16_t *a_scales;
  const int8_t *bt_values;
  const uint16_t *bt_scales;

  /* Diagnostic/oracle E4M3 noised operands. NULL until
   * pmkcore_v4_prepare_oracle_noised or pmkcore_v4_export_vectors is called. */
  const uint8_t *a_noised;
  const uint8_t *bt_noised;
  const uint8_t *a_noise_e;
  const uint8_t *a_noise_f;
  const uint8_t *bt_noise_e;
  const uint8_t *bt_noise_f;

  const uint16_t *a_alpha;
  const uint16_t *a_beta;
  const uint16_t *a_l2;
  const uint16_t *bt_alpha;
  const uint16_t *bt_beta;
  const uint16_t *bt_l2;

  uint8_t key_a[32];
  uint8_t key_b[32];
  uint8_t hash_a[32];
  uint8_t hash_b[32];
  uint8_t noise_seed_a[32];
  uint8_t noise_seed_b[32];
  uint8_t jackpot_key[32];
} Pmk4GpuJobDesc;

typedef struct {
  uint32_t t_rows;
  uint32_t t_cols;
  uint8_t message[64];
  uint8_t hash[32];
  uint32_t policy_pass;
  uint32_t is_share;
  uint32_t is_block;
} Pmk4TileResult;

int32_t pmkcore_v4_init(uint32_t num_threads);
const char *pmkcore_v4_strerror(int32_t code);

int32_t pmkcore_v4_job_create_grid_b200(const uint8_t proposed_header[76],
                                        const uint8_t ancestor_header[108],
                                        const uint8_t *ancestor_chain,
                                        uint64_t ancestor_chain_len,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        Pmk4Job **out_job);
void pmkcore_v4_job_free(Pmk4Job *job);

int32_t pmkcore_v4_gpu_descriptor(const Pmk4Job *job, Pmk4GpuJobDesc *out);

int32_t pmkcore_v4_prepare_oracle_noised(Pmk4Job *job);
int32_t pmkcore_v4_tile_cpu_oracle(const Pmk4Job *job,
                                   uint32_t t_rows,
                                   uint32_t t_cols,
                                   const uint8_t share_bound[32],
                                   const uint8_t block_bound[32],
                                   Pmk4TileResult *out);

int32_t pmkcore_v4_scan_cpu_oracle(const Pmk4Job *job,
                                   const uint8_t share_bound[32],
                                   const uint8_t block_bound[32],
                                   Pmk4TileResult *out,
                                   uint64_t out_cap,
                                   uint64_t *out_len);

int32_t pmkcore_v4_classify_tile_message(const Pmk4Job *job,
                                         uint32_t t_rows,
                                         uint32_t t_cols,
                                         const uint8_t message[64],
                                         const uint8_t share_bound[32],
                                         const uint8_t block_bound[32],
                                         Pmk4TileResult *out);

int32_t pmkcore_v4_bound_for_nbits(const Pmk4Job *job, uint32_t nbits, uint8_t out_bound[32]);

int32_t pmkcore_v4_build_plain_proof(const Pmk4Job *job,
                                     uint32_t t_rows,
                                     uint32_t t_cols,
                                     uint8_t *out,
                                     uint64_t out_cap,
                                     uint64_t *out_len);

int32_t pmkcore_v4_verify_plain_proof(const uint8_t proposed_header[76],
                                      const uint8_t *proof,
                                      uint64_t proof_len,
                                      const uint8_t *nbits_override_le_u32,
                                      uint8_t *accepted);

int32_t pmkcore_v4_mutate_plain_proof(const uint8_t *proof,
                                      uint64_t proof_len,
                                      uint32_t mutation_id,
                                      uint8_t *out,
                                      uint64_t out_cap,
                                      uint64_t *out_len);

int32_t pmkcore_v4_build_certificate(const uint8_t proposed_header[76],
                                     const uint8_t *plain_proof,
                                     uint64_t plain_proof_len,
                                     uint8_t *public_out,
                                     uint64_t public_cap,
                                     uint64_t *public_len,
                                     uint8_t *proof_out,
                                     uint64_t proof_cap,
                                     uint64_t *proof_len);

int32_t pmkcore_v4_export_vectors(const Pmk4Job *job, uint8_t *out, uint64_t out_cap, uint64_t *out_len);

#ifdef __cplusplus
}
#endif
#endif
