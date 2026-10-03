/* pmkcore C ABI — Pearl v3 job builder (see pmkcore/src/lib.rs, SPEC §4 R-A1).
 * All functions are reentrant except pmkcore_init. Return 0 on success, a negative PMK_E_* on error.
 * Callers own every buffer; pmkcore writes in place (e.g. into a shared MTLBuffer's contents). */
#ifndef PMKCORE_H
#define PMKCORE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  PMK_OK = 0,
  PMK_E_NULL = -1,
  PMK_E_HEADER = -2,      /* header is not a round-tripping 76-byte IncompleteBlockHeader */
  PMK_E_CONFIG = -3,      /* config is not a round-tripping 52-byte MiningConfiguration */
  PMK_E_POLICY = -4,      /* rank != 128, k % 128 != 0, k outside [2048, 8192], or MoE */
  PMK_E_SHAPE = -5,       /* m/n zero, > 2^24, or not multiples of the pattern periods */
  PMK_E_BUFFER = -6,      /* buffer shorter than pmkcore_padded_len(rows, k) */
  PMK_E_RANGE = -7,       /* caller-supplied matrix value outside [-64, 64] */
  PMK_E_ENTROPY = -8,     /* OS entropy source failed */
  PMK_E_THREADPOOL = -9,  /* pmkcore_init called twice or after first use */
  PMK_E_ILLEGAL_OFFSET = -10,
  PMK_E_PROOF = -11,
  PMK_E_PANIC = -12,
  PMK_E_RESOURCE_LIMIT = -13,
};

typedef struct {
  uint8_t job_key[32];       /* blake3(header76 || config52) */
  uint8_t raw_root_a[32];    /* keyed-BLAKE3 Merkle root of padded row-major A (proof carries this) */
  uint8_t salted_root_a[32]; /* bind_root_a(raw_root_a, m): cert-v3 seed input */
  uint32_t m, n, k, reserved;
} PmkTemplate;

typedef struct {
  uint8_t raw_root_b[32];    /* keyed-BLAKE3 Merkle root of padded row-major B^T (n x k) */
  uint8_t b_noise_seed[32];
  uint8_t a_noise_seed[32];
} PmkJob;

typedef struct PmkOracleJob PmkOracleJob;

typedef struct {
  uint32_t t_rows;
  uint32_t t_cols;
  uint32_t transcript[16];
  uint8_t hash[32];
  uint32_t is_share;
  uint32_t is_block;
} PmkTileResult;

/* Size the worker pool (0 = all logical CPUs). Optional; call once before anything else. */
int32_t pmkcore_init(uint32_t num_threads);
/* Chunk-padded byte length of a rows x cols int8 matrix (multiple of 1024). */
uint64_t pmkcore_padded_len(uint64_t rows, uint64_t cols);
const char *pmkcore_strerror(int32_t code);

/* Per template: validates header/config/shape, computes job_key and A's roots. a_buf must hold
 * pmkcore_padded_len(m, k) bytes; generate_a != 0 fills A from the CSPRNG (OS-entropy-keyed
 * AES-128-CTR), otherwise the caller's A is range-checked. The chunk pad is zeroed. */
int32_t pmkcore_template_init(const uint8_t header[76], const uint8_t config[52], uint32_t m, uint32_t n,
                              uint8_t *a_buf, uint64_t a_len, uint8_t generate_a, PmkTemplate *out);

/* Per job: fresh B^T (n x k row-major, int8 in [-64, 64]) written into bt_buf
 * (>= pmkcore_padded_len(n, k) bytes; pad zeroed), its raw root, and the salted cert-v3 seeds. */
int32_t pmkcore_build_job(const PmkTemplate *tmpl, uint8_t *bt_buf, uint64_t bt_len, PmkJob *out);

/* Same outputs for a caller-filled B^T (range-checked; pad zeroed). */
int32_t pmkcore_commit_job(const PmkTemplate *tmpl, uint8_t *bt_buf, uint64_t bt_len, PmkJob *out);

/* B1 Pearl oracle ABI. pattern: 0 = NA, 1 = SG. */
int32_t pmkcore_build_config(uint32_t pattern, uint32_t k, uint32_t m, uint32_t n, uint8_t out_config[52]);
int32_t pmkcore_build_config_diagnostic(uint32_t pattern, uint32_t k, uint32_t m, uint32_t n, uint8_t out_config[52]);

int32_t pmkcore_oracle_job_create(const uint8_t header[76], const uint8_t config[52], uint32_t m, uint32_t n,
                                  const uint8_t *a, uint64_t a_len, const uint8_t *bt, uint64_t bt_len,
                                  PmkOracleJob **out_job);
void pmkcore_oracle_job_free(PmkOracleJob *job);

int32_t pmkcore_oracle_tile(const PmkOracleJob *job, uint32_t t_rows, uint32_t t_cols,
                            const uint8_t share_bound[32], const uint8_t block_bound[32],
                            PmkTileResult *out);

/* To query capacity, pass out = NULL and out_cap = 0; out_len receives the required element count. */
int32_t pmkcore_oracle_scan(const PmkOracleJob *job, const uint8_t share_bound[32], const uint8_t block_bound[32],
                            PmkTileResult *out, uint64_t out_cap, uint64_t *out_len);

/* To query capacity, pass out = NULL and out_cap = 0; out_len receives the required byte count. */
int32_t pmkcore_oracle_build_plain_proof(const PmkOracleJob *job, uint32_t t_rows, uint32_t t_cols,
                                         uint8_t *out, uint64_t out_cap, uint64_t *out_len);
int32_t pmkcore_oracle_export_vectors(const PmkOracleJob *job, uint8_t *out, uint64_t out_cap, uint64_t *out_len);

#ifdef __cplusplus
}
#endif
#endif
