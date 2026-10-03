#include "libpmk.h"
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
_Static_assert(sizeof(pmk_slot) == 104, "Metal slot ABI");

int32_t pmk_valid_signal_bytes(const int8_t *bytes, uint64_t count) {
#if defined(__aarch64__)
    uint64_t index = 0;
    const int8x16_t lo = vdupq_n_s8(127);
    const int8x16_t hi = vdupq_n_s8(-128);
    while (index + 64 <= count) {
        int8x16_t minv = lo;
        int8x16_t maxv = hi;
        int8x16_t v0 = vld1q_s8(bytes + index);
        int8x16_t v1 = vld1q_s8(bytes + index + 16);
        int8x16_t v2 = vld1q_s8(bytes + index + 32);
        int8x16_t v3 = vld1q_s8(bytes + index + 48);
        minv = vminq_s8(minv, v0);
        minv = vminq_s8(minv, v1);
        minv = vminq_s8(minv, v2);
        minv = vminq_s8(minv, v3);
        maxv = vmaxq_s8(maxv, v0);
        maxv = vmaxq_s8(maxv, v1);
        maxv = vmaxq_s8(maxv, v2);
        maxv = vmaxq_s8(maxv, v3);
        if (vminvq_s8(minv) < -64 || vmaxvq_s8(maxv) > 64) {
            return 0;
        }
        index += 64;
    }
    while (index < count) {
        if (bytes[index] < -64 || bytes[index] > 64) {
            return 0;
        }
        index += 1;
    }
    return 1;
#else
    for (uint64_t index = 0; index < count; ++index) {
        if (bytes[index] < -64 || bytes[index] > 64) {
            return 0;
        }
    }
    return 1;
#endif
}
