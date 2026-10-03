#include "../include/pmkcore.h"
#include <stddef.h>
_Static_assert(sizeof(PmkTemplate) == 112, "F2 template layout");
_Static_assert(sizeof(PmkJob) == 96, "F2 job layout");
_Static_assert(sizeof(PmkTileResult) == 112, "B1 result size");
_Static_assert(offsetof(PmkTileResult, transcript) == 8, "transcript offset");
_Static_assert(offsetof(PmkTileResult, hash) == 72, "hash offset");
_Static_assert(offsetof(PmkTileResult, is_share) == 104, "share offset");
_Static_assert(offsetof(PmkTileResult, is_block) == 108, "block offset");
