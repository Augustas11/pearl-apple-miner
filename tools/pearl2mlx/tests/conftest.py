import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# Tests run on the CPU only (shared machine; no GPU load).
mx.set_default_device(mx.cpu)
