from pmk_miner.kernel import resolve_v3_kernel, scheme_for_kernel
from pmk_miner.scheme import V3_NA_SCHEME, V3_SG_SCHEME


def test_auto_kernel_selects_na_only_for_apple10_plus():
    assert resolve_v3_kernel("auto", "Apple10") == ("na", "Apple10")
    assert resolve_v3_kernel("auto", "Apple11") == ("na", "Apple11")
    assert resolve_v3_kernel("auto", "Apple9") == ("sg", "Apple9")
    assert resolve_v3_kernel("auto", "unknown") == ("sg", "unknown")


def test_explicit_kernel_override_selects_requested_scheme():
    assert resolve_v3_kernel("sg", "Apple10") == ("sg", "Apple10")
    assert resolve_v3_kernel("na", "Apple10") == ("na", "Apple10")
    assert scheme_for_kernel("sg") is V3_SG_SCHEME
    assert scheme_for_kernel("na") is V3_NA_SCHEME


def test_explicit_na_fails_before_native_init_on_pre_apple10():
    import pytest

    with pytest.raises(ValueError, match="Apple10"):
        resolve_v3_kernel("na", "Apple9")
