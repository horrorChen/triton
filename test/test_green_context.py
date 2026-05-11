"""
Tests for CUDA Green Context support in Triton.

Green Contexts (CUDA 12.4+) allow partitioning GPU SMs so that kernels
run on a controlled subset of streaming multiprocessors.

Requirements:
    - CUDA driver >= 12.4
    - GPU with compute capability >= 9.0 (Hopper or newer)
"""

import pytest
import torch
import triton
import triton.language as tl
from unittest import mock
from triton.runtime.green_context import (
    GreenContext,
    _check_green_context_support,
    _get_cuda_driver_version,
    _MIN_CUDA_DRIVER_VERSION,
    _MIN_COMPUTE_CAPABILITY,
)


def _get_device_capability():
    """Return (major, minor) compute capability of current device."""
    return torch.cuda.get_device_capability(0)


def _skip_if_unsupported():
    """Skip test if green contexts are not supported."""
    major, minor = _get_device_capability()
    if major < 9:
        pytest.skip(
            f"Green contexts require compute capability >= 9.0, "
            f"got {major}.{minor}"
        )


# ---------------------------------------------------------------------------
# Triton kernels for testing
# ---------------------------------------------------------------------------


@triton.jit
def _add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _smid_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Kernel that records the SM ID executing each program."""
    pid = tl.program_id(axis=0)
    # Use inline asm to read %smid
    smid = tl.inline_asm_elementwise(
        "mov.u32 $0, %smid;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    if pid < n_elements:
        tl.store(output_ptr + pid, smid)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGreenContextCreation:
    """Test green context lifecycle (create, query, destroy)."""

    def setup_method(self):
        _skip_if_unsupported()

    def test_create_and_destroy(self):
        """Basic create and destroy cycle."""
        gctx = GreenContext(device=0, num_sms=1)
        assert gctx.num_sms >= 1
        assert gctx.device == 0
        assert gctx.green_ctx_handle != 0
        assert gctx.cuda_ctx_handle != 0
        gctx.destroy()

    def test_context_manager(self):
        """Test with-statement lifecycle."""
        with GreenContext(device=0, num_sms=1) as gctx:
            assert gctx.num_sms >= 1
            handle = gctx.green_ctx_handle
            assert handle != 0
        # After exiting, context should be destroyed
        with pytest.raises(RuntimeError, match="destroyed"):
            _ = gctx.num_sms

    def test_sm_count_respects_request(self):
        """Actual SM count should be >= requested (rounding up is OK)."""
        utils = triton.runtime.driver.active.utils
        total_sms = utils.get_device_sm_count(0)

        # Request a small number
        with GreenContext(device=0, num_sms=1) as gctx:
            assert gctx.num_sms >= 1
            assert gctx.num_sms <= total_sms

        # Request half the SMs
        half = max(1, total_sms // 2)
        with GreenContext(device=0, num_sms=half) as gctx:
            assert gctx.num_sms >= half
            assert gctx.num_sms <= total_sms

    def test_invalid_sm_count_zero(self):
        """Requesting 0 SMs should raise ValueError."""
        with pytest.raises(ValueError, match="num_sms must be >= 1"):
            GreenContext(device=0, num_sms=0)

    def test_invalid_sm_count_exceeds_device(self):
        """Requesting more SMs than the device has should raise ValueError."""
        utils = triton.runtime.driver.active.utils
        total_sms = utils.get_device_sm_count(0)
        with pytest.raises(ValueError, match="only has"):
            GreenContext(device=0, num_sms=total_sms + 100)

    def test_double_destroy_is_safe(self):
        """Calling destroy twice should not crash."""
        gctx = GreenContext(device=0, num_sms=1)
        gctx.destroy()
        gctx.destroy()  # Should be a no-op

    def test_repr(self):
        """Test string representation."""
        with GreenContext(device=0, num_sms=1) as gctx:
            r = repr(gctx)
            assert "GreenContext" in r
            assert "active" in r
        r2 = repr(gctx)
        assert "destroyed" in r2


class TestGreenContextPreflightChecks:
    """Test that clear errors are raised on unsupported environments."""

    def test_driver_version_check(self):
        """Should raise RuntimeError when driver < 12.4."""
        with mock.patch(
            "triton.runtime.green_context._get_cuda_driver_version",
            return_value=12030,  # 12.3 — too old
        ):
            with pytest.raises(RuntimeError, match="CUDA driver >= 12.4"):
                _check_green_context_support(device=0)

    def test_compute_capability_check(self):
        """Should raise RuntimeError when GPU < sm_90."""
        with mock.patch(
            "triton.runtime.green_context._get_cuda_driver_version",
            return_value=12040,  # driver OK
        ), mock.patch.object(
            triton.runtime.driver.active,
            "get_device_capability",
            return_value=(8, 0),  # Ampere — too old
        ):
            with pytest.raises(RuntimeError, match="compute capability >= 9.0"):
                _check_green_context_support(device=0)

    def test_passes_on_supported_env(self):
        """Should not raise on a supported environment."""
        _skip_if_unsupported()
        # If we get here, the real hardware supports it — should pass
        _check_green_context_support(device=0)

    def test_real_driver_version(self):
        """Sanity: real driver version should be retrievable and > 0."""
        ver = _get_cuda_driver_version()
        assert ver > 0, "Could not retrieve CUDA driver version"

    def test_constructor_checks_support(self):
        """GreenContext.__init__ should call the preflight check."""
        with mock.patch(
            "triton.runtime.green_context._get_cuda_driver_version",
            return_value=11000,  # CUDA 11.0 — way too old
        ):
            with pytest.raises(RuntimeError, match="CUDA driver >= 12.4"):
                GreenContext(device=0, num_sms=1)


class TestGreenContextStream:
    """Test stream creation on green contexts."""

    def setup_method(self):
        _skip_if_unsupported()

    def test_create_stream(self):
        """Create a raw stream from green context."""
        with GreenContext(device=0, num_sms=1) as gctx:
            stream = gctx.create_stream()
            assert stream != 0
            assert isinstance(stream, int)

    def test_create_multiple_streams(self):
        """Create multiple streams from the same green context."""
        with GreenContext(device=0, num_sms=1) as gctx:
            s1 = gctx.create_stream()
            s2 = gctx.create_stream()
            assert s1 != 0
            assert s2 != 0
            assert s1 != s2

    def test_create_torch_stream(self):
        """Create a torch.cuda.ExternalStream from green context."""
        with GreenContext(device=0, num_sms=1) as gctx:
            ts = gctx.create_torch_stream()
            assert isinstance(ts, torch.cuda.ExternalStream)

    def test_torch_stream_context_manager(self):
        """torch_stream() should set the current CUDA stream."""
        with GreenContext(device=0, num_sms=1) as gctx:
            with gctx.torch_stream() as stream:
                current = torch.cuda.current_stream()
                assert current.cuda_stream == stream.cuda_stream

    def test_stream_after_destroy_raises(self):
        """Creating a stream after destroy should raise."""
        gctx = GreenContext(device=0, num_sms=1)
        gctx.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            gctx.create_stream()


class TestGreenContextKernelLaunch:
    """Test launching Triton kernels on green context streams."""

    def setup_method(self):
        _skip_if_unsupported()

    def test_vector_add_on_green_context(self):
        """Launch a simple vector add kernel on a green context stream."""
        n_elements = 1024
        x = torch.rand(n_elements, device="cuda")
        y = torch.rand(n_elements, device="cuda")
        output = torch.empty_like(x)
        expected = x + y

        with GreenContext(device=0, num_sms=4) as gctx:
            with gctx.torch_stream():
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                _add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)

        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected)

    def test_vector_add_large(self):
        """Larger vector add to exercise multiple thread blocks."""
        n_elements = 1024 * 1024
        x = torch.rand(n_elements, device="cuda")
        y = torch.rand(n_elements, device="cuda")
        output = torch.empty_like(x)
        expected = x + y

        with GreenContext(device=0, num_sms=8) as gctx:
            with gctx.torch_stream():
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
                _add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected)

    def test_multiple_kernels_same_green_ctx(self):
        """Launch multiple kernels on the same green context stream."""
        n = 4096
        with GreenContext(device=0, num_sms=4) as gctx:
            with gctx.torch_stream():
                for _ in range(5):
                    x = torch.rand(n, device="cuda")
                    y = torch.rand(n, device="cuda")
                    output = torch.empty_like(x)
                    expected = x + y
                    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
                    _add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, expected)

    def test_different_green_contexts_different_sm_counts(self):
        """
        Create green contexts with different SM counts. All should
        produce correct results.
        """
        n = 8192

        for num_sms in [1, 4, 8]:
            x = torch.rand(n, device="cuda")
            y = torch.rand(n, device="cuda")
            output = torch.empty_like(x)
            expected = x + y

            with GreenContext(device=0, num_sms=num_sms) as gctx:
                with gctx.torch_stream():
                    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
                    _add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
                    torch.cuda.synchronize()

            torch.testing.assert_close(output, expected)


class TestGreenContextSmPartitioning:
    """
    Test that green contexts actually restrict execution to a subset of SMs.
    Uses inline ASM to read %smid and verify SM usage.
    """

    def setup_method(self):
        _skip_if_unsupported()

    def test_sm_ids_within_bounds(self):
        """
        Verify that kernels running on a green context only use a subset
        of SM IDs. The set of SM IDs used should be limited.
        """
        utils = triton.runtime.driver.active.utils
        total_sms = utils.get_device_sm_count(0)

        # Request a small fraction of SMs
        requested = max(1, total_sms // 4)

        n_programs = min(4096, total_sms * 16)  # Many programs to saturate
        output = torch.zeros(n_programs, dtype=torch.int32, device="cuda")

        with GreenContext(device=0, num_sms=requested) as gctx:
            actual_sms = gctx.num_sms
            with gctx.torch_stream():
                _smid_kernel[(n_programs,)](output, n_programs, BLOCK_SIZE=1)

        torch.cuda.synchronize()
        sm_ids = output.cpu().tolist()
        unique_sms = set(sm_ids)

        # The number of unique SM IDs should not exceed the allocated count.
        # Note: due to hardware scheduling, we may see fewer SMs than allocated,
        # but never more.
        assert len(unique_sms) <= actual_sms, (
            f"Green context allocated {actual_sms} SMs but kernel ran on "
            f"{len(unique_sms)} unique SMs: {sorted(unique_sms)}"
        )
        # Sanity: at least 1 SM was used
        assert len(unique_sms) >= 1

    def test_individual_contexts_bound_sm_count(self):
        """
        Two independently created green contexts (NOT a disjoint pair)
        each bound their SM usage.  Note: independent contexts may
        overlap — for true disjoint isolation see
        test_isolated_pair_sm_ids_disjoint.
        """
        utils = triton.runtime.driver.active.utils
        total_sms = utils.get_device_sm_count(0)

        if total_sms < 8:
            pytest.skip("Need >= 8 SMs for this test")

        quarter = max(1, total_sms // 4)
        n_programs = min(2048, total_sms * 8)

        sms_per_ctx = []
        for _ in range(2):
            output = torch.zeros(n_programs, dtype=torch.int32, device="cuda")
            with GreenContext(device=0, num_sms=quarter) as gctx:
                actual = gctx.num_sms
                with gctx.torch_stream():
                    _smid_kernel[(n_programs,)](
                        output, n_programs, BLOCK_SIZE=1
                    )
                    torch.cuda.synchronize()
            unique = set(output.cpu().tolist())
            sms_per_ctx.append((unique, actual))

        for i, (sms, actual) in enumerate(sms_per_ctx):
            assert len(sms) <= actual, (
                f"Context {i} used {len(sms)} SMs, "
                f"expected at most {actual} (allocated)"
            )


class TestGreenContextWithDefaultStream:
    """Compare results between green context stream and default stream."""

    def setup_method(self):
        _skip_if_unsupported()

    def test_results_match_default_stream(self):
        """
        A kernel launched on a green context stream should produce the
        same results as the same kernel on the default stream.
        """
        n = 16384
        x = torch.rand(n, device="cuda")
        y = torch.rand(n, device="cuda")

        # Default stream
        out_default = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_kernel[grid](x, y, out_default, n, BLOCK_SIZE=512)
        torch.cuda.synchronize()

        # Green context stream
        out_green = torch.empty_like(x)
        with GreenContext(device=0, num_sms=4) as gctx:
            with gctx.torch_stream():
                _add_kernel[grid](x, y, out_green, n, BLOCK_SIZE=512)

        torch.cuda.synchronize()
        torch.testing.assert_close(out_default, out_green)


# ---------------------------------------------------------------------------
# Persistent kernels for resource isolation test
# ---------------------------------------------------------------------------


class TestGreenContextResourceIsolation:
    """
    Prove that two green contexts with non-overlapping SM partitions
    can execute kernels *truly concurrently*.

    The spin-wait + atomic-inc test is run as a subprocess because:
    - ``while`` loops with ``tl.atomic_add`` in Triton work correctly
      but interact poorly with pytest's import/compilation caching.
    - The standalone script ``test_spin_debug.py`` has been proven to
      pass reliably. Running it as a subprocess isolates the test.
    """

    def setup_method(self):
        _skip_if_unsupported()

    def test_concurrent_spin_wait_no_deadlock(self):
        """
        Two isolated green contexts must allow concurrent execution.
        A deadlock (timeout) means SM isolation is not working.

        Uses a subprocess with timeout to avoid pytest/CUDA signal
        interaction issues.
        """
        import subprocess
        import os

        script = os.path.join(
            os.path.dirname(__file__),
            "test_green_context_isolation.py",
        )
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Isolation subprocess failed (rc={result.returncode}):\n"
            f"STDOUT (last 3000 chars):\n"
            f"{result.stdout[-3000:] if result.stdout else '<empty>'}\n"
            f"STDERR (last 2000 chars):\n"
            f"{result.stderr[-2000:] if result.stderr else '<empty>'}"
        )

    def test_isolated_pair_sm_ids_disjoint(self):
        """
        Verify that create_isolated_pair actually assigns disjoint SM sets.
        """
        utils = triton.runtime.driver.active.utils
        total_sms = utils.get_device_sm_count(0)
        quarter_sms = max(2, total_sms // 4)

        if total_sms < 8:
            pytest.skip("Need >= 8 total SMs for this test")

        gctx1, gctx2 = GreenContext.create_isolated_pair(
            device=0, num_sms_each=quarter_sms
        )
        try:
            n_programs = min(4096, total_sms * 16)

            out1 = torch.zeros(n_programs, dtype=torch.int32, device="cuda")
            out2 = torch.zeros(n_programs, dtype=torch.int32, device="cuda")

            with gctx1.torch_stream():
                _smid_kernel[(n_programs,)](out1, n_programs, BLOCK_SIZE=1)
            with gctx2.torch_stream():
                _smid_kernel[(n_programs,)](out2, n_programs, BLOCK_SIZE=1)

            torch.cuda.synchronize()
            sms1 = set(out1.cpu().tolist())
            sms2 = set(out2.cpu().tolist())

            overlap = sms1 & sms2
            assert len(overlap) == 0, (
                f"SM partitions should be disjoint but share {len(overlap)} "
                f"SM(s): {sorted(overlap)}\n"
                f"  gctx1 SMs ({len(sms1)}): {sorted(sms1)}\n"
                f"  gctx2 SMs ({len(sms2)}): {sorted(sms2)}"
            )
        finally:
            gctx1.destroy()
            gctx2.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
