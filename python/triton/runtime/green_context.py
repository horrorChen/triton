"""
Green Context support for Triton (CUDA 12.4+).

Green Contexts allow partitioning GPU SM resources, enabling kernels to run
on a subset of SMs. This is useful for multi-tenant GPU sharing, resource
isolation, and controlled performance experiments.

Usage:
    import torch
    import triton
    from triton.runtime.green_context import GreenContext

    with GreenContext(device=0, num_sms=32) as gctx:
        # Get a torch stream bound to the green context
        with gctx.torch_stream():
            # Triton picks up the current torch stream automatically
            kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
"""

import contextlib
import ctypes
import functools
import logging
import threading
import triton

logger = logging.getLogger(__name__)

# Minimum CUDA driver version for green context support (12.4 = 12040)
_MIN_CUDA_DRIVER_VERSION = 12040
_MIN_COMPUTE_CAPABILITY = (9, 0)  # Hopper

# CUDA stream creation flags
_CU_STREAM_NON_BLOCKING = 0x1


@functools.lru_cache(maxsize=1)
def _get_cuda_driver_version() -> int:
    """Return CUDA driver version as an integer (e.g. 12040 for 12.4)."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return 0
    ver = ctypes.c_int(0)
    if libcuda.cuDriverGetVersion(ctypes.byref(ver)) != 0:
        return 0
    return ver.value


def _check_green_context_support(device: int) -> None:
    """
    Raise a clear error if the current environment doesn't support
    green contexts.

    Checks:
        1. CUDA driver version >= 12.4 (API introduced in 12.4)
        2. Compute capability >= 9.0 (Hopper or newer)
    """
    # --- Driver version check ---
    driver_ver = _get_cuda_driver_version()
    if driver_ver < _MIN_CUDA_DRIVER_VERSION:
        major = driver_ver // 1000
        minor = (driver_ver % 1000) // 10
        raise RuntimeError(
            f"Green contexts require CUDA driver >= 12.4, but the current "
            f"driver reports version {major}.{minor} ({driver_ver}). "
            f"Please update your CUDA driver."
        )

    # --- Compute capability check ---
    cc = triton.runtime.driver.active.get_device_capability(device)
    if cc < _MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"Green contexts require compute capability >= 9.0 (Hopper+), "
            f"but device {device} has compute capability {cc[0]}.{cc[1]}. "
            f"Green contexts are not supported on this GPU."
        )


class GreenContext:
    """
    Manages a CUDA Green Context with a specified number of SMs.

    A Green Context creates a lightweight execution environment on a subset
    of the GPU's streaming multiprocessors. Kernels launched on streams
    created from this context will only execute on the allocated SMs.

    Requires CUDA 12.4+ driver and Hopper (sm_90) or newer GPU.

    Args:
        device: CUDA device index (default: 0).
        num_sms: Number of SMs to allocate for this context. Must be >= 1.
                 The actual number may be rounded up due to hardware
                 partitioning constraints.
    """

    def __init__(self, device: int = 0, num_sms: int = 1):
        if num_sms < 1:
            raise ValueError(f"num_sms must be >= 1, got {num_sms}")

        # Pre-flight: fail fast with a clear message if unsupported
        _check_green_context_support(device)

        utils = triton.runtime.driver.active.utils

        # Query total SM count for validation
        total_sms = utils.get_device_sm_count(device)
        if num_sms > total_sms:
            raise ValueError(
                f"Requested {num_sms} SMs but device {device} only has "
                f"{total_sms} SMs"
            )

        # Create the green context: returns (green_ctx_handle, cuda_ctx_handle)
        green_ctx, cuda_ctx = utils.create_green_context(device, num_sms)
        actual_sms = utils.get_green_ctx_sm_count(green_ctx)
        self._init_attrs(
            green_ctx=green_ctx,
            cuda_ctx=cuda_ctx,
            device=device,
            requested_sms=num_sms,
            actual_sms=actual_sms,
            utils=utils,
        )

    def _init_attrs(self, *, green_ctx, cuda_ctx, device, requested_sms,
                    actual_sms, utils):
        """Centralized attribute initialization."""
        self._green_ctx = green_ctx
        self._cuda_ctx = cuda_ctx
        self._device = device
        self._requested_sms = requested_sms
        self._actual_sms = actual_sms
        self._streams = []        # raw CUstream handles
        self._torch_streams = []  # torch.cuda.ExternalStream wrappers
        self._destroyed = False
        self._lock = threading.Lock()
        self._utils = utils

    @property
    def device(self) -> int:
        """CUDA device index."""
        return self._device

    @property
    def num_sms(self) -> int:
        """Actual number of SMs allocated (may differ from requested)."""
        self._check_alive()
        return self._actual_sms

    @property
    def requested_sms(self) -> int:
        """Number of SMs originally requested."""
        return self._requested_sms

    @property
    def green_ctx_handle(self) -> int:
        """Raw CUgreenCtx handle (uint64)."""
        self._check_alive()
        return self._green_ctx

    @property
    def cuda_ctx_handle(self) -> int:
        """CUcontext handle derived from the green context."""
        self._check_alive()
        return self._cuda_ctx

    def create_stream(self, priority: int = 0) -> int:
        """
        Create a raw CUDA stream bound to this green context.

        Kernels launched on this stream will only use the SMs allocated
        to this green context.

        Args:
            priority: Stream priority (lower = higher priority). Default: 0.

        Returns:
            Raw stream handle (uint64). Use torch_stream() for Triton integration.
        """
        self._check_alive()
        # CU_STREAM_NON_BLOCKING is required for green context streams
        # because the green context already owns a default stream.
        stream = self._utils.create_green_ctx_stream(
            self._green_ctx, _CU_STREAM_NON_BLOCKING, priority
        )
        self._streams.append(stream)
        return stream

    def create_torch_stream(self, priority: int = 0):
        """
        Create a torch.cuda.ExternalStream bound to this green context.

        This wraps a raw green context CUDA stream as a PyTorch stream
        object that can be used with torch.cuda.stream() context manager
        and Triton kernel launches.

        Args:
            priority: Stream priority (lower = higher priority). Default: 0.

        Returns:
            torch.cuda.ExternalStream wrapping the green context stream.
        """
        import torch
        raw_stream = self.create_stream(priority=priority)
        ext_stream = torch.cuda.ExternalStream(
            raw_stream, device=self._device
        )
        self._torch_streams.append(ext_stream)
        return ext_stream

    @contextlib.contextmanager
    def torch_stream(self, priority: int = 0):
        """
        Context manager that sets a green context stream as the current
        PyTorch CUDA stream. Triton kernels launched inside this block
        will automatically use this stream and be restricted to the
        allocated SMs.

        .. note::
            Each call creates a new CUDA stream. If called in a tight
            loop, prefer ``create_torch_stream()`` once and reuse it
            with ``torch.cuda.stream()``.

        Usage:
            with gctx.torch_stream():
                kernel[grid](args, BLOCK_SIZE=256)

        Args:
            priority: Stream priority (lower = higher priority). Default: 0.

        Yields:
            The torch.cuda.ExternalStream being used.
        """
        import torch
        ext_stream = self.create_torch_stream(priority=priority)
        with torch.cuda.stream(ext_stream):
            yield ext_stream

    def destroy(self):
        """Destroy all owned streams, then the green context itself."""
        with self._lock:
            if self._destroyed:
                return
            self._destroyed = True
        # Explicitly destroy streams before the context
        for stream_handle in self._streams:
            try:
                self._utils.destroy_stream(stream_handle)
            except Exception:
                pass  # best-effort; context destruction will clean up
        self._utils.destroy_green_context(self._green_ctx)
        self._streams.clear()
        self._torch_streams.clear()

    def _check_alive(self):
        if self._destroyed:
            raise RuntimeError("GreenContext has been destroyed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            # Log instead of silently swallowing
            logger.warning(
                "GreenContext.__del__: failed to destroy context", exc_info=True
            )

    def __repr__(self):
        status = "destroyed" if self._destroyed else "active"
        return (
            f"GreenContext(device={self._device}, "
            f"requested_sms={self._requested_sms}, "
            f"actual_sms={self._actual_sms if not self._destroyed else '?'}, "
            f"status={status})"
        )

    @staticmethod
    def create_isolated_pair(device: int = 0, num_sms_each: int = None):
        """
        Create two GreenContexts with *non-overlapping* SM partitions.

        Both partitions receive ``num_sms_each`` SMs (subject to HW
        rounding). The underlying C function uses a single
        ``cuDevSmResourceSplitByCount`` call with ``nbGroups=2`` to
        guarantee disjoint SM sets.

        Args:
            device: CUDA device index (default: 0).
            num_sms_each: SMs per context. If None, defaults to
                          ``total_sms // 4`` to leave headroom for
                          hardware rounding.

        Returns:
            (gctx1, gctx2) — two GreenContext objects. The caller is
            responsible for destroying them (or using them in with-blocks).
        """
        _check_green_context_support(device)

        utils = triton.runtime.driver.active.utils
        total_sms = utils.get_device_sm_count(device)

        if num_sms_each is None:
            num_sms_each = total_sms // 4
        if num_sms_each < 1:
            raise ValueError(f"num_sms_each must be >= 1, got {num_sms_each}")
        if num_sms_each > total_sms:
            raise ValueError(
                f"num_sms_each ({num_sms_each}) exceeds total SMs ({total_sms})"
            )

        g1, c1, g2, c2 = utils.create_green_context_pair(
            device, num_sms_each
        )

        gctx1 = GreenContext._from_raw_handles(
            green_ctx=g1, cuda_ctx=c1, device=device,
            requested_sms=num_sms_each, utils=utils,
        )
        gctx2 = GreenContext._from_raw_handles(
            green_ctx=g2, cuda_ctx=c2, device=device,
            requested_sms=num_sms_each, utils=utils,
        )
        return gctx1, gctx2

    @classmethod
    def _from_raw_handles(cls, *, green_ctx, cuda_ctx, device,
                          requested_sms, utils):
        """Construct a GreenContext from pre-created CUDA handles.

        This bypasses __init__ (which would create a new independent
        partition) while still going through the centralized
        ``_init_attrs`` method so all attributes are initialized
        consistently.
        """
        obj = object.__new__(cls)
        actual_sms = utils.get_green_ctx_sm_count(green_ctx)
        obj._init_attrs(
            green_ctx=green_ctx,
            cuda_ctx=cuda_ctx,
            device=device,
            requested_sms=requested_sms,
            actual_sms=actual_sms,
            utils=utils,
        )
        return obj
