#!/usr/bin/env python3
"""
Standalone green context SM isolation test.

This is the EXACT pattern from the proven debug script (test_spin_debug.py)
that consistently passes. The key requirements are:
  1. Pre-compile kernels individually (tests 1-2)
  2. Run spin+inc on regular streams first (test 3 — primes multi-stream)
  3. Then run spin+inc on green context streams (test 4)
  4. All concurrent launches MUST be in a separate thread to avoid
     PyTorch's implicit stream-event dependencies.

Exit code 0 = PASS, non-zero = FAIL.
"""

import sys
import threading
import torch
import triton
import triton.language as tl
from triton.runtime.green_context import GreenContext


@triton.jit
def _spin_kernel(flag_ptr, done_ptr, target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    while tl.atomic_add(flag_ptr, 0) < target:
        pass
    tl.store(done_ptr + pid, 1)


@triton.jit
def _inc_kernel(flag_ptr, num_tiles, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    for tile_id in range(pid, num_tiles, num_pids):
        tl.atomic_add(flag_ptr, 1)


def _launch_concurrent(stream1, stream2, spin_grid, inc_grid, flag, done, target):
    """Launch spin+inc in a thread to avoid PyTorch stream-event deps."""
    result = {"ok": False}

    def run():
        with torch.cuda.stream(stream1):
            _spin_kernel[(spin_grid,)](flag, done, target, BLOCK_SIZE=1)
        with torch.cuda.stream(stream2):
            _inc_kernel[(inc_grid,)](flag, target, BLOCK_SIZE=1)
        torch.cuda.synchronize()
        result["ok"] = True

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=60)
    return result["ok"]


def main():
    major, _ = torch.cuda.get_device_capability(0)
    if major < 9:
        print(f"SKIP: compute capability {major}.x < 9.0")
        sys.exit(0)

    # Step 1: Pre-compile spin kernel
    print("Step 1: Pre-compile spin kernel", flush=True)
    flag1 = torch.tensor([10], dtype=torch.int32, device="cuda")
    done1 = torch.zeros(4, dtype=torch.int32, device="cuda")
    _spin_kernel[(4,)](flag1, done1, 10, BLOCK_SIZE=1)
    torch.cuda.synchronize()
    assert done1.tolist() == [1, 1, 1, 1]

    # Step 2: Pre-compile inc kernel
    print("Step 2: Pre-compile inc kernel", flush=True)
    flag2 = torch.zeros(1, dtype=torch.int32, device="cuda")
    _inc_kernel[(4,)](flag2, 8, BLOCK_SIZE=1)
    torch.cuda.synchronize()
    assert flag2.item() == 8

    # Step 3: Verify spin+inc works on regular CUDA streams
    print("Step 3: Spin+inc on regular streams", flush=True)
    flag3 = torch.zeros(1, dtype=torch.int32, device="cuda")
    done3 = torch.zeros(4, dtype=torch.int32, device="cuda")
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    ok3 = _launch_concurrent(s1, s2, 4, 4, flag3, done3, 4)
    assert ok3, f"Regular-stream spin+inc failed: flag={flag3.item()}"
    print(f"  PASS: flag={flag3.item()}", flush=True)

    # Step 4: Spin+inc on green context pair
    print("Step 4: Spin+inc on green context pair", flush=True)
    gctx1, gctx2 = GreenContext.create_isolated_pair(device=0)
    try:
        grid1 = gctx1.num_sms
        grid2 = gctx2.num_sms
        target = grid2
        print(f"  gctx1={grid1} SMs, gctx2={grid2} SMs, target={target}", flush=True)

        flag4 = torch.zeros(1, dtype=torch.int32, device="cuda")
        done4 = torch.zeros(grid1, dtype=torch.int32, device="cuda")
        ts1 = gctx1.create_torch_stream()
        ts2 = gctx2.create_torch_stream()

        ok4 = _launch_concurrent(ts1, ts2, grid1, grid2, flag4, done4, target)

        if not ok4:
            print(f"  FAIL: TIMED OUT — SM isolation not working")
            print(f"    flag={flag4.item()}, done={done4[:grid1].tolist()}")
            sys.exit(1)

        flag_val = flag4.item()
        done_vals = done4[:grid1].tolist()
        assert flag_val >= target, f"flag={flag_val} < target={target}"
        assert all(v == 1 for v in done_vals), f"done={done_vals}"
        print(f"  PASS: flag={flag_val}, all {grid1} spinners completed", flush=True)
    finally:
        gctx1.destroy()
        gctx2.destroy()

    print("ALL PASSED")


if __name__ == "__main__":
    main()
