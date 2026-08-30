"""GPU pre-flight: wait for the device to be free before loading anything.

The driver releases a finished process's memory a few seconds after it exits; a run launched
back-to-back in the same script waits for that here instead of failing at vLLM start-up.
"""

import time


def cuda_usage():
    """(used_gb, total_gb) of the current CUDA device, or None when there is no CUDA."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30, total / 2**30


def wait_gpu_free(allow_shared=False, wait_s=300, poll_s=5, threshold=0.10, probe=cuda_usage, log=print):
    """Return once less than `threshold` of the GPU memory is in use.

    Polls every `poll_s` seconds for up to `wait_s` seconds (0 = check once). If the device is still
    occupied: `allow_shared` → warn and continue, else SystemExit before any model is loaded.
    """
    usage = probe()
    if usage is None:
        return
    used, total = usage
    if used <= threshold * total:
        return
    log(f"[gpu] {used:.1f} GiB of {total:.0f} GiB in use by another process; waiting up to {wait_s:.0f} s for it to be released")
    deadline = time.monotonic() + wait_s
    waited = 0.0
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        waited += poll_s
        used, total = probe()
        if used <= threshold * total:
            log(f"[gpu] released after {waited:.0f} s ({used:.1f} GiB in use); starting")
            return
    msg = f"GPU still has {used:.1f} GiB in use after {wait_s:.0f} s (nvidia-smi shows who)."
    if allow_shared:
        log("[warning] " + msg + " Continuing (--allow_shared_gpu).")
    else:
        raise SystemExit(msg + " Refusing to start; raise --gpu_wait or pass --allow_shared_gpu to override.")
