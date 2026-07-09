"""Auto batch size finder — ternary search for maximum throughput.

Models the throughput-vs-batch_size curve as unimodal (单峰) and uses
ternary search to find the optimum in O(log n) steps.  OOM-safe:
throughput 0 for any batch size that doesn't fit.
"""

import logging
import multiprocessing
import multiprocessing.context
import sys
import time
from configparser import ConfigParser
from io import StringIO

import torch


def _suppress_multiprocess_resource_tracker_error():
    """Suppress the ``multiprocess`` resource tracker ``_recursion_count``
    AttributeError on Python ≥ 3.12.  Harmless — fires during GC teardown.
    """
    _original_hook = sys.unraisablehook

    def _hook(unraisable):
        if (
            isinstance(unraisable.exc_value, AttributeError)
            and "_recursion_count" in str(unraisable.exc_value)
        ):
            return
        _original_hook(unraisable)

    sys.unraisablehook = _hook


_suppress_multiprocess_resource_tracker_error()


# ---------------------------------------------------------------------------
# subprocess worker — measures one batch size
# ---------------------------------------------------------------------------


def _worker(
    config_str: str,
    sample_tensors: tuple[torch.Tensor, ...],
    bs: int,
    device: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """Run forward+backward for *bs* in a fresh subprocess.

    Puts ``(bs, elapsed)`` on *result_queue*.  *elapsed* is the **total**
    wall-clock time from subprocess start (includes model loading, warmup,
    and forward+backward).  ``-1`` = OOM, ``-2`` = other error.
    """
    start = time.perf_counter()

    import configparser
    import logging
    import sys as _sys

    # Suppress multiprocess resource tracker error in subprocess (Python ≥ 3.12).
    _original_hook = _sys.unraisablehook

    def _hook(unraisable):
        if (
            isinstance(unraisable.exc_value, AttributeError)
            and "_recursion_count" in str(unraisable.exc_value)
        ):
            return
        _original_hook(unraisable)

    _sys.unraisablehook = _hook

    logging.basicConfig(level=logging.WARNING)

    config = configparser.ConfigParser()
    config.read_string(config_str)

    from src.models.imputators import imputator  # noqa: E402
    from src.models.jointModel import JointModel  # noqa: E402
    from src.models.predictors import predictor  # noqa: E402
    from src.rng import torch_seed_scope  # noqa: E402

    logger = logging.getLogger("auto_bs_worker")
    logger.addHandler(logging.NullHandler())

    base_seed = config.getint("experiment", "seed", fallback=0)
    with torch_seed_scope(
        base_seed,
        "auto_batch_worker",
        "imputer_init",
        config.get("imputer", "model"),
        bs,
    ):
        imputer_model = imputator(config.get("imputer", "model"), config)

    with torch_seed_scope(
        base_seed,
        "auto_batch_worker",
        "predictor_init",
        config.get("predictor", "model"),
        bs,
    ):
        predictor_model = predictor(config.get("predictor", "model"), config)

    joint_model = JointModel(
        imputer=imputer_model,
        predictor=predictor_model,
        optimizer_name=config.get("training", "optimizer"),
        lr=config.getfloat("training", "lr"),
        device=device,
        tb_summary_writer=None,
        logger=logger,
    )
    joint_model.configure_random_generators(base_seed)
    joint_model.to(device)

    X_N, X_NA, X_NA_mask, X_A_mask = sample_tensors[:4]

    try:
        _X_N = X_N[:bs].to(device)
        _X_NA = X_NA[:bs].to(device)
        _X_NA_mask = X_NA_mask[:bs].to(device)
        _X_A_mask = X_A_mask[:bs].to(device)

        # warmup
        joint_model(_X_NA, _X_NA_mask, _X_N, _X_A_mask)
        loss = joint_model(_X_NA, _X_NA_mask, _X_N, _X_A_mask)["loss"]
        if loss is not None:
            loss.backward()
        joint_model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        output = joint_model(_X_NA, _X_NA_mask, _X_N, _X_A_mask)
        loss = output["loss"]
        if loss is not None:
            loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        joint_model.zero_grad(set_to_none=True)
        result_queue.put((bs, elapsed))
    except RuntimeError as e:
        if "out of memory" in str(e):
            joint_model.zero_grad(set_to_none=True)
            result_queue.put((bs, -1.0))
        else:
            result_queue.put((bs, -2.0))
    except Exception as e:
        result_queue.put((bs, -2.0))


# ---------------------------------------------------------------------------
# measurement — spawns one subprocess per batch size
# ---------------------------------------------------------------------------


def _measure(
    ctx: multiprocessing.context.BaseContext,
    config_str: str,
    sample_tensors: tuple[torch.Tensor, ...],
    bs: int,
    device: str,
    logger: logging.Logger,
    best_tp: float = 0.0,
) -> float:
    """Spawn a subprocess to measure throughput for batch size *bs*.

    Returns total-throughput (samples/sec, model loading included),
    or 0.0 on OOM / error.

    If *best_tp* > 0, the parent kills the subprocess after
    ``bs / best_tp`` seconds — at that point it can no longer beat the
    best end-to-end throughput seen so far.
    """
    result_queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(config_str, sample_tensors, bs, device, result_queue),
    )
    proc.start()

    timeout = bs / best_tp if best_tp > 0 else 300.0
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        logger.info(f"  batch_size={bs:3d} → killed (>{timeout:.1f}s, best_tp={best_tp:.2f})")
        torch.cuda.empty_cache()
        return 0.0

    try:
        _bs, elapsed = result_queue.get_nowait()
    except Exception:
        logger.warning(f"batch_size={bs} subprocess exited without result")
        return 0.0

    if elapsed < 0:
        if elapsed == -1.0:
            logger.info(f"  batch_size={bs:3d} → OOM")
        else:
            logger.warning(f"  batch_size={bs:3d} → error")
        torch.cuda.empty_cache()
        return 0.0

    tp = bs / elapsed
    logger.info(f"  batch_size={bs:3d} → {elapsed:.2f}s  ({tp:.2f} samples/s)")
    torch.cuda.empty_cache()
    return tp


# ---------------------------------------------------------------------------
# ternary search over exponents — every candidate is a power of 2
# ---------------------------------------------------------------------------


def _ternary_search(
    ctx: multiprocessing.context.BaseContext,
    config_str: str,
    sample_tensors: tuple[torch.Tensor, ...],
    lo_exp: int,
    hi_exp: int,
    device: str,
    logger: logging.Logger,
    cache: dict[int, float],
) -> int:
    """Ternary search over exponents ``[lo_exp, hi_exp]``.

    Batch size = ``2^exp``.  Returns the batch size (power of 2) with the
    highest measured throughput.
    """
    l, r = lo_exp, hi_exp

    while r - l > 2:
        m1 = l + (r - l) // 3
        m2 = r - (r - l) // 3
        bs1, bs2 = 1 << m1, 1 << m2

        best_tp = max((v for v in cache.values() if v > 0), default=0.0)
        if bs1 not in cache:
            cache[bs1] = _measure(
                ctx, config_str, sample_tensors, bs1, device, logger, best_tp,
            )
        tp1 = cache[bs1]

        # Recompute — bs1 may have raised the bar.
        best_tp = max((v for v in cache.values() if v > 0), default=0.0)
        if bs2 not in cache:
            cache[bs2] = _measure(
                ctx, config_str, sample_tensors, bs2, device, logger, best_tp,
            )
        tp2 = cache[bs2]

        if tp1 <= 0 and tp2 <= 0:
            r = m1 - 1
        elif tp1 <= 0:
            l = m1
        elif tp2 <= 0:
            r = m2
        elif tp1 < tp2:
            l = m1
        else:
            r = m2

    # Evaluate remaining exponents in [l, r].
    for exp in range(l, r + 1):
        bs = 1 << exp
        if bs not in cache:
            best_tp = max((v for v in cache.values() if v > 0), default=0.0)
            cache[bs] = _measure(
                ctx, config_str, sample_tensors, bs, device, logger, best_tp,
            )

    best = max(
        (b for b in cache if cache[b] > 0),
        key=lambda b: cache[b],
        default=1,
    )
    return best


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def find_max_batch_size(
    model: torch.nn.Module,
    sample_batch: tuple[torch.Tensor, ...],
    target_batch_size: int,
    device: str,
    logger: logging.Logger,
    config: ConfigParser | None = None,
) -> int:
    """Find the power-of-2 batch size that maximises throughput.

    Ternary search over exponents ``[0, max_exp]`` where ``2^max_exp`` is
    the largest power of 2 ≤ *target_batch_size*.
    """
    if device == "cpu" or not torch.cuda.is_available():
        logger.info("Device is CPU – skipping auto batch size search.")
        return target_batch_size

    if target_batch_size <= 1:
        return target_batch_size

    if config is None:
        return _find_sync(model, sample_batch, target_batch_size, device, logger)

    X_N, X_NA, X_NA_mask, X_A_mask = sample_batch[:4]

    buf = StringIO()
    config.write(buf)
    config_str = buf.getvalue()
    sample_tensors = (X_N, X_NA, X_NA_mask, X_A_mask)

    ctx = multiprocessing.get_context("spawn")
    cache: dict[int, float] = {}

    max_exp = target_batch_size.bit_length() - 1  # 2^max_exp ≤ target_batch_size

    # Measure bs=1 first — fastest, establishes baseline.
    cache[1] = _measure(ctx, config_str, sample_tensors, 1, device, logger, 0.0)
    if cache[1] <= 0:
        logger.warning("Even batch_size=1 OOM – falling back to 1.")
        return 1

    if max_exp <= 1:
        # Only 2 and maybe 4 — test directly.
        for exp in range(1, max_exp + 1):
            bs = 1 << exp
            best_tp = max((v for v in cache.values() if v > 0), default=0.0)
            cache[bs] = _measure(ctx, config_str, sample_tensors, bs, device, logger, best_tp)
        best = max((b for b in cache if cache[b] > 0), key=lambda b: cache[b], default=1)
    else:
        logger.info(f"Ternary search over 2^[1, {max_exp}] = [2, {1 << max_exp}]")
        best = _ternary_search(
            ctx, config_str, sample_tensors,
            1, max_exp, device, logger, cache,
        )

    best_tp = cache.get(best, 0.0)
    logger.info(
        f"Auto batch size: {best} (throughput: {best_tp:.2f} samples/s, "
        f"target was {target_batch_size})"
    )
    return best


# ---------------------------------------------------------------------------
# synchronous fallback (no subprocess)
# ---------------------------------------------------------------------------


def _find_sync(
    model: torch.nn.Module,
    sample_batch: tuple[torch.Tensor, ...],
    target_batch_size: int,
    device: str,
    logger: logging.Logger,
) -> int:
    """Synchronous fallback — ternary search over exponents."""
    X_N, X_NA, X_NA_mask, X_A_mask = sample_batch[:4]
    cache: dict[int, float] = {}

    def _test(bs: int) -> float:
        if bs in cache:
            return cache[bs]
        try:
            _X_N = X_N[:bs].to(device)
            _X_NA = X_NA[:bs].to(device)
            _X_NA_mask = X_NA_mask[:bs].to(device)
            _X_A_mask = X_A_mask[:bs].to(device)

            model(_X_NA, _X_NA_mask, _X_N, _X_A_mask)
            model.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            start = time.perf_counter()

            output = model(_X_NA, _X_NA_mask, _X_N, _X_A_mask)
            loss = output["loss"]
            if loss is not None:
                loss.backward()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            model.zero_grad(set_to_none=True)

            cache[bs] = bs / elapsed
            logger.info(f"  batch_size={bs:3d} → {elapsed:.2f}s  ({cache[bs]:.2f} samples/s)")
        except RuntimeError as e:
            if "out of memory" in str(e):
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                cache[bs] = 0.0
                logger.info(f"  batch_size={bs:3d} → OOM")
            else:
                raise
        return cache[bs]

    # Measure bs=1 first.
    if _test(1) <= 0:
        logger.warning("Even batch_size=1 OOM – falling back to 1.")
        return 1

    max_exp = target_batch_size.bit_length() - 1
    if max_exp <= 1:
        for exp in range(1, max_exp + 1):
            _test(1 << exp)
        best = max((b for b in cache if cache[b] > 0), key=lambda b: cache[b])
    else:
        l, r = 1, max_exp
        while r - l > 2:
            m1 = l + (r - l) // 3
            m2 = r - (r - l) // 3
            bs1, bs2 = 1 << m1, 1 << m2
            tp1, tp2 = _test(bs1), _test(bs2)
            if tp1 <= 0 and tp2 <= 0:
                r = m1 - 1
            elif tp1 <= 0:
                l = m1
            elif tp2 <= 0:
                r = m2
            elif tp1 < tp2:
                l = m1
            else:
                r = m2

        for exp in range(l, r + 1):
            _test(1 << exp)

        best = max((b for b in cache if cache[b] > 0), key=lambda b: cache[b])

    logger.info(f"Auto batch size: {best} (throughput: {cache[best]:.2f} samples/s)")
    torch.cuda.empty_cache()
    return best
