"""NAS EP using cuPyNumeric array operations and the exact 46-bit LCG.

The default path computes pair seeds with timed skip-ahead powers, so each
stream's 256 pairs can be evaluated as an array. It still materializes large
intermediates and evaluates masked rejected-pair math, unlike cuNumeric's
single fused per-batch kernel. It does not use custom
tasks or host-generated random samples.
"""

import math
import cupynumeric as np
import numpy as host_np
from legate.core import TaskTarget, get_legate_runtime

from core import register_benchmark

MK, NQ = 8, 10
SEED, MULTIPLIER = 271828183.0, 1220703125.0
EPSILON = 1.0e-8
CLASSES = {
    "S": (24, -3.247834652034740e3, -6.958407078382297e3),
    "W": (25, -2.863319731645753e3, -6.320053679109499e3),
    "A": (28, -4.295875165629892e3, -1.580732573678431e4),
    "B": (30, 4.033815542441498e4, -2.660669192809235e4),
    "C": (32, 4.764367927995374e4, -8.084072988043731e4),
    "D": (36, 1.982481200946593e5, -1.020596636361769e5),
    "E": (40, -5.319717441530e5, -3.688834557731e5),
    # Weak scaling at class B's samples per GPU, named <class below>.<gpus>:
    # B.2 on 2 GPUs, class C on 4, C.8 on 8. No NAS reference.
    "B.2": (31, None, None),
    "C.8": (33, None, None),
    # Large weak scaling (configs/multi_gpu/large) at class C's samples per GPU:
    # class C on 1 GPU, C.8 (the same m=33 size) on 2, C.4 on 4, C.16 on 8.
    "C.4": (34, None, None),
    "C.16": (35, None, None),
}


def randlc_scalar(x, a=MULTIPLIER):
    r23, t23, r46, t46 = 2.0**-23, 2.0**23, 2.0**-46, 2.0**46
    a1 = int(r23*a); a2 = a-t23*a1
    x1 = int(r23*x); x2 = x-t23*x1
    t1 = a1*x2+a2*x1; z = t1-t23*int(r23*t1)
    t3 = t23*z+a2*x2; x = t3-t46*int(r46*t3)
    return x, r46*x


def ipow46(a, exponent):
    if exponent == 0:
        return 1.0
    q, r, n = a, 1.0, exponent
    while n > 1:
        n2 = n//2
        if 2*n2 == n:
            q, _ = randlc_scalar(q, q); n = n2
        else:
            r, _ = randlc_scalar(r, q); n -= 1
    return randlc_scalar(r, q)[0]


def mul_mod46(x, multiplier):
    r23, t23, r46, t46 = 2.0**-23, 2.0**23, 2.0**-46, 2.0**46
    a1 = int(r23*multiplier); a2 = multiplier-t23*a1
    x1 = np.trunc(r23*x); x2 = x-t23*x1
    t1 = a1*x2+a2*x1; z = t1-t23*np.trunc(r23*t1)
    t3 = t23*z+a2*x2
    return t3-t46*np.trunc(r46*t3)


def mul_mod46_powers(x, multiplier):
    """The same exact modular product with an array of skip-ahead powers."""
    r23, t23, r46, t46 = 2.0**-23, 2.0**23, 2.0**-46, 2.0**46
    a1 = np.trunc(r23*multiplier); a2 = multiplier-t23*a1
    x1 = np.trunc(r23*x); x2 = x-t23*x1
    t1 = a1*x2+a2*x1; z = t1-t23*np.trunc(r23*t1)
    t3 = t23*z+a2*x2
    return t3-t46*np.trunc(r46*t3)


def pair_powers():
    # This RNG work is performed inside run(), as in the scalar-kernel models.
    powers = host_np.empty(1 << MK, dtype=host_np.float64)
    power = MULTIPLIER
    pair_jump, _ = randlc_scalar(MULTIPLIER, MULTIPLIER)
    for pair in range(1 << MK):
        powers[pair] = power
        power, _ = randlc_scalar(power, pair_jump)
    return np.asarray(powers)


def batch_histogram_scalar(batch):
    """A few untimed CPU spot checks for the histogram partials."""
    seed = SEED
    power = ipow46(MULTIPLIER, 2*(1 << MK))
    while batch:
        if batch & 1:
            seed, _ = randlc_scalar(seed, power)
        power, _ = randlc_scalar(power, power)
        batch >>= 1
    q = [0] * NQ
    for _ in range(1 << MK):
        seed, u1 = randlc_scalar(seed)
        seed, u2 = randlc_scalar(seed)
        x1, x2 = 2.0*u1-1.0, 2.0*u2-1.0
        radius = x1*x1+x2*x2
        if radius <= 1.0:
            scale = math.sqrt(-2.0*math.log(radius)/radius)
            bin = int(max(abs(x1*scale), abs(x2*scale)))
            if bin < NQ:
                q[bin] += 1
    return q


class NASEmbarrassinglyParallel:
    name = "nas_ep"
    throughput_label = "G random numbers/s"
    correctness_reference = "NPB-GPU"

    def __init__(self, T, N, M, **kwargs):
        self.T, self.N, self.M = T, N, M
        self.class_name = str(kwargs.pop("class", "S")).upper()
        if kwargs:
            raise ValueError(f"Unknown NAS EP options: {', '.join(kwargs)}")
        if self.class_name not in CLASSES:
            raise ValueError(f"Unknown NAS EP class {self.class_name}")
        self.m = CLASSES[self.class_name][0]
        expected = 1 << (self.m+1)
        if T is not np.float64 or M != 1 or N != expected:
            raise ValueError(
                f"NAS EP class {self.class_name} requires Float64, N={expected}, M=1"
            )
        self.batches = 1 << (self.m-MK)

    def dims(self):
        return self.N, self.M

    def correctness_dims(self):
        return self.N, self.M

    def initialize(self):
        values = [np.zeros(self.batches, dtype=np.float64) for _ in range(13)]
        # Built on the device: host-attached masks are re-copied on every use.
        indices = np.arange(self.batches, dtype=np.uint64)
        masks = [((indices >> bit) & 1).astype(np.float64) for bit in range(self.m-MK)]
        return {"values": values, "masks": masks}

    def reset(self, state):
        state["values"][0].fill(SEED)
        for value in state["values"][1:]:
            value.fill(0.0)

    def run(self, state):
        values = state["values"]
        seed, power = values[0], ipow46(MULTIPLIER, 2*(1 << MK))
        for mask in state["masks"]:
            candidate = mul_mod46(seed, power)
            seed = seed + mask*(candidate-seed)
            power, _ = randlc_scalar(power, power)

        powers = pair_powers()
        q, sx, sy = values[1:11], values[11], values[12]
        # Cap each slab at 2^25 pairs per GPU, so weak scaling keeps the
        # per-GPU slab size. Slabs are (pairs, batches): row-major sums over
        # axis 0 are much faster than over a short trailing axis.
        gpus = max(1, get_legate_runtime().get_machine().count(TaskTarget.GPU))
        chunk = min(1 << MK, max(1, (gpus << 25)//self.batches))
        for lo in range(0, 1 << MK, chunk):
            hi = min(lo + chunk, 1 << MK)
            seed1 = mul_mod46_powers(seed[None, :], powers[lo:hi, None])
            seed2 = mul_mod46(seed1, MULTIPLIER)
            u1, u2 = (2.0**-46)*seed1, (2.0**-46)*seed2
            x1, x2 = 2.0*u1-1.0, 2.0*u2-1.0
            radius = x1*x1 + x2*x2
            accepted = np.minimum(np.floor(1.0/radius), 1.0)
            safe = np.minimum(radius, 1.0)
            scale = np.sqrt(-2.0*np.log(safe)/safe)
            g1, g2 = x1*scale, x2*scale
            # Rejected pairs get bin -1, so each bin is one compare-and-count.
            bins = np.where(accepted > 0, np.floor(np.maximum(np.abs(g1), np.abs(g2))), -1.0)
            for bin in range(NQ):
                q[bin] += np.sum(bins == float(bin), axis=0)
            sx += np.sum(accepted*g1, axis=0)
            sy += np.sum(accepted*g2, axis=0)
        return values

    def check_correctness(self):
        state = self.initialize()
        self.reset(state)
        values = self.run(state)
        sx = float(host_np.asarray(values[11]).sum())
        sy = float(host_np.asarray(values[12]).sum())
        _, expected_x, expected_y = CLASSES[self.class_name]
        if expected_x is None:  # weak-scaling size without a NAS reference
            return "skipped"
        ok = (abs((sx-expected_x)/expected_x) <= EPSILON and
              abs((sy-expected_y)/expected_y) <= EPSILON)
        if ok:
            q = [host_np.asarray(value) for value in values[1:11]]
            for batch in (0, 1, self.batches//2, self.batches-1):
                expected = batch_histogram_scalar(batch)
                if any(q[bin][batch] != expected[bin] for bin in range(NQ)):
                    ok = False
                    break
        return "pass" if ok else "fail"


register_benchmark("nas_ep", NASEmbarrassinglyParallel)
