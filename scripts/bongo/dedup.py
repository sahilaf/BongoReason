"""MinHash + LSH near-duplicate detection.

``datasketch`` is not installed here and adding it would mean an extra Colab
install step, so this is hand-rolled on numpy.  With 128 permutations in 8 bands
of 16 rows the banding curve puts the 50% detection point near Jaccard 0.88,
which suits a 0.90 target.  Banding only proposes candidates; every pair is
confirmed with an exact Jaccard before it is treated as a duplicate.
"""

import hashlib
from collections import defaultdict

import numpy as np

MERSENNE = (1 << 61) - 1
BANDS, ROWS = 8, 16
NUM_PERMS = BANDS * ROWS

# A bucket this large means the shingle sets are degenerate (near-empty
# questions); pairing them all would be quadratic and meaningless.
MAX_BUCKET = 200


def stable_hash(s: str) -> int:
    """64-bit hash that is identical across processes and machines.

    Python's builtin ``hash()`` is salted per interpreter (PYTHONHASHSEED), so
    using it here would make dedup and decontamination non-reproducible between
    runs — the pipeline must give the same dataset every time.
    """
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def signatures(shingle_sets, num_perms=NUM_PERMS, seed=0):
    """Return an (n, num_perms) uint64 MinHash signature matrix."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE, size=num_perms, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=num_perms, dtype=np.uint64)

    sigs = np.full((len(shingle_sets), num_perms), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, shingles in enumerate(shingle_sets):
        if not shingles:
            continue
        hashes = np.fromiter(
            (stable_hash(s) for s in sorted(shingles)),
            dtype=np.uint64,
            count=len(shingles),
        )
        sigs[i] = ((np.outer(hashes, a) + b) % MERSENNE).min(axis=0)
    return sigs


def candidate_pairs(sigs):
    """Yield index pairs that share at least one full band."""
    n = sigs.shape[0]
    seen = set()
    for band in range(BANDS):
        lo, hi = band * ROWS, (band + 1) * ROWS
        buckets = defaultdict(list)
        for i in range(n):
            buckets[sigs[i, lo:hi].tobytes()].append(i)
        for members in buckets.values():
            if not 2 <= len(members) <= MAX_BUCKET:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pair = (members[x], members[y])
                    if pair not in seen:
                        seen.add(pair)
                        yield pair


def cross_candidates(query_sigs, ref_sigs):
    """Yield (query_index, ref_index) pairs sharing a band across two corpora."""
    seen = set()
    for band in range(BANDS):
        lo, hi = band * ROWS, (band + 1) * ROWS
        buckets = defaultdict(list)
        for j in range(ref_sigs.shape[0]):
            buckets[ref_sigs[j, lo:hi].tobytes()].append(j)
        for i in range(query_sigs.shape[0]):
            for j in buckets.get(query_sigs[i, lo:hi].tobytes(), ()):
                if (i, j) not in seen:
                    seen.add((i, j))
                    yield i, j


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class Union:
    """Union-find over record indices."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)
