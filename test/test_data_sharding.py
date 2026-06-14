"""
Unit tests for data sharding logic.

Covers:
- Non-overlapping splits
- Collectively exhaustive splits
- Per-rank expected sample counts
- Odd/even dataset sizes
- Various world sizes
"""
import unittest
from utils.distributed import shard_items


class TestDataShardingNonOverlapping(unittest.TestCase):
    def _all_shards(self, n, world_size):
        items = list(range(n))
        return [shard_items(items, r, world_size) for r in range(world_size)]

    def test_no_item_appears_twice(self):
        for n in [1, 2, 7, 10, 13, 50]:
            for ws in [1, 2, 3, 4]:
                shards = self._all_shards(n, ws)
                flat = [x for s in shards for x in s]
                self.assertEqual(len(flat), len(set(flat)),
                                 f"Duplicate found: n={n}, world_size={ws}")

    def test_union_covers_all_items(self):
        for n in [0, 1, 3, 10, 17, 100]:
            for ws in [1, 2, 3, 4, 5]:
                items = list(range(n))
                shards = self._all_shards(n, ws)
                union = sorted(x for s in shards for x in s)
                self.assertEqual(union, items,
                                 f"Missing items: n={n}, world_size={ws}")


class TestDataShardingCounts(unittest.TestCase):
    def test_total_count_preserved(self):
        for n in range(0, 25):
            for ws in [1, 2, 3, 4]:
                items = list(range(n))
                counts = [len(shard_items(items, r, ws)) for r in range(ws)]
                self.assertEqual(sum(counts), n, f"n={n}, ws={ws}")

    def test_max_min_count_differ_by_at_most_one(self):
        for n in range(1, 25):
            for ws in [1, 2, 3, 4]:
                items = list(range(n))
                counts = [len(shard_items(items, r, ws)) for r in range(ws)]
                self.assertLessEqual(max(counts) - min(counts), 1,
                                     f"Imbalanced: n={n}, ws={ws}, counts={counts}")

    def test_single_gpu_gets_all_items(self):
        items = list(range(20))
        self.assertEqual(shard_items(items, 0, 1), items)

    def test_empty_dataset(self):
        for ws in [1, 2, 4]:
            for r in range(ws):
                self.assertEqual(shard_items([], r, ws), [])


class TestDataShardingOrder(unittest.TestCase):
    """Shards should preserve the original ordering within each rank's slice."""

    def test_stride_sampling_order(self):
        items = list("abcdefgh")
        r0 = shard_items(items, 0, 2)
        r1 = shard_items(items, 1, 2)
        self.assertEqual(r0, ["a", "c", "e", "g"])
        self.assertEqual(r1, ["b", "d", "f", "h"])


if __name__ == "__main__":
    unittest.main()

