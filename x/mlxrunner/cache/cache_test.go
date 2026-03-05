package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func TestKVCacheRestoreNilRewind(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	// Feed 10 tokens (one at a time).
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}

	// Restore(nil, 5) should rewind to 5.
	if !c.Restore(nil, 5) {
		t.Fatal("Restore(nil, 5) failed")
	}
	if c.Offset() != 5 {
		t.Fatalf("offset after Restore = %d, want 5", c.Offset())
	}

	// Restore(nil, 0) should rewind to 0.
	if !c.Restore(nil, 0) {
		t.Fatal("Restore(nil, 0) failed")
	}
	if c.Offset() != 0 {
		t.Fatalf("offset after Restore(0) = %d, want 0", c.Offset())
	}
}

func TestKVCacheRestoreNilClamp(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Restore beyond current offset should clamp.
	if !c.Restore(nil, 100) {
		t.Fatal("Restore(nil, 100) failed")
	}
	if c.Offset() != 5 {
		t.Fatalf("offset = %d, want 5", c.Offset())
	}

	// Negative target should clamp to 0.
	if !c.Restore(nil, -1) {
		t.Fatal("Restore(nil, -1) failed")
	}
	if c.Offset() != 0 {
		t.Fatalf("offset = %d, want 0", c.Offset())
	}
}

func TestKVCacheSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	// Feed 10 tokens.
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Snapshot [5, 10).
	snap := c.Snapshot(5)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	// Rewind to 5.
	c.Restore(nil, 5)
	if c.Offset() != 5 {
		t.Fatalf("offset = %d, want 5", c.Offset())
	}

	// Restore from snapshot to 10.
	if !c.Restore(snap, 10) {
		t.Fatal("Restore(snap, 10) failed")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset after restore = %d, want 10", c.Offset())
	}
}

func TestKVCacheSnapshotRestoreNeedBase(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Snapshot [5, 10).
	snap := c.Snapshot(5)

	// Free the cache completely — offset is now 0.
	c.Free()

	// Restore should fail because cache doesn't have data up to fromOffset=5.
	if c.Restore(snap, 10) {
		t.Fatal("expected Restore to fail with no base data")
	}
}

func TestKVCacheMerge(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Snapshot [0, 5).
	c.Restore(nil, 5)
	snap1 := c.Snapshot(0)

	// Restore to 5 and continue to 10.
	c.Restore(nil, 5)
	for i := 5; i < 10; i++ {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Snapshot [5, 10).
	snap2 := c.Snapshot(5)

	// Merge [0,5) + [5,10) into [0,10).
	merged := c.Merge(snap1, snap2)
	if merged == nil {
		t.Fatal("Merge returned nil")
	}

	// Restore from merged.
	c.Free()
	c2 := NewKVCache()
	if !c2.Restore(merged, 10) {
		t.Fatal("Restore(merged, 10) failed")
	}
	if c2.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c2.Offset())
	}
}

func TestKVCacheMergeNil(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}
	snap := c.Snapshot(0)

	if c.Merge(nil, snap) != nil {
		t.Fatal("Merge(nil, snap) should return nil")
	}
	if c.Merge(snap, nil) != nil {
		t.Fatal("Merge(snap, nil) should return nil")
	}
	if c.Merge(nil, nil) != nil {
		t.Fatal("Merge(nil, nil) should return nil")
	}
}

func TestRecurrentCacheSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)

	// Initialize state.
	_ = c.ConvState(1, mlx.DTypeFloat16)
	_ = c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(10)

	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}

	// Take snapshot.
	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	// Advance more.
	c.Advance(5)
	if c.Offset() != 15 {
		t.Fatalf("offset = %d, want 15", c.Offset())
	}

	// Restore(nil) should fail — recurrent state can't rewind.
	if c.Restore(nil, 10) {
		t.Fatal("Restore(nil) should fail for RecurrentCache")
	}

	// Restore from snapshot should succeed.
	if !c.Restore(snap, 10) {
		t.Fatal("Restore(snap, 10) failed")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset after restore = %d, want 10", c.Offset())
	}
}

func TestRecurrentCacheMerge(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	_ = c.ConvState(1, mlx.DTypeFloat16)
	_ = c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(5)

	snap1 := c.Snapshot(0)
	c.Advance(5)
	snap2 := c.Snapshot(5)

	// Child supersedes parent for recurrent caches.
	merged := c.Merge(snap1, snap2)
	if merged != snap2 {
		t.Fatal("Merge should return child for recurrent caches")
	}

	// Nil parent is fine — child is self-contained.
	if c.Merge(nil, snap2) != snap2 {
		t.Fatal("Merge(nil, snap) should return child")
	}
	// Nil child returns nil — can't reconstruct the range.
	if c.Merge(snap1, nil) != nil {
		t.Fatal("Merge(snap, nil) should return nil")
	}
}

func TestRecurrentCacheFree(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	_ = c.ConvState(1, mlx.DTypeFloat16)
	_ = c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(5)

	c.Free()
	if c.Offset() != 0 {
		t.Fatalf("offset not cleared after Free")
	}
}

func TestRotatingKVCacheRestoreWithinWindow(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(8)

	// Feed 5 tokens.
	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}
	if c.Offset() != 5 {
		t.Fatalf("offset = %d, want 5", c.Offset())
	}

	// Restore(nil, 3) within window should succeed.
	if !c.Restore(nil, 3) {
		t.Fatal("Restore(nil, 3) failed")
	}
	if c.Offset() != 3 {
		t.Fatalf("offset after Restore = %d, want 3", c.Offset())
	}
}

func TestRotatingKVCacheRestoreOutsideWindow(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(4)

	// Feed 10 tokens (window size 4, so positions 0-5 are evicted).
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Offset 3 is outside the window.
	if c.Restore(nil, 3) {
		t.Fatal("Restore(nil, 3) should fail when outside window")
	}
}

func TestRotatingKVCacheSnapshotRestore(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(8)

	// Prefill: feed 6 tokens via concat (multi-token).
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 6, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 6, 8)
	c.Update(k, v)
	if c.Offset() != 6 {
		t.Fatalf("offset after prefill = %d, want 6", c.Offset())
	}

	// Snapshot after prefill.
	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	// Generate 4 tokens.
	for range 4 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}
	if c.Offset() != 10 {
		t.Fatalf("offset after generation = %d, want 10", c.Offset())
	}

	// Restore to 6.
	if !c.Restore(snap, 6) {
		t.Fatal("Restore(snap, 6) failed")
	}
	if c.Offset() != 6 {
		t.Fatalf("offset after restore = %d, want 6", c.Offset())
	}
}

func TestKVCacheSplit(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	// Feed 10 tokens.
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	// Snapshot [0, 10).
	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("Snapshot returned nil")
	}

	// Split at offset 4: parent [0,4), child [4,10).
	parent, child := c.Split(snap, 4)
	if parent == nil || child == nil {
		t.Fatal("Split returned nil")
	}

	ps := parent.(*kvSnapshot)
	cs := child.(*kvSnapshot)

	if ps.fromOffset != 0 || ps.toOffset != 4 {
		t.Fatalf("parent range = [%d,%d), want [0,4)", ps.fromOffset, ps.toOffset)
	}
	if cs.fromOffset != 4 || cs.toOffset != 10 {
		t.Fatalf("child range = [%d,%d), want [4,10)", cs.fromOffset, cs.toOffset)
	}

	// Verify parent can restore to 4.
	c2 := NewKVCache()
	if !c2.Restore(parent, 4) {
		t.Fatal("Restore(parent, 4) failed")
	}
	if c2.Offset() != 4 {
		t.Fatalf("offset after parent restore = %d, want 4", c2.Offset())
	}

	// Verify child can restore to 10 (needs base at 4).
	if !c2.Restore(child, 10) {
		t.Fatal("Restore(child, 10) failed")
	}
	if c2.Offset() != 10 {
		t.Fatalf("offset after child restore = %d, want 10", c2.Offset())
	}
}

func TestKVCacheSplitBoundary(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	snap := c.Snapshot(2) // [2, 5)

	// Split at or before fromOffset → (nil, snap).
	p, ch := c.Split(snap, 2)
	if p != nil {
		t.Fatal("Split at fromOffset should return nil parent")
	}
	if ch != snap {
		t.Fatal("Split at fromOffset should return original as child")
	}

	// Split at or beyond toOffset → (snap, nil).
	p, ch = c.Split(snap, 5)
	if p != snap {
		t.Fatal("Split at toOffset should return original as parent")
	}
	if ch != nil {
		t.Fatal("Split at toOffset should return nil child")
	}

	// Split nil → (nil, nil).
	p, ch = c.Split(nil, 3)
	if p != nil || ch != nil {
		t.Fatal("Split(nil) should return (nil, nil)")
	}
}

func TestKVCacheSplitMergeRoundTrip(t *testing.T) {
	skipIfNoMLX(t)
	c := NewKVCache()

	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	snap := c.Snapshot(0)

	// Split then merge should produce equivalent snapshot.
	parent, child := c.Split(snap, 6)
	merged := c.Merge(parent, child)

	c2 := NewKVCache()
	if !c2.Restore(merged, 10) {
		t.Fatal("Restore(merged, 10) failed")
	}
	if c2.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c2.Offset())
	}
}

func TestRecurrentCacheSplit(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	_ = c.ConvState(1, mlx.DTypeFloat16)
	_ = c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(10)

	snap := c.Snapshot(0)

	// Recurrent can't split — parent is nil, child is original.
	parent, child := c.Split(snap, 5)
	if parent != nil {
		t.Fatal("RecurrentCache.Split should return nil parent")
	}
	if child != snap {
		t.Fatal("RecurrentCache.Split should return original as child")
	}
}

func TestRotatingKVCacheSplit(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(8)

	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 6, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 6, 8)
	c.Update(k, v)

	snap := c.Snapshot(0)

	// Rotating can't split — parent is nil, child is original.
	parent, child := c.Split(snap, 3)
	if parent != nil {
		t.Fatal("RotatingKVCache.Split should return nil parent")
	}
	if child != snap {
		t.Fatal("RotatingKVCache.Split should return original as child")
	}
}

func TestRotatingKVCacheFree(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRotatingKVCache(8)

	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 4, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 4, 8)
	c.Update(k, v)

	c.Free()
	if c.Offset() != 0 {
		t.Fatalf("offset not cleared after Free")
	}
}
