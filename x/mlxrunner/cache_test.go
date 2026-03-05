package mlxrunner

import (
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// mockSnapshot is a test snapshot that tracks its range without MLX arrays.
type mockSnapshot struct {
	from, to int
}

func (s *mockSnapshot) Size() int { return 0 }
func (s *mockSnapshot) Close()    {}

// mockCache implements cache.Cache for testing without MLX initialization.
type mockCache struct {
	offset int
}

func (c *mockCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) { return nil, nil }
func (c *mockCache) State() []*mlx.Array                                     { return nil }
func (c *mockCache) Free()                                                   { c.offset = 0 }
func (c *mockCache) Offset() int                                             { return c.offset }

func (c *mockCache) Snapshot(fromOffset int) cache.Snapshot {
	return &mockSnapshot{from: fromOffset, to: c.offset}
}

func (c *mockCache) Restore(snapshot cache.Snapshot, target int) bool {
	c.offset = target
	return true
}

func (c *mockCache) Merge(parent, child cache.Snapshot) cache.Snapshot {
	if child == nil {
		return nil
	}
	return child
}

func (c *mockCache) Split(snapshot cache.Snapshot, at int) (cache.Snapshot, cache.Snapshot) {
	if snapshot == nil {
		return nil, nil
	}
	s := snapshot.(*mockSnapshot)
	if at <= s.from {
		return nil, snapshot
	}
	if at >= s.to {
		return snapshot, nil
	}
	return &mockSnapshot{from: s.from, to: at}, &mockSnapshot{from: at, to: s.to}
}

// TestSnapshotSplitsExistingChild verifies that an intermediate snapshot
// at a branch point correctly splits an existing child node and attaches
// fresh snapshots from the live caches.
func TestSnapshotSplitsExistingChild(t *testing.T) {
	// Trie: root → [1,2,3,4,5,6,7,8] with a paged-out snapshot.
	root := &trieNode{lastUsed: time.Now()}
	existing := &trieNode{
		tokens:    []int32{1, 2, 3, 4, 5, 6, 7, 8},
		endOffset: 8,
		parent:    root,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&mockSnapshot{from: 0, to: 8}},
	}
	root.children = []*trieNode{existing}

	mc := &mockCache{offset: 3}
	kvc := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		caches:     []cache.Cache{mc},
	}

	session := &cacheSession{
		cache:          kvc,
		inputs:         []int32{1, 2, 3, 10, 11, 12},
		snapshotOffset: 3,
		caches:         kvc.caches,
	}

	// Intermediate snapshot at the branch point.
	session.snapshot(false)

	// The existing child [1..8] should be split into [1,2,3] and [4,5,6,7,8].
	if len(root.children) != 1 {
		t.Fatalf("root should have 1 child (the split parent), got %d", len(root.children))
	}
	branchNode := root.children[0]
	if branchNode.endOffset != 3 {
		t.Fatalf("branch node endOffset = %d, want 3", branchNode.endOffset)
	}
	if !slices.Equal(branchNode.tokens, []int32{1, 2, 3}) {
		t.Fatalf("branch node tokens = %v, want [1,2,3]", branchNode.tokens)
	}
	if !branchNode.hasSnapshots() {
		t.Fatal("branch node should have snapshots")
	}
	if branchNode.user {
		t.Fatal("branch node should not be a user snapshot")
	}

	// Verify the snapshot is fresh (from live cache at offset 3, not from split).
	snap := branchNode.snapshots[0].(*mockSnapshot)
	if snap.from != 0 || snap.to != 3 {
		t.Fatalf("branch snapshot range = [%d,%d), want [0,3)", snap.from, snap.to)
	}

	// The old child should now be under the branch node.
	if len(branchNode.children) != 1 {
		t.Fatalf("branch node should have 1 child, got %d", len(branchNode.children))
	}
	remainder := branchNode.children[0]
	if !slices.Equal(remainder.tokens, []int32{4, 5, 6, 7, 8}) {
		t.Fatalf("remainder tokens = %v, want [4,5,6,7,8]", remainder.tokens)
	}

	// activePath should include the new branch node.
	if len(kvc.activePath) != 2 || kvc.activePath[1] != branchNode {
		t.Fatalf("activePath should be [root, branchNode], got %d nodes", len(kvc.activePath))
	}

	// snapshotOffset should be cleared.
	if session.snapshotOffset != 0 {
		t.Fatalf("snapshotOffset should be 0 after intermediate snapshot, got %d", session.snapshotOffset)
	}
}

// TestSnapshotIntermediateThenFinal verifies the full flow: an intermediate
// snapshot at a branch point followed by a final user snapshot at end of prefill.
func TestSnapshotIntermediateThenFinal(t *testing.T) {
	// Trie: root → [1,2,3,4,5] (existing conversation).
	root := &trieNode{lastUsed: time.Now()}
	existing := &trieNode{
		tokens:    []int32{1, 2, 3, 4, 5},
		endOffset: 5,
		parent:    root,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&mockSnapshot{from: 0, to: 5}},
	}
	root.children = []*trieNode{existing}

	mc := &mockCache{offset: 3}
	kvc := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		caches:     []cache.Cache{mc},
	}

	// New request diverges at token 3: [1,2,3,10,11,12].
	session := &cacheSession{
		cache:          kvc,
		inputs:         []int32{1, 2, 3, 10, 11, 12},
		snapshotOffset: 3,
		caches:         kvc.caches,
	}

	// Step 1: Intermediate snapshot at branch point (offset 3).
	session.snapshot(false)

	branchNode := root.children[0]
	if branchNode.endOffset != 3 {
		t.Fatalf("after intermediate: branch endOffset = %d, want 3", branchNode.endOffset)
	}

	// Step 2: Advance cache to end of prefill, then final snapshot.
	mc.offset = 6
	session.snapshot(true)

	// A new child [10,11,12] should be created under the branch node.
	if len(branchNode.children) != 2 {
		t.Fatalf("branch node should have 2 children (old + new), got %d", len(branchNode.children))
	}

	// Find the new child (not the [4,5] remainder).
	var newChild *trieNode
	for _, child := range branchNode.children {
		if child.tokens[0] == 10 {
			newChild = child
			break
		}
	}
	if newChild == nil {
		t.Fatal("expected new child starting with token 10")
	}
	if !slices.Equal(newChild.tokens, []int32{10, 11, 12}) {
		t.Fatalf("new child tokens = %v, want [10,11,12]", newChild.tokens)
	}
	if !newChild.user {
		t.Fatal("final snapshot should set user = true")
	}
	if !newChild.hasSnapshots() {
		t.Fatal("final snapshot should have snapshots")
	}

	// activePath should be [root, branchNode, newChild].
	if len(kvc.activePath) != 3 {
		t.Fatalf("activePath should have 3 nodes, got %d", len(kvc.activePath))
	}
	if kvc.activePath[2] != newChild {
		t.Fatal("activePath[2] should be the new child")
	}
}

// TestSnapshotSpansMultipleNodes verifies that when edgeTokens spans multiple
// existing trie nodes (e.g. a snapshot during generation duplicates output
// from a previous run), the code walks down the trie instead of creating
// a duplicate subtree.
func TestSnapshotSpansMultipleNodes(t *testing.T) {
	// Trie: root → child A [1,2,3] → child B [4,5,6]
	root := &trieNode{lastUsed: time.Now()}
	childA := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&mockSnapshot{from: 0, to: 3}},
	}
	childB := &trieNode{
		tokens:    []int32{4, 5, 6},
		endOffset: 6,
		parent:    childA,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&mockSnapshot{from: 3, to: 6}},
	}
	childA.children = []*trieNode{childB}
	root.children = []*trieNode{childA}

	mc := &mockCache{offset: 6}
	kvc := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		caches:     []cache.Cache{mc},
	}

	session := &cacheSession{
		cache:  kvc,
		inputs: []int32{1, 2, 3, 4, 5, 6, 10, 11},
		caches: kvc.caches,
	}

	session.snapshot(true)

	// Should reuse childA and childB, not create a duplicate [1,2,3,4,5,6] node.
	if len(root.children) != 1 {
		t.Fatalf("root should still have 1 child, got %d", len(root.children))
	}
	if root.children[0] != childA {
		t.Fatal("root child should still be childA")
	}

	// childB should be reused directly.
	if len(childA.children) != 1 || childA.children[0] != childB {
		t.Fatalf("childA should still have childB as only child, got %d children", len(childA.children))
	}

	// activePath should include root, childA, childB.
	if len(kvc.activePath) != 3 {
		t.Fatalf("activePath should have 3 nodes, got %d", len(kvc.activePath))
	}
	if kvc.activePath[1] != childA || kvc.activePath[2] != childB {
		t.Fatal("activePath should be [root, childA, childB]")
	}

	// childB should have fresh snapshots.
	if !childB.hasSnapshots() {
		t.Fatal("childB should have snapshots")
	}
	if !childB.user {
		t.Fatal("childB should be marked as user snapshot")
	}
}

// TestSnapshotSpansMultipleNodesAndExtends verifies that when edgeTokens
// spans existing nodes and continues past them, the last matched leaf node
// is extended in place (non-user snapshots are dropped to allow compression).
func TestSnapshotSpansMultipleNodesAndExtends(t *testing.T) {
	// Trie: root → child A [1,2,3] → child B [4,5]
	root := &trieNode{lastUsed: time.Now()}
	childA := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&mockSnapshot{from: 0, to: 3}},
	}
	childB := &trieNode{
		tokens:    []int32{4, 5},
		endOffset: 5,
		parent:    childA,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&mockSnapshot{from: 3, to: 5}},
	}
	childA.children = []*trieNode{childB}
	root.children = []*trieNode{childA}

	// Cache is at offset 8, edgeTokens = [1,2,3,4,5,6,7,8]
	// spans childA + childB + 3 new tokens.
	mc := &mockCache{offset: 8}
	kvc := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		caches:     []cache.Cache{mc},
	}

	session := &cacheSession{
		cache:  kvc,
		inputs: []int32{1, 2, 3, 4, 5, 6, 7, 8, 9},
		caches: kvc.caches,
	}

	session.snapshot(true)

	// root should still have just childA.
	if len(root.children) != 1 || root.children[0] != childA {
		t.Fatal("root should still have childA as only child")
	}

	// childB should be extended in place to [4,5,6,7,8] with no children.
	if len(childB.children) != 0 {
		t.Fatalf("childB should have no children, got %d", len(childB.children))
	}
	if !slices.Equal(childB.tokens, []int32{4, 5, 6, 7, 8}) {
		t.Fatalf("childB tokens = %v, want [4,5,6,7,8]", childB.tokens)
	}
	if childB.endOffset != 8 {
		t.Fatalf("childB endOffset = %d, want 8", childB.endOffset)
	}

	// activePath should be [root, childA, childB].
	if len(kvc.activePath) != 3 {
		t.Fatalf("activePath should have 3 nodes, got %d", len(kvc.activePath))
	}
	if kvc.activePath[2] != childB {
		t.Fatal("activePath[2] should be childB")
	}
}

// TestSnapshotExtendsLeaf verifies the basic case where snapshot extends
// a leaf node with no children (first request, no branching).
func TestSnapshotExtendsLeaf(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}

	mc := &mockCache{offset: 5}
	kvc := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		caches:     []cache.Cache{mc},
	}

	session := &cacheSession{
		cache:  kvc,
		inputs: []int32{1, 2, 3, 4, 5, 6},
		caches: kvc.caches,
	}

	session.snapshot(true)

	// Should create a child [1,2,3,4,5] under root.
	if len(root.children) != 1 {
		t.Fatalf("root should have 1 child, got %d", len(root.children))
	}
	child := root.children[0]
	if !slices.Equal(child.tokens, []int32{1, 2, 3, 4, 5}) {
		t.Fatalf("child tokens = %v, want [1,2,3,4,5]", child.tokens)
	}
	if child.endOffset != 5 {
		t.Fatalf("child endOffset = %d, want 5", child.endOffset)
	}
	if !child.user {
		t.Fatal("child should be a user snapshot")
	}
	if !child.hasSnapshots() {
		t.Fatal("child should have snapshots")
	}
}

// TestSnapshotExistingNodeInPath verifies that when a node at the target
// offset already exists in the active path, snapshot attaches to it directly.
func TestSnapshotExistingNodeInPath(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}
	child := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{child}

	mc := &mockCache{offset: 3}
	kvc := &kvCache{
		root:       root,
		activePath: []*trieNode{root, child},
		caches:     []cache.Cache{mc},
	}

	session := &cacheSession{
		cache:  kvc,
		inputs: []int32{1, 2, 3, 4, 5},
		caches: kvc.caches,
	}

	session.snapshot(true)

	if !child.hasSnapshots() {
		t.Fatal("existing node should have snapshots attached")
	}
	if !child.user {
		t.Fatal("existing node should be marked as user snapshot")
	}
}
