package mlxrunner

import (
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func newTestTrie(tokens []int32) *trieNode {
	root := &trieNode{lastUsed: time.Now()}
	if len(tokens) > 0 {
		child := &trieNode{
			tokens:    slices.Clone(tokens),
			endOffset: len(tokens),
			parent:    root,
			lastUsed:  time.Now(),
		}
		root.children = []*trieNode{child}
	}
	return root
}

func TestFindBestMatchEmpty(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}
	path, matched := findBestMatch(root, []int32{1, 2, 3})
	if matched != 0 {
		t.Fatalf("expected 0 matched, got %d", matched)
	}
	if len(path) != 1 || path[0] != root {
		t.Fatalf("expected path with just root, got %d nodes", len(path))
	}
}

func TestFindBestMatchFull(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	path, matched := findBestMatch(root, []int32{1, 2, 3, 4, 5})
	if matched != 5 {
		t.Fatalf("expected 5 matched, got %d", matched)
	}
	if len(path) != 2 {
		t.Fatalf("expected 2 nodes in path, got %d", len(path))
	}
}

func TestFindBestMatchPartial(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	path, matched := findBestMatch(root, []int32{1, 2, 3, 6, 7})
	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if len(path) != 2 {
		t.Fatalf("expected 2 nodes in path (root + partial child), got %d", len(path))
	}
}

func TestFindBestMatchNoMatch(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3})
	path, matched := findBestMatch(root, []int32{4, 5, 6})
	if matched != 0 {
		t.Fatalf("expected 0 matched, got %d", matched)
	}
	if len(path) != 1 {
		t.Fatalf("expected just root, got %d", len(path))
	}
}

func TestSplitNode(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	child := root.children[0]

	// Split at position 3: [1,2,3] becomes parent, [4,5] becomes child.
	newParent := splitNode(child, 3, nil)

	if newParent.endOffset != 3 {
		t.Fatalf("newParent.endOffset = %d, want 3", newParent.endOffset)
	}
	if !slices.Equal(newParent.tokens, []int32{1, 2, 3}) {
		t.Fatalf("newParent.tokens = %v, want [1,2,3]", newParent.tokens)
	}
	if !slices.Equal(child.tokens, []int32{4, 5}) {
		t.Fatalf("child.tokens = %v, want [4,5]", child.tokens)
	}
	if child.endOffset != 5 {
		t.Fatalf("child.endOffset = %d, want 5", child.endOffset)
	}
	if len(newParent.children) != 1 || newParent.children[0] != child {
		t.Fatal("newParent should have child as only child")
	}
	if child.parent != newParent {
		t.Fatal("child.parent should be newParent")
	}
	if newParent.parent != root {
		t.Fatal("newParent.parent should be root")
	}
	if len(root.children) != 1 || root.children[0] != newParent {
		t.Fatal("root should have newParent as only child")
	}
}

func TestMergeWithChild(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	child := root.children[0]

	// Split to create two nodes: [1,2,3] and [4,5].
	newParent := splitNode(child, 3, nil)

	// Merge them back.
	mergeWithChild(newParent, nil)

	if !slices.Equal(newParent.tokens, []int32{1, 2, 3, 4, 5}) {
		t.Fatalf("merged tokens = %v, want [1,2,3,4,5]", newParent.tokens)
	}
	if newParent.endOffset != 5 {
		t.Fatalf("merged endOffset = %d, want 5", newParent.endOffset)
	}
	if len(newParent.children) != 0 {
		t.Fatalf("merged node should have no children, got %d", len(newParent.children))
	}
}

func TestRemoveNode(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3})
	child := root.children[0]

	removeNode(child)
	if len(root.children) != 0 {
		t.Fatalf("root should have no children after removal, got %d", len(root.children))
	}
}

func TestWalkNodes(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3})
	child := root.children[0]
	grandchild := &trieNode{
		tokens:    []int32{4, 5},
		endOffset: 5,
		parent:    child,
		lastUsed:  time.Now(),
	}
	child.children = []*trieNode{grandchild}

	var visited []*trieNode
	walkNodes(root, func(n *trieNode) bool {
		visited = append(visited, n)
		return true
	})

	if len(visited) != 3 {
		t.Fatalf("expected 3 nodes visited, got %d", len(visited))
	}
}

func TestMergeWithChildSnapshots(t *testing.T) {
	skipIfNoMLX(t)
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	child := root.children[0]

	// Split to create parent [1,2,3] and child [4,5].
	newParent := splitNode(child, 3, nil)

	// Create a mock cache for merge operations.
	mockCache := cache.NewKVCache()

	// Give both nodes snapshot data (using nil values to test merge logic).
	newParent.snapshots = []cache.Snapshot{nil}
	child.snapshots = []cache.Snapshot{nil}

	mergeWithChild(newParent, []cache.Cache{mockCache})

	if len(newParent.snapshots) != 1 {
		t.Fatalf("expected 1 snapshot entry, got %d", len(newParent.snapshots))
	}
}

func TestFindBestMatchMultipleBranches(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}

	branch1 := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	branch2 := &trieNode{
		tokens:    []int32{4, 5, 6},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{branch1, branch2}

	// Match branch 1.
	path, matched := findBestMatch(root, []int32{1, 2, 3, 7})
	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if len(path) != 2 || path[1] != branch1 {
		t.Fatal("expected to match branch1")
	}

	// Match branch 2.
	path, matched = findBestMatch(root, []int32{4, 5, 6, 8})
	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if len(path) != 2 || path[1] != branch2 {
		t.Fatal("expected to match branch2")
	}

	// Match neither.
	_, matched = findBestMatch(root, []int32{7, 8, 9})
	if matched != 0 {
		t.Fatalf("expected 0 matched, got %d", matched)
	}
}

func TestFindBestMatchPrefersFullEdge(t *testing.T) {
	// Reproduce the cache miss scenario: two siblings share a token prefix
	// but one has a shorter edge that matches fully while the other has a
	// longer edge that only matches partially. findBestMatch must prefer
	// the full-edge match to avoid an unnecessary path switch.
	root := &trieNode{lastUsed: time.Now()}

	// Shared prefix: tokens [1,2,3]
	shared := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{shared}

	// Two siblings under shared, both starting with token 10.
	// longer has tokens [10,11,12,13,14] (5 tokens, ends at 8)
	// shorter has tokens [10,11,12] (3 tokens, ends at 6)
	longer := &trieNode{
		tokens:    []int32{10, 11, 12, 13, 14},
		endOffset: 8,
		parent:    shared,
		lastUsed:  time.Now(),
	}
	shorter := &trieNode{
		tokens:    []int32{10, 11, 12},
		endOffset: 6,
		parent:    shared,
		lastUsed:  time.Now(),
	}
	// Put longer first so naive first-match would pick it.
	shared.children = []*trieNode{longer, shorter}

	// Input matches shared [1,2,3] then shorter [10,11,12] fully,
	// then continues with new tokens. A naive first-match would pick
	// longer and get a partial match of 3/5 tokens, same matched count
	// but triggering an unnecessary split and path switch.
	input := []int32{1, 2, 3, 10, 11, 12, 99, 100}
	path, matched := findBestMatch(root, input)

	if matched != 6 {
		t.Fatalf("expected 6 matched, got %d", matched)
	}
	if len(path) != 3 {
		t.Fatalf("expected 3 nodes in path, got %d", len(path))
	}
	if path[2] != shorter {
		t.Fatal("expected findBestMatch to pick shorter (full edge match), not longer (partial)")
	}
}

func TestFindBestMatchPrefersLongerPartial(t *testing.T) {
	// When no child fully matches, prefer the child with the longer
	// partial match (more tokens cached).
	root := &trieNode{lastUsed: time.Now()}

	child1 := &trieNode{
		tokens:    []int32{1, 2, 3, 4, 5},
		endOffset: 5,
		parent:    root,
		lastUsed:  time.Now(),
	}
	child2 := &trieNode{
		tokens:    []int32{1, 2, 9},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{child2, child1}

	// Input diverges at token index 3 — matches 3 of child1's 5 tokens
	// but only 2 of child2's 3 tokens (child2 has 9 at index 2).
	input := []int32{1, 2, 3, 7, 8}
	path, matched := findBestMatch(root, input)

	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if path[1] != child1 {
		t.Fatal("expected findBestMatch to pick child1 (longer partial match)")
	}
}

func TestSplitNodeWithSnapshots(t *testing.T) {
	skipIfNoMLX(t)
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	child := root.children[0]

	// Give the child a KV cache snapshot covering [0, 5).
	kv := cache.NewKVCache()
	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		kv.Update(k, v)
	}
	child.snapshots = []cache.Snapshot{kv.Snapshot(0)}
	child.user = true

	caches := []cache.Cache{kv}

	// Split at position 3: parent [1,2,3], child [4,5].
	newParent := splitNode(child, 3, caches)

	// Parent should have snapshot [0, 3) but not be pinned
	// (captureIntermediateSnapshot pins it later with complete data).
	if !newParent.hasSnapshots() {
		t.Fatal("newParent should have snapshots after split")
	}
	if newParent.user {
		t.Fatal("newParent should not be a user snapshot after splitNode")
	}

	// Child should have snapshot [3, 5).
	if !child.hasSnapshots() {
		t.Fatal("child should have snapshots after split")
	}
	if !child.user {
		t.Fatal("child should remain a user snapshot")
	}
}

func TestSplitNodeWithoutCaches(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	child := root.children[0]

	child.snapshots = []cache.Snapshot{nil} // placeholder

	// Split without caches should invalidate snapshots.
	newParent := splitNode(child, 3, nil)

	if newParent.hasSnapshots() {
		t.Fatal("newParent should not have snapshots without caches")
	}
	if child.hasSnapshots() {
		t.Fatal("child should not have snapshots without caches")
	}
}

func TestStartOffset(t *testing.T) {
	node := &trieNode{
		tokens:    []int32{4, 5, 6},
		endOffset: 6,
	}
	if node.startOffset() != 3 {
		t.Fatalf("startOffset = %d, want 3", node.startOffset())
	}
}
