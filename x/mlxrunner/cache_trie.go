package mlxrunner

import (
	"fmt"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

// trieNode represents a node in the compressed prefix trie for KV cache branching.
// Each node stores a compressed edge (multiple tokens) and optional paged-out
// snapshot data per cache layer.
type trieNode struct {
	tokens    []int32 // compressed edge — multiple tokens per node
	endOffset int     // cumulative tokens from root to end of this node
	parent    *trieNode
	children  []*trieNode
	lastUsed  time.Time        // for LRU eviction
	snapshots []cache.Snapshot // per-layer paged-out snapshot data (nil if not paged out)
	user      bool             // true = explicit restore point (resist auto-merge)
}

// startOffset returns the cumulative token offset at the start of this node's edge.
func (n *trieNode) startOffset() int {
	return n.endOffset - len(n.tokens)
}

// hasSnapshots returns true if any layer has snapshot data.
func (n *trieNode) hasSnapshots() bool {
	for _, s := range n.snapshots {
		if s != nil {
			return true
		}
	}
	return false
}

// hasAllSnapshots returns true if every layer has snapshot data.
func (n *trieNode) hasAllSnapshots() bool {
	if len(n.snapshots) == 0 {
		return false
	}
	for _, s := range n.snapshots {
		if s == nil {
			return false
		}
	}
	return true
}

// findBestMatch walks the trie matching input tokens, returning the path of
// nodes traversed and the total number of tokens matched.
func findBestMatch(root *trieNode, tokens []int32) (path []*trieNode, matched int) {
	if root == nil {
		return nil, 0
	}

	path = []*trieNode{root}
	pos := 0

	node := root
	for pos < len(tokens) {
		// When multiple children share the same first token (e.g. after
		// a split), prefer the child whose full edge matches over one
		// that only partially matches. This is just being defensive - it
		// shouldn't actually happen.
		var best *trieNode
		bestMatched := 0
		bestFull := false
		for _, child := range node.children {
			edge := child.tokens
			if len(edge) == 0 {
				continue
			}
			if edge[0] != tokens[pos] {
				continue
			}
			// Count matching tokens in this child's edge.
			j := 0
			for j < len(edge) && pos+j < len(tokens) && edge[j] == tokens[pos+j] {
				j++
			}
			full := j == len(edge)
			// Prefer full edge matches; among same type, prefer longer.
			if best == nil || (full && !bestFull) || (full == bestFull && j > bestMatched) {
				best = child
				bestMatched = j
				bestFull = full
			}
		}
		if best == nil {
			break
		}

		pos += bestMatched
		path = append(path, best)

		if !bestFull {
			// Partial match within this edge
			break
		}
		node = best
	}

	return path, pos
}

// appendTokens either creates a new child node or extends the leaf in place,
// returning the node that now holds the tokens.
func (n *trieNode) appendTokens(root *trieNode, tokens []int32, endOffset int) *trieNode {
	if n == root || len(n.children) > 0 || n.hasSnapshots() {
		child := &trieNode{
			tokens:    make([]int32, len(tokens)),
			endOffset: endOffset,
			parent:    n,
			lastUsed:  n.lastUsed,
		}
		copy(child.tokens, tokens)
		n.children = append(n.children, child)
		return child
	}
	n.tokens = append(n.tokens, tokens...)
	n.endOffset = endOffset
	return n
}

// removeNode removes a leaf node from the trie.
func removeNode(node *trieNode) {
	if node.parent == nil {
		panic("removeNode called on root")
	}
	if len(node.children) != 0 {
		panic("removeNode called on non-leaf node")
	}
	p := node.parent
	for i, child := range p.children {
		if child == node {
			p.children = append(p.children[:i], p.children[i+1:]...)
			break
		}
	}
	node.parent = nil
	cache.CloseSnapshots(node.snapshots)
	node.snapshots = nil
}

// splitNode splits a node at the given token offset within its edge,
// creating a new parent node. Returns the new parent.
// `at` is relative to the node's edge (0-based index into node.tokens).
// If caches are provided, snapshots are split between parent and child
// using Cache.Split; otherwise snapshots are invalidated.
func splitNode(node *trieNode, at int, caches []cache.Cache) *trieNode {
	if at <= 0 || at >= len(node.tokens) {
		panic(fmt.Sprintf("splitNode: invalid split offset %d for node with %d tokens", at, len(node.tokens)))
	}

	// Create new parent with the prefix of the edge.
	newParent := &trieNode{
		tokens:    make([]int32, at),
		endOffset: node.startOffset() + at,
		parent:    node.parent,
		children:  []*trieNode{node},
		lastUsed:  node.lastUsed,
	}
	copy(newParent.tokens, node.tokens[:at])

	// Update the original node to have only the suffix.
	node.tokens = node.tokens[at:]
	// endOffset stays the same for the original node.

	// Split snapshots between parent and child using Cache.Split.
	if node.hasSnapshots() {
		oldSnaps := node.snapshots
		parentSnaps := make([]cache.Snapshot, len(oldSnaps))
		childSnaps := make([]cache.Snapshot, len(oldSnaps))
		for i, snap := range oldSnaps {
			if snap != nil {
				parentSnaps[i], childSnaps[i] = caches[i].Split(snap, newParent.endOffset)
				// Close original if Split produced distinct snapshots.
				if parentSnaps[i] != snap && childSnaps[i] != snap {
					snap.Close()
				}
			}
		}
		newParent.snapshots = parentSnaps
		node.snapshots = childSnaps
	} else {
		cache.CloseSnapshots(node.snapshots)
		node.snapshots = nil
	}

	// Reparent: replace node with newParent in the old parent's children.
	if node.parent != nil {
		for i, child := range node.parent.children {
			if child == node {
				node.parent.children[i] = newParent
				break
			}
		}
	}
	node.parent = newParent

	return newParent
}

// mergeWithChild merges a node with its single child: concatenates tokens,
// merges snapshot data via Cache.Merge, and removes the child.
func mergeWithChild(node *trieNode, caches []cache.Cache) {
	if len(node.children) != 1 {
		panic(fmt.Sprintf("mergeWithChild called on node with %d children", len(node.children)))
	}

	child := node.children[0]

	// Concatenate tokens.
	node.tokens = append(node.tokens, child.tokens...)
	node.endOffset = child.endOffset

	// Merge snapshots per layer.
	if len(node.snapshots) > 0 || len(child.snapshots) > 0 {
		merged := make([]cache.Snapshot, len(caches))
		for i := range caches {
			var ps, cs cache.Snapshot
			if node.snapshots != nil {
				ps = node.snapshots[i]
			}
			if child.snapshots != nil {
				cs = child.snapshots[i]
			}

			merged[i] = caches[i].Merge(ps, cs)

			// Close old snapshots that were replaced by the merge result.
			if ps != nil && ps != merged[i] {
				ps.Close()
			}
			if cs != nil && cs != merged[i] {
				cs.Close()
			}
		}
		node.snapshots = merged
	}

	// Adopt grandchildren.
	node.children = child.children
	for _, gc := range node.children {
		gc.parent = node
	}

	// Inherit user flag from child if child was a user-created snapshot node.
	node.user = child.user

	// Update lastUsed to the more recent of the two.
	if child.lastUsed.After(node.lastUsed) {
		node.lastUsed = child.lastUsed
	}

	child.parent = nil
	child.children = nil
	// Child snapshots already closed in merge loop above.
	child.snapshots = nil
}

// walkNodes calls fn for every node in the trie (depth-first).
// If fn returns false, the walk stops.
func walkNodes(root *trieNode, fn func(*trieNode) bool) {
	if root == nil {
		return
	}
	var walk func(*trieNode) bool
	walk = func(n *trieNode) bool {
		if !fn(n) {
			return false
		}
		for _, child := range n.children {
			if !walk(child) {
				return false
			}
		}
		return true
	}
	walk(root)
}
