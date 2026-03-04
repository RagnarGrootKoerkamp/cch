#![feature(iter_array_chunks, portable_simd)]
use epserde::deser::Deserialize;
use epserde::ser::Serialize;
use itertools::Itertools;
use log::{debug, info, trace};
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use std::{collections::HashMap, ops::Range, path::Path, simd::cmp::SimdOrd};

#[cfg(feature = "sse")]
const L: usize = 4;
#[cfg(feature = "sse")]
type S = std::simd::i32x4;
#[cfg(not(any(feature = "sse", feature = "avx512")))]
const L: usize = 8;
#[cfg(not(any(feature = "sse", feature = "avx512")))]
type S = std::simd::i32x8;
#[cfg(feature = "avx512")]
const L: usize = 16;
#[cfg(feature = "avx512")]
type S = std::simd::i32x16;

mod dijkstra;

type NodeId = u32;
type EdgeId = u32;
type Weight = i32;

const INVALID_ID: u32 = u32::MAX;
const W_INF: Weight = i32::MAX / 3;

/// Input: a path to a file
/// Output: a vec of binary-encoded u32s
fn read_vec(path: &Path) -> Vec<u32> {
    std::fs::read(path)
        .expect(&format!("reading {}", path.display()))
        .into_iter()
        .array_chunks()
        .map(u32::from_le_bytes)
        .collect()
}

#[derive(Clone, Copy, epserde::Epserde)]
#[repr(C)]
#[epserde_zero_copy]
struct Node {
    // rank is implicit via ID (=index) of self.
    // rank: u32,
    /// Index into list of edges of the first outgoing edge.
    first_edge_idx: EdgeId,
    /// Index into list of ranges of the first outgoing edge range.
    first_range_idx: EdgeId,
    /// Parent ID.
    parent: NodeId,
}

const UP: usize = 0;
const DOWN: usize = 1;

/// Edge goes to `head`.
/// Tail is implicit via `first_out` of the source node.
#[derive(Clone, Copy, epserde::Epserde)]
#[repr(C)]
#[epserde_zero_copy]
struct HalfEdge {
    /// Edge goes to `head`.
    head: NodeId,
    /// Weight for up and down direction.
    weight: [Weight; 2],
    /// For use during perfect customization.
    deleted: [bool; 2],
}

/// Edge from `tail` to `head`.
#[derive(Clone, Copy, epserde::Epserde)]
#[repr(C)]
#[epserde_zero_copy]
struct Edge {
    tail: NodeId,
    head: NodeId,
    weight: Weight,
}

impl Edge {
    /// return (low, high) end.
    fn undirected(&self) -> (NodeId, NodeId) {
        (self.tail.min(self.head), self.tail.max(self.head))
    }
    /// Direction of the edge.
    fn dir(&self) -> usize {
        if self.tail < self.head {
            UP
        } else {
            DOWN
        }
    }
}

/// A range of 8 EdgeIds with consecutive `head` values.
///
/// (end is omitted because it's always exactly 8)
#[derive(Debug, Clone, Copy, epserde::Epserde)]
#[repr(C)]
#[epserde_zero_copy]
struct EdgeRange {
    /// Head of the first edge in the range.
    first_head: NodeId,
}

#[derive(epserde::Epserde)]
pub struct CCH {
    /// The number of nodes
    n: u32,

    /// The input edges and weights, without any permutation applied to them.
    input_edges: Vec<Edge>,

    /// Input node `i` maps to internal ID `rank[i]`.
    rank: Vec<NodeId>,

    /// The sorted nodes and their parents.
    nodes: Vec<Node>,
    /// The undirected CCH edges.
    edges: Vec<HalfEdge>,
    /// Ranges of edges
    edge_ranges: Vec<EdgeRange>,
    /// Flattened edge weights
    edge_weigths: [Vec<Weight>; 2],

    /// Up and down distance cache to each node for queries.
    dist: [Vec<Weight>; 2],
}

impl CCH {
    /// path: "graph/europe"
    #[inline(never)]
    pub fn new(path: &Path) -> Self {
        let n;
        let mut input_edges = vec![];
        let mut adj;
        // the internal ID of each node.
        let mut rank;
        {
            // Read files.
            info!("Reading..");
            let order = read_vec(&path.with_added_extension("order"));
            let first_out = read_vec(&path.with_added_extension("first_out"));
            let head = read_vec(&path.with_added_extension("head"));
            // Also read the weight already.
            let weight = read_vec(&path.with_added_extension("dist"));

            n = order.len();

            // Compute target rank of each node, ie inverse of `order`.
            info!("Rank..");
            rank = {
                let mut rank = vec![0; n];
                for (i, &node) in order.iter().enumerate() {
                    rank[node as usize] = i as u32;
                }
                rank
            };

            // Convert to adjacency list and reorder according to rank.
            info!("To adjacency..");
            adj = vec![Vec::new(); n];
            for i in 0..n {
                for j in first_out[i]..first_out[i + 1] {
                    let head = head[j as usize];
                    let weight = weight[j as usize] as i32;
                    assert!(weight < W_INF / 2, "weight {weight} is too large");
                    input_edges.push(Edge {
                        tail: i as u32,
                        head,
                        weight,
                    });

                    // permute
                    let tail = rank[i as usize];
                    let head = rank[head as usize];
                    let (u, v) = (tail.min(head), tail.max(head));
                    adj[u as usize].push(v);
                }
            }

            for i in 0..n {
                adj[i].sort_unstable();
                adj[i].dedup();
            }
        };

        if true {
            // Slightly reduced chordal closure to compute parents.
            info!("Compute parents..");
            let mut parents = vec![INVALID_ID; n];
            for u in 0..n as u32 {
                let mut nbs = std::mem::take(&mut adj[u as usize]);
                nbs.sort_unstable();
                nbs.dedup();
                if u == n as u32 - 1 {
                    break;
                }
                let parent = nbs[0];
                parents[u as usize] = parent;
                assert!(parent > u);
                adj[parent as usize].extend(&nbs[1..]);
                adj[u as usize] = nbs;
            }

            info!("Compute DFS order..");
            // Re-order the nodes by a DFS traversal.
            let mut done = vec![false; n];
            let mut new_order = vec![];
            let mut buf = vec![];
            for u in 0..n as u32 {
                if done[u as usize] {
                    continue;
                }
                let mut p = u;
                while p != INVALID_ID && !done[p as usize] {
                    done[p as usize] = true;
                    buf.push(p);
                    p = parents[p as usize];
                }
                new_order.extend(buf.drain(..).rev());
            }
            new_order.reverse();
            assert_eq!(new_order.len(), n);

            info!("Apply permutation..");
            let mut new_rank = vec![0; n];
            for (i, &node) in new_order.iter().enumerate() {
                new_rank[node as usize] = i as u32;
            }
            // remap all nodes.
            let mut new_adj = vec![Vec::new(); n];
            for u in 0..n as u32 {
                let new_u = new_rank[u as usize];
                for &v in &adj[u as usize] {
                    let new_v = new_rank[v as usize];
                    new_adj[new_u as usize].push(new_v);
                }
            }

            // Update total permutation on input IDs
            for i in 0..n {
                rank[i] = new_rank[rank[i] as usize];
            }

            adj = new_adj;
        }

        // The parent of each node. Computed during chordal completion.
        let mut nodes = vec![];
        let mut edges = vec![];

        // Chordal closure: the upper-context of each node is a clique.
        info!("Chordal closure..");
        for u in 0..n as u32 {
            let mut nbs = std::mem::take(&mut adj[u as usize]);
            nbs.sort_unstable();
            nbs.dedup();
            if u == n as u32 - 1 {
                assert_eq!(nbs.len(), 0);
                nodes.push(Node {
                    first_edge_idx: edges.len() as u32,
                    first_range_idx: 0,
                    parent: INVALID_ID,
                });
                break;
            }
            let parent = nbs[0];
            assert!(parent > u);

            nodes.push(Node {
                first_edge_idx: edges.len() as u32,
                first_range_idx: 0,
                parent,
            });

            // store up-edges of u
            edges.extend(nbs.iter().map(|&head| HalfEdge {
                head,
                weight: [W_INF; 2],
                deleted: [false; 2],
            }));

            // copy neighbourhood to parent
            adj[parent as usize].extend(&nbs[1..]);
        }

        nodes.push(Node {
            first_edge_idx: edges.len() as u32,
            first_range_idx: 0,
            parent: INVALID_ID,
        });

        for i in 0..n as u32 {
            assert!(nodes[i as usize].parent > i);
        }

        info!("n:           {n:>9}");
        info!("input edges: {:>9}", input_edges.len());
        info!("added edges: {:>9}", edges.len() - input_edges.len());
        info!("total edges: {:>9}", edges.len());

        // Add buffers at the end.
        nodes.extend(
            std::iter::repeat(Node {
                first_edge_idx: 0,
                first_range_idx: 0,
                parent: INVALID_ID,
            })
            .take(2 * L),
        );
        edges.extend(
            std::iter::repeat(HalfEdge {
                head: 0,
                weight: [W_INF; 2],
                deleted: [false; 2],
            })
            .take(2 * L),
        );

        Self {
            n: n as u32,
            input_edges,
            rank,
            nodes,
            edges,
            edge_weigths: [vec![], vec![]],
            edge_ranges: vec![],
            dist: [vec![], vec![]],
        }
    }

    #[allow(unused)]
    fn parent_stats(&self) {
        let mut child_cnt = vec![0; self.n as usize];
        for u in 0..self.n as u32 {
            let p = self.nodes[u as usize].parent;
            if p != INVALID_ID {
                child_cnt[p as usize] += 1;
            }
        }
        debug!(
            "Last 1000 degs: {:?}",
            child_cnt.iter().rev().take(1000).collect_vec()
        );

        let mut deg_cnt = HashMap::new();
        for cnt in child_cnt {
            *deg_cnt.entry(cnt).or_insert(0) += 1;
        }
        let mut deg_cnt = deg_cnt.into_iter().collect_vec();
        deg_cnt.sort_unstable_by_key(|(deg, _)| *deg);
        debug!("Parent degree distribution:");
        for (deg, cnt) in deg_cnt {
            debug!("  {deg:>3}: {cnt:>9}");
        }
    }

    #[allow(unused)]
    fn save(&mut self, path: &Path) {
        info!("Saving..");
        self.input_edges.clear();
        self.dist[0].clear();
        self.dist[1].clear();
        let mut file = std::fs::File::create(path.with_added_extension("cch")).unwrap();
        unsafe { self.serialize(&mut file).unwrap() };
    }
    #[allow(unused)]
    fn read(path: &Path) -> Self {
        info!("Reading..");
        let mut path = path.with_added_extension("cch");
        unsafe { Self::load_full(&path).unwrap() }
    }

    fn edge_range(&self, u: NodeId) -> std::ops::Range<usize> {
        let i = self.nodes[u as usize].first_edge_idx as usize;
        let j = self.nodes[u as usize + 1].first_edge_idx as usize;
        i..j
    }

    fn find_edge_index(&self, u: NodeId, v: NodeId) -> usize {
        assert!(u < v);

        // Linear scan
        // let edge_range = self.edge_range(u);
        // edge_range.start
        //     + self.edges[edge_range]
        //         .iter()
        //         .position(|e| e.head == v)
        //         .unwrap();

        // Binary search
        let edge_range = self.edge_range(u);
        let idx = edge_range.start
            + self.edges[edge_range]
                .binary_search_by_key(&v, |e| e.head)
                .unwrap();
        assert_eq!(self.edges[idx].head, v);
        idx
    }

    #[allow(unused)]
    fn find_edge(&self, u: NodeId, v: NodeId) -> &HalfEdge {
        let idx = self.find_edge_index(u, v);
        &self.edges[idx]
    }

    fn find_edge_mut(&mut self, u: NodeId, v: NodeId) -> &mut HalfEdge {
        let idx = self.find_edge_index(u, v);
        &mut self.edges[idx]
    }

    /// Use the already-permuted weights to customize all edges.
    #[inline(never)]
    fn customize(&mut self, perfect: bool) {
        // Copy the input weights into the CCH edges.
        info!("Set weights..");
        for mut e in std::mem::take(&mut self.input_edges) {
            e.tail = self.rank[e.tail as usize];
            e.head = self.rank[e.head as usize];
            let (u, v) = e.undirected();
            self.find_edge_mut(u, v).weight[e.dir()] = e.weight;
        }

        // For each node from low to high, relax the edges between its upper neighbours.
        info!("Relax upper edges..");
        for u in 0..self.n {
            let edge_range = self.edge_range(u);
            trace!("relax {u} with {} nbs..", edge_range.len());

            // Relax
            for i in edge_range.clone() {
                let ux = self.edges[i];
                let x = ux.head;
                let mut idx = self.edge_range(x).start as usize;
                for j in i + 1..edge_range.end {
                    let uy = self.edges[j];
                    let y = uy.head;
                    assert!(x < y);
                    while self.edges[idx].head < y {
                        idx += 1;
                    }
                    assert_eq!(self.edges[idx].head, y);
                    let xy = &mut self.edges[idx];

                    for dir in [UP, DOWN] {
                        let xy_relax = ux.weight[dir ^ 1] + uy.weight[dir];
                        let old = xy.weight[dir];
                        xy.weight[dir] = xy.weight[dir].min(xy_relax);
                        trace!(
                            "relax {idx} {x}-{y} dir {dir} from {old} with {xy_relax} to {}",
                            xy.weight[dir]
                        );
                    }
                }
            }
        }

        if perfect {
            // Drop edges that are never in a shortest path.
            // For each node from high to low, check if the middle and top edge of
            // its triangles between upper neighbours are needed.
            // If they are not tight, update their length and mark them for deletion.
            info!("Drop redundant lower/middle edges..");
            for u in (0..self.n).rev() {
                let edge_range = self.edge_range(u);
                for i in edge_range.clone() {
                    let ux = self.edges[i];
                    let x = ux.head;
                    let mut idx = self.edge_range(x).start as usize;
                    for j in i + 1..edge_range.end {
                        let uy = self.edges[j];
                        let y = uy.head;
                        assert!(x < y);
                        while self.edges[idx].head < y {
                            idx += 1;
                        }
                        assert_eq!(self.edges[idx].head, y);
                        let xy = self.edges[idx];

                        // u < x < y
                        for dir in [UP, DOWN] {
                            let ux_relax = uy.weight[dir] + xy.weight[dir ^ 1];
                            if ux_relax <= ux.weight[dir] {
                                let ux = &mut self.edges[i];
                                ux.weight[dir] = ux_relax;
                                ux.deleted[dir] = true;
                            }
                            let uy_relax = ux.weight[dir] + xy.weight[dir];
                            if uy_relax <= uy.weight[dir] {
                                let uy = &mut self.edges[j];
                                uy.weight[dir] = uy_relax;
                                uy.deleted[dir] = true;
                            }
                        }
                    }
                }
            }

            // Remove the deleted edges.
            // For count for each node the number of non-deleted outgoing edges, and compactify the non-deleted edges.
            info!("Compress edges..");

            let mut delete_one = 0;
            let mut delete_both = 0;

            let num_edges = self.edges.len();
            // output edge index
            let mut i = 0;
            for u in 0..self.n {
                let edge_range = self.edge_range(u);
                self.nodes[u as usize].first_edge_idx = i as u32;
                for j in edge_range {
                    let e = self.edges[j];
                    if e.deleted[0] && e.deleted[1] {
                        delete_both += 1;
                    } else if e.deleted[0] || e.deleted[1] {
                        delete_one += 1;
                    }

                    // if one of the two directions is not deleted, keep the edge
                    if !e.deleted[0] || !e.deleted[1] {
                        self.edges[i] = e;
                        i += 1;
                    }
                }
            }
            self.nodes[self.n as usize].first_edge_idx = i as u32;
            self.edges.truncate(i);
            info!("Pruned edges: {:>8}", num_edges - self.edges.len());
            info!("Remaining:    {:>8}", self.edges.len());
            info!("delete one    {:>8}", delete_one);
            info!("delete both   {:>8}", delete_both);
        }

        // // Sort outgoing edges by increasing weight
        // for u in 0..self.n {
        //     let edge_range = self.edge_range(u);
        //     self.edges[edge_range].sort_unstable_by_key(|e| e.weight);
        // }

        // Add dummy INF edges if neighbours are at most GAP apart.
        info!("Add dummy edges.. Starting with {} edges", self.edges.len());
        const GAP: u32 = 4;
        let mut new_edges = vec![];
        for u in 0..self.n {
            let edge_range = self.edge_range(u);
            self.nodes[u as usize].first_edge_idx = new_edges.len() as u32;
            if edge_range.is_empty() {
                continue;
            }
            let mut last_v = self.edges[edge_range.start].head;
            let mut start = last_v;
            let mut end = start + L as u32;
            for i in edge_range {
                let e = self.edges[i];
                let v = e.head;
                if v <= end + GAP {
                    for v_new in last_v + 1..v {
                        new_edges.push(HalfEdge {
                            head: v_new,
                            weight: [W_INF; 2],
                            deleted: [false; 2],
                        });
                    }
                } else {
                    // break the range, but round to multiple of 8 first
                    for v_new in last_v + 1..end {
                        new_edges.push(HalfEdge {
                            head: v_new,
                            weight: [W_INF; 2],
                            deleted: [false; 2],
                        });
                    }
                    start = v;
                    end = start + L as u32;
                }
                new_edges.push(e);
                while v >= end {
                    end += L as u32;
                }
                last_v = v;
            }
            // round the last range to multiple of 8
            for v_new in last_v + 1..end {
                new_edges.push(HalfEdge {
                    head: v_new,
                    weight: [W_INF; 2],
                    deleted: [false; 2],
                });
            }
        }
        info!("End with {} edges", new_edges.len());
        self.nodes[self.n as usize].first_edge_idx = new_edges.len() as u32;
        self.edges = new_edges;

        // Compute edge ranges
        for u in 0..self.n {
            let edge_range = self.edge_range(u);
            // let ranges = compress(self.edges[edge_range.clone()].iter().map(|e| e.head));
            let ranges = compress(&self.edges, edge_range.clone());
            self.nodes[u as usize].first_range_idx = self.edge_ranges.len() as u32;
            self.edge_ranges.extend(ranges);
        }
        info!("Num ranges: {:>8}", self.edge_ranges.len());
        self.nodes[self.n as usize].first_range_idx = self.edge_ranges.len() as u32;
        // Fill edge weights
        for e in &self.edges {
            for dir in [UP, DOWN] {
                self.edge_weigths[dir].push(e.weight[dir]);
            }
        }
        for dir in [UP, DOWN] {
            self.edge_weigths[dir].extend(std::iter::repeat(W_INF).take(2 * L));
        }
    }

    #[inline(never)]
    fn query(&mut self, s: NodeId, t: NodeId) -> Weight {
        if self.dist[0].is_empty() {
            self.dist = [vec![W_INF; self.nodes.len()], vec![W_INF; self.nodes.len()]];
        }

        let s = self.rank[s as usize];
        let t = self.rank[t as usize];

        self.dist[UP][s as usize] = 0;
        self.dist[DOWN][t as usize] = 0;

        let mut num_visited_nodes = 0;
        let mut num_expanded_nodes = 0;
        let mut num_edges = 0;
        let mut num_pruned = 0;
        let num_pruned_edges = 0;

        // Process the smaller of s and t.
        let mut cur = [s, t];
        while cur[0] != cur[1] {
            let dir = if cur[UP] <= cur[DOWN] { UP } else { DOWN };
            num_visited_nodes += 1;
            let x = cur[dir];
            let dx = self.dist[dir][x as usize];
            self.expand_node(&mut num_expanded_nodes, &mut num_edges, dir, x, dx);

            // Go to parent.
            cur[dir] = self.nodes[x as usize].parent;
        }
        let mut x = cur[0];
        let mut best_dist = W_INF;
        while x != INVALID_ID {
            best_dist = best_dist.min(self.dist[UP][x as usize] + self.dist[DOWN][x as usize]);

            for dir in [UP, DOWN] {
                num_visited_nodes += 1;
                let dx = self.dist[dir][x as usize];
                if dx >= best_dist {
                    num_pruned += 1;
                    // cleanup for reuse
                    self.dist[dir][x as usize] = W_INF;
                    continue;
                }
                self.expand_node(&mut num_expanded_nodes, &mut num_edges, dir, x, dx);
            }
            // Go to parent.
            x = self.nodes[x as usize].parent;
        }

        debug!("dists from {s:>10}-{t:>10}: {best_dist:>10}. {num_visited_nodes:>6} visited, {num_pruned:>6} pruned, {num_expanded_nodes:>6} expanded, {num_edges:>6} relaxed, {num_pruned_edges:>6} pruned edges");
        best_dist
    }

    #[inline(always)]
    fn expand_node(
        &mut self,
        num_expanded_nodes: &mut i32,
        num_edges: &mut usize,
        dir: usize,
        x: u32,
        dx: i32,
    ) {
        // Distance to a parent can be INF in case edges were pruned.
        if dx >= W_INF {
            return;
        }
        *num_expanded_nodes += 1;
        let edge_range = self.edge_range(x);
        if edge_range.is_empty() {
            self.dist[dir][x as usize] = W_INF;
            return;
        }
        if log::log_enabled!(log::Level::Trace) {
            let outdeg = edge_range
                .clone()
                .filter(|i| self.edges[*i].weight[0] < W_INF || self.edges[*i].weight[1] < W_INF)
                .count();
            trace!("expanding {x} with degree {outdeg} ({})", edge_range.len());
        }

        *num_edges += edge_range.len();
        let ranges = &self.edge_ranges[self.nodes[x as usize].first_range_idx as usize
            ..self.nodes[x as usize + 1].first_range_idx as usize];
        let d = &mut self.dist[dir];
        let w = &self.edge_weigths[dir];

        let dx_simd = S::splat(dx);
        let mut i0 = edge_range.start as usize;
        for range in ranges {
            unsafe {
                let v0 = range.first_head;

                // A single loop body
                {
                    let old_dists = S::from_array(
                        *d.get_unchecked(v0 as usize..v0 as usize + L)
                            .as_array()
                            .unwrap(),
                    );
                    let ws = S::from_array(*w.get_unchecked(i0..i0 + L).as_array().unwrap());
                    let new_dists = ws + dx_simd;
                    let min_dists = old_dists.simd_min(new_dists);
                    *d.get_unchecked_mut(v0 as usize..v0 as usize + L)
                        .as_mut_array()
                        .unwrap() = min_dists.to_array();
                }
                i0 += 8;
            }
        }
        // cleanup for reuse
        self.dist[dir][x as usize] = W_INF;
    }
}

fn main() {
    env_logger::init();

    let path = Path::new("graphs/europe");
    let mut cch = if false {
        // write
        let mut cch = CCH::new(path);
        cch.customize(true);
        // cch.save(path);
        cch
    } else {
        // read
        CCH::read(path)
    };
    // cch.parent_stats();

    let q = 20000;

    let n = cch.n;

    let mut rng = SmallRng::from_seed([123; _]);
    let queries = (0..q)
        .map(|_| (rng.random_range(0..n), rng.random_range(0..n)))
        .collect_vec();

    let sources = read_vec(&Path::new("queries/sources"));
    for i in (0..25).rev() {
        let targets = read_vec(&Path::new(&format!("queries/targets_{i}")));
        for _ in 0..5 {
            // q random queries to warm/trash cache
            for (s, t) in &queries {
                cch.query(*s, *t);
            }

            let mut out = vec![];
            let start = std::time::Instant::now();
            for (s, t) in std::iter::zip(&sources, &targets) {
                out.push(cch.query(*s, *t));
            }
            let elapsed = start.elapsed();
            info!(
                "Queries 2^{i}.. done {} us/q",
                elapsed.as_nanos() / sources.len() as u128 / 1000
            );
        }
    }
    // if q == 10 {
    //     info!("CCH dists: {out:?}");
    //     let dists = dijkstra::dijkstra(path, &queries);
    //     info!("Dijkstra:  {:?}", dists);
    //     let target_sum = dists.iter().sum::<Weight>();
    //     info!("target_sum: {target_sum}");
    //     let sum = out.iter().sum::<Weight>();
    //     assert_eq!(sum, target_sum);
    // }
}

/// Takes a range of EdgeId and splits it into multiple ranges of EdgeId
/// where the neighbours are consecutive.
fn compress(edges: &Vec<HalfEdge>, mut range: Range<usize>) -> Vec<EdgeRange> {
    // let in_range = range.clone();
    let mut ranges = vec![];
    let Some(mut start) = range.next() else {
        return ranges;
    };
    let mut prev = edges[start].head;
    for x in range.clone() {
        if edges[x].head != prev + 1 || (x - start) == 8 {
            ranges.push(EdgeRange {
                first_head: edges[start].head,
            });
            start = x;
        }
        prev = edges[x].head;
    }
    ranges.push(EdgeRange {
        first_head: edges[start].head,
    });
    // if in_range.len() > 100 {
    //     eprintln!("compress {in_range:?} to {ranges:?}");
    // }
    ranges
}
