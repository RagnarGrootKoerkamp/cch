#![feature(iter_array_chunks)]
use log::{debug, info};
use rand::RngExt;
use std::path::Path;

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

#[derive(Clone)]
struct Node {
    // rank is implicit via ID (=index) of self.
    // rank: u32,
    /// Index into list of edges of the first outgoing edge.
    first_out: EdgeId,
    /// Parent ID.
    parent: NodeId,
}

type DIR = usize;
const UP: usize = 0;
const DOWN: usize = 1;

/// Edge goes to `head`.
/// Tail is implicit via `first_out` of the source node.
#[derive(Clone, Copy)]
struct HalfEdge {
    /// Edge goes to `head`.
    head: NodeId,
    /// Weight for up and down direction.
    weight: [Weight; 2],
    /// For use during perfect customization.
    deleted: bool,
}

/// Edge from `tail` to `head`.
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

pub struct CCH {
    /// The number of nodes
    n: u32,

    /// The input edges and weights after permuting nodes by `order`.
    input_edges: Vec<Edge>,

    /// The sorted nodes and their parents.
    nodes: Vec<Node>,
    /// The undirected CCH edges.
    edges: Vec<HalfEdge>,
}

impl CCH {
    /// path: "graph/europe"
    pub fn new(path: &Path) -> Self {
        let n;
        let mut input_edges = vec![];
        let mut adj;
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
            let rank = {
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
                    // permute
                    let tail = rank[i as usize];
                    let head = rank[head as usize];
                    input_edges.push(Edge { tail, head, weight });

                    let (u, v) = (tail.min(head), tail.max(head));
                    adj[u as usize].push(v);
                }
            }
        };

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
                    first_out: edges.len() as u32,
                    parent: INVALID_ID,
                });
                break;
            }
            let parent = nbs[0];
            assert!(parent > u);

            nodes.push(Node {
                first_out: edges.len() as u32,
                parent,
            });

            // store up-edges of u
            edges.extend(nbs.iter().map(|&head| HalfEdge {
                head,
                weight: [W_INF; 2],
                deleted: false,
            }));

            // copy neighbourhood to parent
            adj[parent as usize].extend(&nbs[1..]);
        }

        nodes.push(Node {
            first_out: edges.len() as u32,
            parent: INVALID_ID,
        });

        for i in 0..n as u32 {
            assert!(nodes[i as usize].parent > i);
        }

        info!("n: {n}");
        info!("nodes len: {}", nodes.len());

        Self {
            n: n as u32,
            input_edges,
            nodes,
            edges,
        }
    }

    fn edge_range(&self, u: NodeId) -> std::ops::Range<usize> {
        let i = self.nodes[u as usize].first_out as usize;
        let j = self.nodes[u as usize + 1].first_out as usize;
        i..j
    }

    fn find_edge_index(&self, u: NodeId, v: NodeId) -> usize {
        assert!(u < v);
        let edge_range = self.edge_range(u);
        edge_range.start
            + self.edges[edge_range]
                .binary_search_by_key(&v, |e| e.head)
                .unwrap()
    }

    fn find_edge(&self, u: NodeId, v: NodeId) -> &HalfEdge {
        let idx = self.find_edge_index(u, v);
        &self.edges[idx]
    }

    fn find_edge_mut(&mut self, u: NodeId, v: NodeId) -> &mut HalfEdge {
        let idx = self.find_edge_index(u, v);
        &mut self.edges[idx]
    }

    /// Use the already-permuted weights to customize all edges.
    fn customize(&mut self, perfect: bool) {
        // Copy the input weights into the CCH edges.
        info!("Set weights..");
        for e in std::mem::take(&mut self.input_edges) {
            let dir = e.dir();
            let (u, v) = e.undirected();
            // Linear scan for the corresponding edge.
            self.find_edge_mut(u, v).weight[dir] = e.weight;
        }

        // For each node from low to high, relax the edges between its upper neighbours.
        info!("Relax upper edges..");
        for u in 0..self.n {
            let edge_range = self.edge_range(u);
            for i in edge_range.clone() {
                for j in i + 1..edge_range.end {
                    let ux = self.edges[i];
                    let uy = self.edges[j];
                    let x = ux.head;
                    let y = uy.head;
                    assert!(x < y);
                    let xy = self.find_edge_mut(x, y);

                    for dir in [UP, DOWN] {
                        let xy_relax = ux.weight[dir ^ 1] + uy.weight[dir];
                        xy.weight[dir] = xy.weight[dir].min(xy_relax);
                    }
                }
            }
        }

        if !perfect {
            return;
        }

        // Drop edges that are never in a shortest path.
        // For each node from high to low, check if the middle and top edge of
        // its triangles between upper neighbours are needed.
        // If they are not tight, update their length and mark them for deletion.
        info!("Drop redundant lower/middle edges..");
        for u in (0..self.n).rev() {
            let edge_range = self.edge_range(u);
            for i in edge_range.clone() {
                for j in i + 1..edge_range.end {
                    let ux = self.edges[i];
                    let uy = self.edges[j];
                    let x = ux.head;
                    let y = uy.head;
                    assert!(x < y);
                    let xy = *self.find_edge(x, y);

                    // i < x < y
                    for dir in [UP, DOWN] {
                        let ux_relax = uy.weight[dir] + xy.weight[dir ^ 1];
                        if ux_relax < ux.weight[dir] {
                            let ux = &mut self.edges[i];
                            ux.weight[dir] = ux_relax;
                            ux.deleted = true;
                        }
                        let uy_relax = ux.weight[dir] + xy.weight[dir];
                        if uy_relax < uy.weight[dir] {
                            let uy = &mut self.edges[j];
                            uy.weight[dir] = uy_relax;
                            uy.deleted = true;
                        }
                    }
                }
            }
        }

        // Remove the deleted edges.
        // For count for each node the number of non-deleted outgoing edges, and compactify the non-deleted edges.
        info!("Compress edges..");

        // output edge index
        let mut i = 0;
        for u in 0..self.n {
            let edge_range = self.edge_range(u);
            self.nodes[u as usize].first_out = i as u32;
            for j in edge_range {
                if !self.edges[j].deleted {
                    self.edges[i] = self.edges[j];
                    i += 1;
                }
            }
        }
        self.nodes[self.n as usize].first_out = i as u32;
    }

    fn dists(&self, s: NodeId, dir: DIR) -> Vec<Weight> {
        let mut dists = vec![W_INF; self.nodes.len()];
        dists[s as usize] = 0;

        let mut num_visited_nodes = 0;
        let mut num_expanded_nodes = 0;
        let mut num_edges = 0;

        let mut u = s;
        while u != INVALID_ID {
            num_visited_nodes += 1;
            let du = dists[u as usize];
            // Distance to a parent can be INF in case edges were pruned.
            if du < W_INF {
                // eprintln!("u {u} du {du}");
                num_expanded_nodes += 1;
                let edge_range = self.edge_range(u);
                for e in &self.edges[edge_range] {
                    num_edges += 1;
                    let v = e.head;
                    let dv = dists[v as usize];
                    let new_dist = du + e.weight[dir];
                    if new_dist < dv {
                        dists[v as usize] = new_dist;
                    }
                }
                // cleanup for reuse
                // dists[u as usize] = W_INF;
            }

            // Go to parent.
            u = self.nodes[u as usize].parent;
        }
        debug!("dists from {s} in dir {dir}: {num_visited_nodes} visited, {num_expanded_nodes} expanded, {num_edges} edges relaxed");

        dists
    }
    pub fn query_full(&self, s: NodeId, t: NodeId) -> Weight {
        debug!("query {s} {t}");
        let ds = self.dists(s, UP);
        let dt = self.dists(t, DOWN);
        let dist = (0..self.nodes.len()).map(|i| ds[i] + dt[i]).min().unwrap();
        debug!("dist {s}-{t}: {dist}");
        dist
    }
}

fn main() {
    env_logger::init();

    let path = Path::new("graphs/europe");
    let mut cch = CCH::new(path);
    cch.customize(true);

    // Generate 1000 random query pairs.
    let q = 100;

    let n = cch.n;
    let mut rng = rand::rng();
    let queries = (0..q).map(|_| (rng.random_range(0..n), rng.random_range(0..n)));

    info!("Queries..");
    for (s, t) in queries {
        cch.query_full(s, t);
    }
}
