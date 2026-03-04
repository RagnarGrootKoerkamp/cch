use crate::{read_vec, Weight, W_INF};
use log::{debug, info, trace};
use std::{cmp::Reverse, path::Path};

/// Return the sum of shortest path lengths.
#[allow(unused)]
pub fn dijkstra(path: &Path, queries: &[(u32, u32)]) -> Vec<Weight> {
    // Read files.
    info!("Reading..");
    let first_out = read_vec(&path.with_added_extension("first_out"));
    let head = read_vec(&path.with_added_extension("head"));
    // Also read the weight already.
    let weight = read_vec(&path.with_added_extension("dist"));

    let n = first_out.len() - 1;

    let mut out = vec![];

    let mut dist = vec![W_INF; n];
    let mut q = std::collections::BinaryHeap::new();
    for (i, &(s, t)) in queries.iter().enumerate() {
        debug!("query {i}..");
        dist.fill(W_INF);
        dist[s as usize] = 0;
        q.push(Reverse((0, s)));
        let mut j = 0usize;
        while let Some(Reverse((d, u))) = q.pop() {
            j += 1;
            if j.count_ones() == 1 {
                trace!("query {i}: {s} -> {t}: step {j:>9}: {u:>9} ({d})");
            }
            if u == t {
                break;
            }
            if dist[u as usize] < d {
                continue;
            }
            for j in first_out[u as usize]..first_out[u as usize + 1] {
                let v = head[j as usize];
                let w = weight[j as usize] as Weight;
                if dist[v as usize] > dist[u as usize] + w {
                    dist[v as usize] = dist[u as usize] + w;
                    q.push(Reverse((dist[v as usize], v)));
                }
            }
        }

        debug!("query {i}: {s} -> {t}: {}", dist[t as usize]);
        out.push(dist[t as usize] as Weight);
    }
    info!("Target sum {}", out.iter().sum::<Weight>());
    out
}
