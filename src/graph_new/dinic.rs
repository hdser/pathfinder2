use std::collections::{VecDeque, HashMap};
use crate::types::{Address, U256, Edge};
use crate::graph_new::graph::FlowGraph;
use crate::graph_new::network_flow::{NetworkFlowAlgorithm, NetworkFlow};
use crate::graph_new::path_search::PathSearchAlgorithm;
use crate::graph_new::flow_recorder::FlowRecorder;

#[derive(Clone)]
pub struct DinicsAlgorithm;

impl NetworkFlowAlgorithm for DinicsAlgorithm {
    fn compute_flow(
        &self,
        graph: &FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        _max_distance: Option<u64>,
        _max_transfers: Option<u64>,
        _path_search_algorithm: PathSearchAlgorithm,
        mut recorder: Option<&mut FlowRecorder>
    ) -> (U256, Vec<Edge>) {
        println!("Starting Dinic's algorithm");
        println!("Requested flow: {}", requested_flow.to_decimal());

        let mut residual_graph = graph.clone();
        let mut total_flow = U256::from(0);
        let mut flow_paths = Vec::new();

        // Set a minimum flow increment, similar to Ford-Fulkerson
        let min_flow_increment = U256::from(10000000000); // 0.1 tokens

        let estimated_max_flow = graph.estimate_max_flow(source, sink);
        println!("Estimated max flow: {}", estimated_max_flow.to_decimal());
        let target_flow = std::cmp::min(requested_flow, estimated_max_flow);
        println!("Target flow: {}", target_flow.to_decimal());

        loop {
            let level_graph = self.build_level_graph(&residual_graph, source, sink);
            if level_graph.is_none() {
                break;
            }

            loop {
                let remaining_flow = target_flow - total_flow;
                let max_flow = std::cmp::max(remaining_flow, min_flow_increment);

                let (blocking_flow, path) = self.find_blocking_flow(&residual_graph, &level_graph.as_ref().unwrap(), source, sink, max_flow);
                if blocking_flow == U256::from(0) || blocking_flow < min_flow_increment {
                    break;
                }

                total_flow += blocking_flow;
                self.update_residual_graph(&mut residual_graph, &path, blocking_flow, &mut flow_paths);
                println!("Current total flow: {}", total_flow.to_decimal());
                if let Some(ref mut recorder) = recorder {
                    recorder.record_step(total_flow, path.clone(), blocking_flow, residual_graph.clone());
                }

                if total_flow >= target_flow {
                    println!("Reached or exceeded requested flow");
                    break;
                }
            }

            if total_flow >= target_flow {
                break;
            }
        }

        println!("Dinic's algorithm completed. Final flow: {}", total_flow.to_decimal());
        println!("Number of flow paths before post-processing: {}", flow_paths.len());

        let (final_flow, final_transfers) = NetworkFlow::post_process(total_flow, flow_paths, requested_flow, source, sink);

        println!("Post-processing completed. Final flow: {}", final_flow.to_decimal());
        println!("Number of flow paths after post-processing: {}", final_transfers.len());

        (final_flow, final_transfers)
    }
}

impl DinicsAlgorithm {
    fn build_level_graph(&self, graph: &FlowGraph, source: &Address, sink: &Address) -> Option<HashMap<Address, u64>> {
        let mut level = HashMap::new();
        let mut queue = VecDeque::new();

        level.insert(*source, 0);
        queue.push_back(*source);

        while let Some(node) = queue.pop_front() {
            if node == *sink {
                return Some(level);
            }

            let current_level = *level.get(&node).unwrap();

            for (next, _, capacity) in graph.get_outgoing_edges(&node) {
                if capacity > U256::from(0) && !level.contains_key(&next) {
                    level.insert(next, current_level + 1);
                    queue.push_back(next);
                }
            }
        }

        None
    }

    fn find_blocking_flow(
        &self,
        graph: &FlowGraph,
        level_graph: &HashMap<Address, u64>,
        node: &Address,
        sink: &Address,
        max_flow: U256
    ) -> (U256, Vec<Address>) {
        if node == sink {
            return (max_flow, vec![*sink]);
        }

        let mut path = Vec::new();
        let current_level = *level_graph.get(node).unwrap();

        for (next, _token, capacity) in graph.get_outgoing_edges(node) {
            if capacity > U256::from(0) && level_graph.get(&next) == Some(&(current_level + 1)) {
                let bottleneck = std::cmp::min(max_flow, capacity);
                let (new_flow, mut subpath) = self.find_blocking_flow(graph, level_graph, &next, sink, bottleneck);

                if new_flow > U256::from(0) {
                    path.push(*node);
                    path.append(&mut subpath);
                    return (new_flow, path);
                }
            }
        }

        (U256::from(0), path)
    }

    fn update_residual_graph(&self, graph: &mut FlowGraph, path: &[Address], flow: U256, flow_paths: &mut Vec<Edge>) {
        for window in path.windows(2) {
            let from = window[0];
            let to = window[1];
            let token = graph.get_edge_token(&from, &to).expect("Edge not found");

            graph.decrease_edge_capacity(&from, &to, &token, flow);
            graph.increase_edge_capacity(&to, &from, &token, flow);

            flow_paths.push(Edge {
                from,
                to,
                token,
                capacity: flow,
            });
        }
    }
}
