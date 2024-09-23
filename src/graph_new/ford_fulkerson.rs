use std::collections::HashSet;
use crate::types::{Address, U256, Edge};
use crate::graph_new::graph::FlowGraph;
use crate::graph_new::network_flow::{NetworkFlowAlgorithm, NetworkFlow};
use crate::graph_new::path_search::{PathSearch, PathSearchAlgorithm};
use crate::graph_new::flow_recorder::FlowRecorder;

#[derive(Clone)]
pub struct FordFulkerson;

impl NetworkFlowAlgorithm for FordFulkerson {
    fn compute_flow(
        &self,
        graph: &FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        _max_transfers: Option<u64>,
        path_search_algorithm: PathSearchAlgorithm,
        mut recorder: Option<&mut FlowRecorder>
    ) -> (U256, Vec<Edge>) {
        println!("Starting Optimized Ford-Fulkerson algorithm");
        println!("Requested flow: {}", requested_flow.to_decimal());

        let source_neighbors: HashSet<_> = graph.get_outgoing_edges(source)
            .iter()
            .map(|(to, _, _)| *to)
            .collect();

        let sink_neighbors: HashSet<_> = graph.get_incoming_edges(sink)
            .iter()
            .map(|(from, _, _)| *from)
            .collect();

        // Early return if either source or sink has no neighbors
        if source_neighbors.is_empty() || sink_neighbors.is_empty() {
            println!("No flow possible: source or sink has no neighbors");
            return (U256::from(0), Vec::new());
        }
            
        let mut total_flow = U256::default();
        let mut sink_flow = U256::default();
        let mut flow_paths = Vec::new();
        let mut residual_graph = graph.clone();
        let estimated_max_flow = graph.estimate_max_flow(source, sink);
        println!("Estimated max flow: {}", estimated_max_flow.to_decimal());
        let target_flow = std::cmp::min(requested_flow, estimated_max_flow);
        println!("Target flow: {}", target_flow.to_decimal());
        
        loop {
            let (path_flow, path) = PathSearch::find_path(
                path_search_algorithm,
                &mut residual_graph,
                source,
                sink,
                target_flow - sink_flow,
                max_distance,
                Some(U256::from(100000000000000000))
            );
            
            if path_flow == U256::default() || path.is_empty() {
                println!("No more augmenting paths found");
                break;
            }
        
            let reached_sink = Self::update_flow(&mut residual_graph, &path, path_flow, &mut flow_paths);
            if reached_sink {
                sink_flow += path_flow;
           //     println!("Current flow to sink: {}", sink_flow.to_decimal());
            }
            
            total_flow += path_flow;
           // println!("Current total flow: {}", total_flow.to_decimal());

            if let Some(ref mut recorder) = recorder {
                recorder.record_step(total_flow, path.clone(), path_flow, residual_graph.clone());
            }
         
            if sink_flow >= target_flow {
                println!("Reached or exceeded requested flow to sink");
                break;
            }
        }

        // Calculate the actual sink flow from flow_paths
        let actual_sink_flow = flow_paths.iter()
            .filter(|e| e.to == *sink)
            .fold(U256::from(0), |acc, e| acc + e.capacity);

        println!("Ford-Fulkerson completed. Final total flow: {}", total_flow.to_decimal());
        println!("Final flow to sink: {}", actual_sink_flow.to_decimal());
        println!("Number of flow paths before post-processing: {}", flow_paths.len());

        let (final_flow, final_transfers) = NetworkFlow::post_process(actual_sink_flow, flow_paths, requested_flow, source, sink);
                
        println!("Post-processing completed. Final flow: {}", final_flow.to_decimal());
        println!("Number of flow paths after post-processing: {}", final_transfers.len());

       // NetworkFlow::print_flow_information(graph, &final_transfers);

        (final_flow, final_transfers)
    }
}

impl FordFulkerson {
    fn update_flow(graph: &mut FlowGraph, path: &[Address], path_flow: U256, flow_paths: &mut Vec<Edge>) -> bool {
        let mut reached_sink = false;
        
        for window in path.windows(2) {
            let from = window[0];
            let to = window[1];
            let token = graph.get_edge_token(&from, &to).expect("Edge not found");
            
            // Decrease capacity in the forward direction
            graph.decrease_edge_capacity(&from, &to, &token, path_flow);
            
            // Increase capacity in the reverse direction
            graph.increase_edge_capacity(&to, &from, &token, path_flow);

            flow_paths.push(Edge {
                from,
                to,
                token,
                capacity: path_flow,
            });

            if to == *path.last().unwrap() {
                reached_sink = true;
            }
        }

        reached_sink
    }

}
