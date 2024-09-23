use std::collections::{VecDeque, HashMap};
use crate::types::{Address, U256, Edge};
use crate::graph_new::graph::FlowGraph;
use crate::graph_new::network_flow::{NetworkFlowAlgorithm, NetworkFlow};
use crate::graph_new::flow_recorder::FlowRecorder;
use crate::graph_new::path_search::PathSearchAlgorithm;

#[derive(Clone)]
pub struct PushRelabelAlgorithm;

impl NetworkFlowAlgorithm for PushRelabelAlgorithm {
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
        println!("Starting Push-Relabel algorithm");
        println!("Requested flow: {}", requested_flow.to_decimal());

        let mut residual_graph = graph.clone();
        let mut height: HashMap<Address, usize> = HashMap::new();
        let mut excess: HashMap<Address, U256> = HashMap::new();
        let n = residual_graph.get_nodes().len();
        let mut count: Vec<usize> = vec![0; 2 * n]; // Adjusted size for potential heights

        self.initialize_preflow(&mut residual_graph, source, sink, &mut height, &mut excess, &mut count);

        let mut active_nodes: VecDeque<Address> = residual_graph.get_nodes()
            .iter()
            .filter(|&&node| node != *source && node != *sink && excess.get(&node).unwrap_or(&U256::from(0)) > &U256::from(0))
            .cloned()
            .collect();

        let mut iteration = 0;

        while let Some(node) = active_nodes.pop_front() {
            iteration += 1;

            if excess.get(sink).unwrap_or(&U256::from(0)) >= &requested_flow {
                println!("Reached requested flow at iteration {}", iteration);
                break;
            }

            self.discharge(&mut residual_graph, &node, source, sink, &mut height, &mut excess, &mut active_nodes, &mut count);

            if let Some(ref mut recorder) = recorder {
                recorder.record_step(excess.get(sink).cloned().unwrap_or(U256::from(0)), Vec::new(), U256::from(0), residual_graph.clone());
            }

            if iteration % 1000 == 0 {
                println!("Iteration {}: Current flow to sink: {}", iteration, excess.get(sink).unwrap_or(&U256::from(0)).to_decimal());
            }

            if iteration > 1000000 {
                println!("Exceeded maximum iterations. Terminating.");
                break;
            }
        }

        let total_flow = excess.get(sink).cloned().unwrap_or(U256::from(0));
        println!("Push-Relabel algorithm completed. Final flow: {}", total_flow.to_decimal());

        let flow_paths = self.reconstruct_flow_paths(&residual_graph, graph, source, sink);

        let (final_flow, final_transfers) = NetworkFlow::post_process(total_flow, flow_paths, requested_flow, source, sink);

        println!("Post-processing completed. Final flow: {}", final_flow.to_decimal());
        println!("Number of flow paths after post-processing: {}", final_transfers.len());

        (final_flow, final_transfers)
    }
}

impl PushRelabelAlgorithm {
    fn initialize_preflow(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        height: &mut HashMap<Address, usize>,
        excess: &mut HashMap<Address, U256>,
        count: &mut Vec<usize>
    ) {
        let n = graph.get_nodes().len();
        for node in graph.get_nodes() {
            height.insert(*node, 0);
            excess.insert(*node, U256::from(0));
        }

        height.insert(*source, n);
        count[0] = n - 1;
        count[n] = 1;

        for (to, token, capacity) in graph.get_outgoing_edges(source).clone() {
            if capacity > U256::from(0) {
                graph.decrease_edge_capacity(source, &to, &token, capacity);
                graph.increase_edge_capacity(&to, source, &token, capacity);

                *excess.get_mut(&to).unwrap() += capacity;
                *excess.get_mut(source).unwrap() -= capacity;
                
                if to != *sink {
                    // Add to active nodes if not sink
                    excess.entry(to).or_insert(U256::from(0));
                }
            }
        }

        // Ensure sink's excess is initialized
        excess.entry(*sink).or_insert(U256::from(0));

        println!("Preflow initialization complete. Source excess: {}", excess[source].to_decimal());
    }

    fn discharge(
        &self,
        graph: &mut FlowGraph,
        node: &Address,
        source: &Address,
        sink: &Address,
        height: &mut HashMap<Address, usize>,
        excess: &mut HashMap<Address, U256>,
        active_nodes: &mut VecDeque<Address>,
        count: &mut Vec<usize>
    ) {
        while excess[node] > U256::from(0) {
            let node_height = height[node];

            let mut pushed = false;

            for (to, token, capacity) in graph.get_outgoing_edges(node).clone() {
                if capacity > U256::from(0) && node_height == height[&to] + 1 {
                    let flow = std::cmp::min(excess[node], capacity);

                    graph.decrease_edge_capacity(node, &to, &token, flow);
                    graph.increase_edge_capacity(&to, node, &token, flow);

                    *excess.get_mut(node).unwrap() -= flow;
                    *excess.entry(to).or_insert(U256::from(0)) += flow;

                    if to != *source && to != *sink && excess[&to] == flow {
                        active_nodes.push_back(to);
                    }

                    pushed = true;
                    println!("Pushed {} flow from {} to {}", flow.to_decimal(), node, to);

                    if excess[node] == U256::from(0) {
                        break;
                    }
                }
            }

            if !pushed {
                self.relabel(graph, node, height, count);
                if height[node] >= 2 * graph.get_nodes().len() {
                    // Node cannot push any more flow
                    break;
                }
            }
        }
    }

    fn relabel(
        &self,
        graph: &FlowGraph,
        node: &Address,
        height: &mut HashMap<Address, usize>,
        count: &mut Vec<usize>
    ) {
        let old_height = height[node];
        count[old_height] -= 1;

        let min_height = graph.get_outgoing_edges(node)
            .iter()
            .filter(|(_, _, capacity)| *capacity > U256::from(0))
            .map(|(to, _, _)| height[to])
            .min();

        let new_height = match min_height {
            Some(h) => {
                // Ensure sink remains at height 0
                if h >= usize::MAX - 1 {
                    usize::MAX
                } else {
                    h + 1
                }
            },
            None => {
                // If no neighbors with residual capacity, set height to at least n
                2 * graph.get_nodes().len()
            }
        };

        height.insert(*node, new_height);

        if new_height >= count.len() {
            count.resize(new_height + 1, 0);
        }
        count[new_height] += 1;

        println!(
            "Relabeled node {}. Old height: {}, New height: {}",
            node, old_height, new_height
        );

        // Apply gap heuristic
        if count[old_height] == 0 {
            self.gap_heuristic(old_height, height, count, graph);
        }
    }

    fn gap_heuristic(
        &self,
        gap_height: usize,
        height: &mut HashMap<Address, usize>,
        count: &mut Vec<usize>,
        graph: &FlowGraph
    ) {
        println!("Applying gap heuristic at height {}", gap_height);
        let n = 2 * graph.get_nodes().len();
        for (node, h) in height.iter_mut() {
            if *h > gap_height && *h < n {
                count[*h] -= 1;
                *h = n;
                if *h >= count.len() {
                    count.resize(*h + 1, 0);
                }
                count[*h] += 1;
                println!("Gap heuristic: Node {} height set to {}", node, *h);
            }
        }
    }

    fn reconstruct_flow_paths(
        &self,
        residual_graph: &FlowGraph,
        original_graph: &FlowGraph,
        _source: &Address,
        _sink: &Address
    ) -> Vec<Edge> {
        let mut flow_edges = Vec::new();

        for node in original_graph.get_nodes() {
            for (to, token, capacity) in original_graph.get_outgoing_edges(node) {
                let residual_capacity = residual_graph.get_edge_capacity(node, &to, &token);

                let flow = capacity - residual_capacity;

                if flow > U256::from(0) {
                    flow_edges.push(Edge {
                        from: *node,
                        to,
                        token,
                        capacity: flow,
                    });
                }
            }
        }

        println!("Reconstructed {} flow paths", flow_edges.len());
        flow_edges
    }
}
