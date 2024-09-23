use crate::types::{Address, U256, Edge};
use super::graph::FlowGraph;
use std::collections::{HashMap, VecDeque, HashSet};
use crate::graph_new::path_search::PathSearchAlgorithm;
use crate::graph_new::flow_recorder::FlowRecorder;

pub trait NetworkFlowAlgorithm: Clone {
    fn compute_flow(
        &self,
        graph: &FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        max_transfers: Option<u64>,
        path_search_algorithm: PathSearchAlgorithm,
        recorder: Option<&mut FlowRecorder>
    ) -> (U256, Vec<Edge>);
}

pub struct NetworkFlow {
    pub graph: FlowGraph,
}

impl NetworkFlow {
    pub fn new(graph: FlowGraph) -> Self {
        NetworkFlow { graph }
    }

    pub fn compute_flow<A: NetworkFlowAlgorithm>(
        &self,
        algorithm: A,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        max_transfers: Option<u64>,
        path_search_algorithm: PathSearchAlgorithm,
        recorder: Option<&mut FlowRecorder>
    ) -> (U256, Vec<Edge>) {
        algorithm.compute_flow(&self.graph, source, sink, requested_flow, max_distance, max_transfers, path_search_algorithm, recorder)
    }

    pub fn prune_excess_flow(transfers: Vec<Edge>, flow: U256, requested_flow: U256, _source: &Address, sink: &Address) -> Vec<Edge> {
        println!("Entering prune_excess_flow");
        println!("Current flow: {}, Requested flow: {}", flow.to_decimal(), requested_flow.to_decimal());
    
        if flow <= requested_flow {
            println!("No pruning needed, returning original transfers");
            return transfers;
        }
    
        let mut pruned = Vec::new();
        let mut excess = flow - requested_flow;
        println!("Excess flow to prune: {}", excess.to_decimal());
    
        let mut transfers_iter = transfers.into_iter();
    
        // First, preserve all edges to the sink
        let sink_edges: Vec<_> = transfers_iter.by_ref().filter(|e| e.to == *sink).collect();
        let sink_flow = sink_edges.iter().fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Total flow to sink: {}", sink_flow.to_decimal());
        pruned.extend(sink_edges.iter().cloned());
    
        // Then process the remaining edges
        for edge in transfers_iter {
            if excess > U256::from(0) && edge.capacity <= excess {
                println!("Pruning edge: {:?}", edge);
                excess -= edge.capacity;
            } else {
                let capacity = if excess > U256::from(0) {
                    let new_capacity = edge.capacity - excess;
                    println!("Partially pruning edge: {:?}", edge);
                    println!("New capacity: {}", new_capacity.to_decimal());
                    excess = U256::from(0);
                    new_capacity
                } else {
                    edge.capacity
                };
                pruned.push(Edge { capacity, ..edge });
            }
        }
    
        let pruned_flow = pruned.iter().filter(|e| e.to == *sink).fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Flow after pruning: {}", pruned_flow.to_decimal());
    
        pruned
    }

    pub fn post_process(sink_flow: U256, transfers: Vec<Edge>, requested_flow: U256, source: &Address, sink: &Address) -> (U256, Vec<Edge>) {
        println!("Initial flow: {}", sink_flow.to_decimal());
        println!("Initial transfers count: {}", transfers.len());
    
        let initial_sink_flow = transfers.iter()
            .filter(|e| e.to == *sink)
            .fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Initial flow to sink: {}", initial_sink_flow.to_decimal());
    
        let total_transfer_flow = transfers.iter()
            .fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Total flow of all transfers: {}", total_transfer_flow.to_decimal());
    
        // First pruning step
        let pruned_transfers = if sink_flow > requested_flow {
            Self::prune_excess_flow(transfers, sink_flow, requested_flow, source, sink)
        } else {
            println!("Skipping initial pruning as sink_flow <= requested_flow");
            transfers
        };
        let pruned_flow = pruned_transfers.iter()
            .filter(|e| e.to == *sink)
            .fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Flow after initial pruning: {}", pruned_flow.to_decimal());
        println!("Transfers count after initial pruning: {}", pruned_transfers.len());
    
        // Simplification step
        let simplified_transfers = Self::simplify_transfers(pruned_transfers, source, sink);
        let simplified_flow = simplified_transfers.iter()
            .filter(|e| e.to == *sink)
            .fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Flow after simplification: {}", simplified_flow.to_decimal());
        println!("Transfers count after simplification: {}", simplified_transfers.len());
    
        // Second pruning step (if necessary)
        let final_transfers = if simplified_flow > requested_flow {
            println!("Performing second pruning as simplified_flow > requested_flow");
            Self::prune_excess_flow(simplified_transfers, simplified_flow, requested_flow, source, sink)
        } else {
            simplified_transfers
        };
    
        // Sorting step
        let sorted_transfers = Self::sort_transfers(final_transfers, source, sink);
        let final_flow = sorted_transfers.iter()
            .filter(|e| e.to == *sink)
            .fold(U256::from(0), |acc, e| acc + e.capacity);
    
        println!("Final flow to sink: {}", final_flow.to_decimal());
        println!("Number of edges to sink: {}", sorted_transfers.iter().filter(|e| e.to == *sink).count());
    
        // Sanity check
        if final_flow > requested_flow {
            println!("WARNING: Final flow exceeds requested flow. Requested: {}, Actual: {}", 
                     requested_flow.to_decimal(), final_flow.to_decimal());
        }
    
        (final_flow, sorted_transfers)
    }

    pub fn simplify_transfers(transfers: Vec<Edge>, source: &Address, sink: &Address) -> Vec<Edge> {
        println!("Entering simplify_transfers");
        println!("Initial transfers count: {}", transfers.len());
        let initial_flow = transfers.iter().filter(|e| e.to == *sink).fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Initial flow to sink: {}", initial_flow.to_decimal());
    
        // Step 1: Combine transfers with the same from, to, and token
        let mut combined_transfers: HashMap<(Address, Address, Address), U256> = HashMap::new();
        for transfer in transfers.iter() {
            let key = (transfer.from, transfer.to, transfer.token);
            *combined_transfers.entry(key).or_default() += transfer.capacity;
        }
    
        // Build the simplified transfers
        let simplified: Vec<Edge> = combined_transfers.iter().map(|((from, to, token), &capacity)| 
            Edge { from: *from, to: *to, token: *token, capacity }
        ).collect();
    
        // Build the graph
        let mut graph: HashMap<Address, Vec<&Edge>> = HashMap::new();
        for edge in &simplified {
            graph.entry(edge.from).or_default().push(edge);
        }
    
        // Perform BFS from source to find reachable nodes
        let mut reachable_from_source = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(*source);
        reachable_from_source.insert(*source);
    
        while let Some(node) = queue.pop_front() {
            if let Some(edges) = graph.get(&node) {
                for edge in edges {
                    if reachable_from_source.insert(edge.to) {
                        queue.push_back(edge.to);
                    }
                }
            }
        }
    
        // Build reverse graph
        let mut reverse_graph: HashMap<Address, Vec<&Edge>> = HashMap::new();
        for edge in &simplified {
            reverse_graph.entry(edge.to).or_default().push(edge);
        }
    
        // Perform reverse BFS from sink to find nodes that can reach sink
        let mut can_reach_sink = HashSet::new();
        let mut reverse_queue = VecDeque::new();
        reverse_queue.push_back(*sink);
        can_reach_sink.insert(*sink);
    
        while let Some(node) = reverse_queue.pop_front() {
            if let Some(edges) = reverse_graph.get(&node) {
                for edge in edges {
                    if can_reach_sink.insert(edge.from) {
                        reverse_queue.push_back(edge.from);
                    }
                }
            }
        }
    
        // Keep only edges where both 'from' and 'to' are in the intersection of reachable nodes
        let nodes_in_path = reachable_from_source.intersection(&can_reach_sink).cloned().collect::<HashSet<_>>();
    
        let final_transfers: Vec<Edge> = simplified.into_iter()
            .filter(|edge| nodes_in_path.contains(&edge.from) && nodes_in_path.contains(&edge.to))
            .collect();
    
        let final_flow = final_transfers.iter().filter(|e| e.to == *sink).fold(U256::from(0), |acc, e| acc + e.capacity);
        println!("Final total flow after simplification: {}", final_flow.to_decimal());
        println!("Final transfers count: {}", final_transfers.len());
    
        final_transfers
    }

    pub fn sort_transfers(transfers: Vec<Edge>, source: &Address, sink: &Address) -> Vec<Edge> {
        let mut graph: HashMap<Address, Vec<Edge>> = HashMap::new();
        let mut sink_edges = Vec::new();
        let mut source_edges = Vec::new();
        
        for edge in transfers {
            if edge.to == *sink {
                sink_edges.push(edge);
            } else if edge.from == *source {
                source_edges.push(edge);
            } else {
                graph.entry(edge.from).or_default().push(edge);
            }
        }
    
        let mut sorted = Vec::new();
        let mut stack = source_edges.iter().map(|e| e.to).collect::<Vec<_>>();
        let mut visited = HashSet::new();
    
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
    
            if let Some(edges) = graph.get(&current) {
                for edge in edges {
                    sorted.push(edge.clone());
                    if !visited.contains(&edge.to) {
                        stack.push(edge.to);
                    }
                }
            }
        }
    
        // Add source edges at the beginning
        sorted.splice(0..0, source_edges);
        // Add sink edges at the end
        sorted.extend(sink_edges);
    
        sorted
    }

    pub fn print_sink_edges(edges: &[Edge], sink: &Address) {
        println!("Edges to sink:");
        for edge in edges.iter().filter(|e| e.to == *sink) {
            println!("  From: {}, Token: {}, Capacity: {}", edge.from, edge.token, edge.capacity);
        }
    }

    pub fn print_flow_information(graph: &FlowGraph, transfers: &[Edge]) {
        println!("\nFlow information for each (from, to, token) combination:");
        let mut total_flows: HashMap<(Address, Address), U256> = HashMap::new();
        
        for edge in transfers {
            let trust_limit = graph.get_trust_limit(&edge.from, &edge.to);
            let _edge_capacity = graph.get_edge_capacity(&edge.from, &edge.to, &edge.token);
            let balance = graph.get_balance(&edge.from, &edge.token);
            
            // Calculate total flow for this (from, to) pair
            let total_flow = total_flows.entry((edge.from, edge.to)).or_insert(U256::from(0));
            *total_flow += edge.capacity;
            
            println!("From: {}, To: {}, Token: {}", edge.from, edge.to, edge.token);
            println!("  Flow for this token: {}", edge.capacity.to_decimal());
            println!("  Total Flow (all tokens): {}", total_flow.to_decimal());
            println!("  Trust Limit: {}", trust_limit.to_decimal());
            println!("  Edge Capacity (in this transfer): {}", edge.capacity.to_decimal());
            println!("  Remaining Balance: {}", balance.to_decimal());
            println!();
        }
    }
}