use crate::types::Edge;
use crate::types::edge::EdgeDB;
use crate::types::Address;
use petgraph::graph::DiGraph;
use petgraph::dot::{Dot, Config};
use std::collections::HashMap;

pub struct GraphVisualizer {
    edges: EdgeDB,
}

impl GraphVisualizer {
    pub fn new(edges: EdgeDB) -> Self {
        GraphVisualizer { edges }
    }

    pub fn generate_capacity_graph(&self) -> String {
        let mut graph = DiGraph::new();
        let mut node_indices: HashMap<Address, _> = HashMap::new();

        // Add nodes
        for edge in self.edges.edges() {
            for address in [&edge.from, &edge.to] {
                if !node_indices.contains_key(address) {
                    let index = graph.add_node(shorten_address(address));
                    node_indices.insert(*address, index);
                }
            }
        }

        // Add edges
        for edge in self.edges.edges() {
            if let (Some(&from_idx), Some(&to_idx)) = (node_indices.get(&edge.from), node_indices.get(&edge.to)) {
                graph.add_edge(from_idx, to_idx, edge.capacity.to_string());
            }
        }

        format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]))
    }

    pub fn generate_flow_graph(&self, flow: &[Edge]) -> String {
        let mut graph = DiGraph::<String, ()>::new();
        let mut node_indices = HashMap::new();

        // Add nodes
        for edge in flow {
            for address in [&edge.from, &edge.to] {
                if !node_indices.contains_key(address) {
                    let index = graph.add_node(shorten_address(address));
                    node_indices.insert(*address, index);
                }
            }
        }

        // Generate DOT string with custom formatting
        let dot_string = format!("digraph {{\n    node [shape=circle, style=filled, fillcolor=lightblue];\n    {}\n    {}\n}}",
            // Node definitions with labels on top
            node_indices.iter().map(|(addr, &idx)| {
                format!("    {} [label=\"\", xlabel=\"{}\", xlp=\"0.5,1.2\"]", idx.index(), shorten_address(addr))
            }).collect::<Vec<_>>().join("\n"),
            // Edge definitions with flow values and token addresses
            flow.iter().map(|edge| {
                let from_idx = node_indices[&edge.from].index();
                let to_idx = node_indices[&edge.to].index();
                let capacity_decimal = edge.capacity.to_decimal();
                let token_short = shorten_address(&edge.token);
                format!("    {} -> {} [label=\"{} ({})\"]", from_idx, to_idx, capacity_decimal, token_short)
            }).collect::<Vec<_>>().join("\n")
        );

        dot_string
    }
}

fn shorten_address(address: &Address) -> String {
    let addr_str = address.to_string();
    format!("{}...{}", &addr_str[0..6], &addr_str[addr_str.len()-4..])
}