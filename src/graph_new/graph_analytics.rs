use std::collections::{HashMap, HashSet};
use crate::graph_new::graph::FlowGraph;
use crate::types::{Address, U256};
use plotters::prelude::*;
use plotters::style::Color;
use plotters::element::Rectangle;
use std::error::Error;

pub struct GraphAnalytics;

impl GraphAnalytics {
    pub fn analyze(graph: &FlowGraph) -> AnalysisResult {
        let mut result = AnalysisResult::default();

        result.total_nodes = graph.get_nodes().len();
        result.total_edges = graph.get_edges().len();
        result.unique_tokens = Self::count_unique_tokens(graph);

        let (incoming, outgoing) = Self::count_incoming_outgoing_edges(graph);
        result.nodes_with_incoming_edges = incoming;
        result.nodes_with_outgoing_edges = outgoing;

        result.edge_count_distribution = Self::calculate_edge_distribution(graph);
        result.token_usage = Self::analyze_token_usage(graph);
        result.total_network_value = Self::calculate_total_network_value(graph);

        result
    }

    fn count_unique_tokens(graph: &FlowGraph) -> usize {
        graph.get_edges().keys().map(|key| key.token).collect::<HashSet<_>>().len()
    }

    fn count_incoming_outgoing_edges(graph: &FlowGraph) -> (usize, usize) {
        let mut incoming = HashSet::new();
        let mut outgoing = HashSet::new();

        for edge_key in graph.get_edges().keys() {
            outgoing.insert(edge_key.from);
            incoming.insert(edge_key.to);
        }

        (incoming.len(), outgoing.len())
    }

    fn calculate_edge_distribution(graph: &FlowGraph) -> HashMap<usize, usize> {
        let mut distribution = HashMap::new();
        let mut edge_counts = HashMap::new();

        for edge_key in graph.get_edges().keys() {
            *edge_counts.entry(edge_key.from).or_insert(0) += 1;
            *edge_counts.entry(edge_key.to).or_insert(0) += 1;
        }

        for &count in edge_counts.values() {
            *distribution.entry(count).or_insert(0) += 1;
        }

        distribution
    }

    fn analyze_token_usage(graph: &FlowGraph) -> HashMap<Address, TokenUsage> {
        let mut token_usage = HashMap::new();
    
        for (edge_key, &capacity) in graph.get_edges() {
            let usage = token_usage.entry(edge_key.token).or_insert_with(TokenUsage::new);
            usage.add_capacity(capacity);
            usage.unique_users.insert(edge_key.from);
            usage.unique_users.insert(edge_key.to);
        }
    
        token_usage
    }

    fn calculate_total_network_value(graph: &FlowGraph) -> U256 {
        graph.get_edges().values().fold(U256::from(0), |acc, &value| {
            let sum = acc + value;
            if sum < acc || sum < value {
                println!("Warning: Total network value overflow. Returning maximum U256 value.");
                U256::MAX
            } else {
                sum
            }
        })
    }

    pub fn plot_edge_distribution(analysis_result: &AnalysisResult, output_file: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_y = *analysis_result.edge_count_distribution.values().max().unwrap_or(&0) as f64;
        let max_x = *analysis_result.edge_count_distribution.keys().max().unwrap_or(&0) as f64;

        let mut chart = ChartBuilder::on(&root)
            .caption("Edge Count Distribution", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f64..(max_x + 1.0), 0f64..(max_y * 1.1))?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(
                analysis_result.edge_count_distribution
                    .iter()
                    .map(|(&edge_count, &node_count)| {
                        let x0 = edge_count as f64;
                        let x1 = x0 + 0.8;
                        let y0 = 0.0;
                        let y1 = node_count as f64;
                        Rectangle::new([(x0, y0), (x1, y1)], BLACK.filled())
                    })
            )?
            .label("Node Count")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

        chart
            .configure_series_labels()
            .background_style(WHITE.filled())
            .border_style(BLACK)
            .draw()?;

        root.present()?;

        Ok(())
    }
}

#[derive(Default)]
pub struct AnalysisResult {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub unique_tokens: usize,
    pub nodes_with_incoming_edges: usize,
    pub nodes_with_outgoing_edges: usize,
    pub edge_count_distribution: HashMap<usize, usize>,
    pub token_usage: HashMap<Address, TokenUsage>,
    pub total_network_value: U256,
}

#[derive(Default)]
pub struct TokenUsage {
    pub total_capacity: U256,
    pub involved_in_edges: usize,
    pub unique_users: HashSet<Address>,
}

impl TokenUsage {
    pub fn new() -> Self {
        TokenUsage {
            total_capacity: U256::from(0),
            involved_in_edges: 0,
            unique_users: HashSet::new(),
        }
    }

    pub fn add_capacity(&mut self, capacity: U256) {
        let sum = self.total_capacity + capacity;
        if sum < self.total_capacity || sum < capacity {
            println!("Warning: Token capacity overflow. Setting to maximum U256 value.");
            self.total_capacity = U256::MAX;
        } else {
            self.total_capacity = sum;
        }
        self.involved_in_edges += 1;
    }
}

impl AnalysisResult {
    pub fn print_summary(&self) {
        println!("Graph Analysis Summary");
        println!("======================");
        println!("Total Nodes: {}", self.total_nodes);
        println!("Total Edges: {}", self.total_edges);
        println!("Unique Tokens: {}", self.unique_tokens);
        println!("Nodes with Incoming Edges: {}", self.nodes_with_incoming_edges);
        println!("Nodes with Outgoing Edges: {}", self.nodes_with_outgoing_edges);
        println!("Total Network Value: {}", self.total_network_value);
        
       // println!("\nEdge Count Distribution:");
       // let mut distribution: Vec<_> = self.edge_count_distribution.iter().collect();
       // distribution.sort_by_key(|&(k, _)| k);
       // for (edge_count, node_count) in distribution {
       //     println!("  {} edges: {} nodes", edge_count, node_count);
       // }

        println!("\nTop 10 Tokens by Capacity:");
        let mut tokens: Vec<_> = self.token_usage.iter().collect();
        tokens.sort_by(|a, b| b.1.total_capacity.cmp(&a.1.total_capacity));
        for (token, usage) in tokens.iter().take(10) {
            println!("  Token {}: Capacity {}, Edges {}, Users {}", 
                     token, usage.total_capacity, usage.involved_in_edges, usage.unique_users.len());
        }
    }

    pub fn plot_edge_distribution(&self, output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
        GraphAnalytics::plot_edge_distribution(self, output_file)
    }
}