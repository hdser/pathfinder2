use std::collections::{HashMap, HashSet};
use crate::types::{Address, U256};
use std::cell::RefCell;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct EdgeKey {
    pub from: Address,
    pub to: Address,
    pub token: Address,
}

#[derive(Clone)]
pub struct FlowGraph {
    nodes: HashSet<Address>,
    edges: HashMap<EdgeKey, U256>,
    trust_limits: HashMap<(Address, Address), U256>,
    balances: HashMap<(Address, Address), U256>,
    flows: HashMap<EdgeKey, U256>,
    total_flows: HashMap<(Address, Address), U256>,
    outgoing_edges: HashMap<Address, Vec<EdgeKey>>,
    incoming_edges: HashMap<Address, Vec<EdgeKey>>,
    outgoing_edges_cache: RefCell<HashMap<Address, Vec<(Address, Address, U256)>>>,
    incoming_edges_cache: RefCell<HashMap<Address, Vec<(Address, Address, U256)>>>,
    dirty_nodes: RefCell<HashSet<Address>>,
    capacity_adjustments: HashMap<EdgeKey, U256>,
}

impl FlowGraph {
    pub fn new() -> Self {
        FlowGraph {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            trust_limits: HashMap::new(),
            balances: HashMap::new(),
            flows: HashMap::new(),
            total_flows: HashMap::new(),
            outgoing_edges: HashMap::new(),
            incoming_edges: HashMap::new(),
            outgoing_edges_cache: RefCell::new(HashMap::new()),
            incoming_edges_cache: RefCell::new(HashMap::new()),
            dirty_nodes: RefCell::new(HashSet::new()),
            capacity_adjustments: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: Address, to: Address, token: Address, capacity: U256) {
        let key = EdgeKey { from, to, token };
        self.edges.insert(key.clone(), capacity);
        self.nodes.insert(from);
        self.nodes.insert(to);

        self.outgoing_edges.entry(from).or_default().push(key.clone());
        self.incoming_edges.entry(to).or_default().push(key);

        self.invalidate_cache(&from);
        self.invalidate_cache(&to);
    }

    pub fn remove_edge(&mut self, key: &EdgeKey) {
        if self.edges.remove(key).is_some() {
            if let Some(edges) = self.outgoing_edges.get_mut(&key.from) {
                edges.retain(|e| e != key);
            }
            if let Some(edges) = self.incoming_edges.get_mut(&key.to) {
                edges.retain(|e| e != key);
            }
        }

        self.invalidate_cache(&key.from);
        self.invalidate_cache(&key.to);
    }

    pub fn get_edge_token(&self, from: &Address, to: &Address) -> Option<Address> {
        for key in self.edges.keys() {
            if &key.from == from && &key.to == to {
                return Some(key.token);
            }
        }
        None
    }

    pub fn set_balance(&mut self, address: Address, token: Address, balance: U256) {
        self.balances.insert((address, token), balance);
        self.invalidate_cache(&address);
    }

    pub fn set_trust_limit(&mut self, from: Address, to: Address, trust_limit: U256) {
        self.trust_limits.insert((from, to), trust_limit);
        self.invalidate_cache(&from);
        self.invalidate_cache(&to);
    }

    pub fn get_outgoing_edges(&self, from: &Address) -> Vec<(Address, Address, U256)> {
        if self.dirty_nodes.borrow().contains(from) {
            self.rebuild_cache_for_node(from);
        }
    
        let mut edges = self.outgoing_edges_cache.borrow()
            .get(from)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(to, token, capacity)| {
                let key = EdgeKey { from: *from, to, token };
                let adjustment = self.capacity_adjustments.get(&key).cloned().unwrap_or(U256::from(0));
                if capacity > adjustment {
                    Some((to, token, capacity - adjustment))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Sort edges by capacity in descending order
        edges.sort_by(|a, b| b.2.cmp(&a.2));
        edges
    }

    pub fn get_incoming_edges(&self, to: &Address) -> Vec<(Address, Address, U256)> {
        if self.dirty_nodes.borrow().contains(to) {
            self.rebuild_cache_for_node(to);
        }

        let mut edges = self.incoming_edges_cache.borrow()
            .get(to)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(from, token, capacity)| {
                let key = EdgeKey { from, to: *to, token };
                let adjustment = self.capacity_adjustments.get(&key).cloned().unwrap_or(U256::from(0));
                if capacity > adjustment {
                    Some((from, token, capacity - adjustment))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // Sort edges by capacity in descending order
        edges.sort_by(|a, b| b.2.cmp(&a.2));
        edges
    }


    pub fn decrease_edge_capacity(&mut self, from: &Address, to: &Address, token: &Address, flow: U256) {
        let key = EdgeKey { from: *from, to: *to, token: *token };
        
        let adjustment = self.capacity_adjustments.entry(key.clone()).or_insert(U256::from(0));
        *adjustment = *adjustment + flow;

        self.invalidate_cache(from);
        self.invalidate_cache(to);
    }

    pub fn increase_edge_capacity(&mut self, from: &Address, to: &Address, token: &Address, flow: U256) {
        let key = EdgeKey { from: *from, to: *to, token: *token };
        
        let adjustment = self.capacity_adjustments.entry(key.clone()).or_insert(U256::from(0));
        if *adjustment >= flow {
            *adjustment = *adjustment - flow;
        } else {
            *adjustment = U256::from(0);
        }

        self.invalidate_cache(from);
        self.invalidate_cache(to);
    }

    pub fn update_edge_capacity(&mut self, from: &Address, to: &Address, token: &Address, flow: U256) {
        let key = EdgeKey { from: *from, to: *to, token: *token };
        
        
        // Update edge capacity
        if let Some(capacity) = self.edges.get_mut(&key) {
           // println!("{:?} capacity {} flow {}",key,capacity.to_decimal(), flow.to_decimal());
            if *capacity > flow {
                *capacity -= flow;
            } else {
                self.edges.remove(&key);
                // Remove from outgoing and incoming edges
                if let Some(edges) = self.outgoing_edges.get_mut(from) {
                    edges.retain(|e| e != &key);
                }
                if let Some(edges) = self.incoming_edges.get_mut(to) {
                    edges.retain(|e| e != &key);
                }
            }
        }

        
        // Update balance
        if let Some(balance) = self.balances.get_mut(&(*from, *token)) {
            *balance = if *balance > flow { *balance - flow } else { U256::from(0) };
        }
        
        // Update trust limit, but not for return-to-owner
        if token != to {
            if let Some(trust_limit) = self.trust_limits.get_mut(&(*from, *to)) {
                *trust_limit = if *trust_limit > flow { *trust_limit - flow } else { U256::from(0) };
            }
        }
        
        // Update total flow
        let total_flow_key = (*from, *to);
        let total_flow = self.total_flows.entry(total_flow_key).or_insert(U256::from(0));
        *total_flow += flow;
        
        // Update caches directly
        if let Some(cache) = self.outgoing_edges_cache.borrow_mut().get_mut(from) {
            if let Some(edge) = cache.iter_mut().find(|(t, tok, _)| t == to && tok == token) {
                edge.2 = if edge.2 > flow { edge.2 - flow } else { U256::from(0) };
            }
        }
        if let Some(cache) = self.incoming_edges_cache.borrow_mut().get_mut(to) {
            if let Some(edge) = cache.iter_mut().find(|(f, tok, _)| f == from && tok == token) {
                edge.2 = if edge.2 > flow { edge.2 - flow } else { U256::from(0) };
            }
        }
    }


    pub fn get_trust_limit(&self, from: &Address, to: &Address) -> U256 {
        self.trust_limits.get(&(*from, *to)).cloned().unwrap_or(U256::from(0))
    }

    pub fn get_balance(&self, address: &Address, token: &Address) -> U256 {
        self.balances.get(&(*address, *token)).cloned().unwrap_or(U256::from(0))
    }

    pub fn get_edge_capacity(&self, from: &Address, to: &Address, token: &Address) -> U256 {
        let key = EdgeKey { from: *from, to: *to, token: *token };
        self.edges.get(&key).cloned().unwrap_or(U256::from(0))
    }

    pub fn get_total_flow(&self, from: &Address, to: &Address) -> U256 {
        self.total_flows.get(&(*from, *to)).cloned().unwrap_or(U256::from(0))
    }
    
    pub fn get_trust_utilization(&self, from: &Address, to: &Address) -> U256 {
        let total_flow = self.get_total_flow(from, to);
        let trust_limit = self.get_trust_limit(from, to);
        if trust_limit > U256::from(0) {
            (total_flow * U256::from(100)) / trust_limit
        } else {
            U256::from(0)
        }
    }

    pub fn invalidate_cache(&self, node: &Address) {
        self.dirty_nodes.borrow_mut().insert(*node);
    }

    fn rebuild_cache_for_node(&self, node: &Address) {
        let mut outgoing_edges = Vec::new();
        let mut incoming_edges = Vec::new();
    
        if let Some(outgoing) = self.outgoing_edges.get(node) {
            for key in outgoing {
                let capacity = *self.edges.get(key).unwrap_or(&U256::from(0));
                outgoing_edges.push((key.to, key.token, capacity));
            }
        }
    
        if let Some(incoming) = self.incoming_edges.get(node) {
            for key in incoming {
                let capacity = *self.edges.get(key).unwrap_or(&U256::from(0));
                incoming_edges.push((key.from, key.token, capacity));
            }
        }
    
        // No need to filter capacities here; adjustments are applied in `get_outgoing_edges`
        self.outgoing_edges_cache.borrow_mut().insert(*node, outgoing_edges);
        self.incoming_edges_cache.borrow_mut().insert(*node, incoming_edges);
        self.dirty_nodes.borrow_mut().remove(node);
    }


    pub fn count_unique_tokens(&self) -> usize {
        let unique_tokens: HashSet<Address> = self.edges
            .keys()
            .map(|edge_key| edge_key.token)
            .collect();

        unique_tokens.len()
    }

    pub fn get_nodes(&self) -> &HashSet<Address> {
        &self.nodes
    }

    pub fn get_edges(&self) -> &HashMap<EdgeKey, U256> {
        &self.edges
    }

    pub fn estimate_max_flow(&self, source: &Address, sink: &Address) -> U256 {
        let outgoing_capacity: U256 = self.get_outgoing_edges(source)
            .iter()
            .fold(U256::from(0), |acc, (_, _, capacity)| acc + *capacity);

        let incoming_capacity: U256 = self.get_incoming_edges(sink)
            .iter()
            .fold(U256::from(0), |acc, (_, _, capacity)| acc + *capacity);

        std::cmp::min(outgoing_capacity, incoming_capacity)
    }

    pub fn get_max_capacity(&self) -> U256 {
        self.edges
            .values()
            .fold(U256::from(0), |max, &capacity| std::cmp::max(max, capacity))
    }
}