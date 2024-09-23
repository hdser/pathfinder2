use std::collections::{VecDeque, HashMap,HashSet,BTreeMap, BinaryHeap};
use crate::types::{Address, U256};
use super::graph::FlowGraph;
//use std::time::Instant;
use std::cmp::Ordering;


#[derive(Copy, Clone)]
pub enum PathSearchAlgorithm {
    BFS,
    DFS,
    BiBFS,
    BiDFS,
    AStar,
    BiAStar,
    IDDFS,
    DijkstraMaxFlow,
}

pub trait PathSearchStrategy {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>);
}


pub struct DFSPathSearch;

impl PathSearchStrategy for DFSPathSearch {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        let mut min_flow = requested_flow;

        if self.dfs(graph, source, sink, &mut path, &mut visited, &mut min_flow, max_distance, scale) {
            return (min_flow, path);
        }

        (U256::from(0), Vec::new())
    }
}

impl DFSPathSearch {
    fn dfs(
        &self,
        graph: &FlowGraph,
        current: &Address,
        sink: &Address,
        path: &mut Vec<Address>,
        visited: &mut HashSet<Address>,
        min_flow: &mut U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> bool {
        if max_distance.map_or(false, |max| path.len() > max as usize) {
            return false;
        }

        visited.insert(*current);
        path.push(*current);

        if current == sink {
            return true;
        }

        let edges = graph.get_outgoing_edges(current);
        for (to, _token, capacity) in edges {
            if !visited.contains(&to) {
                let new_flow = std::cmp::min(*min_flow, capacity);
                if new_flow > U256::from(0) && (scale.is_none() || new_flow >= scale.unwrap()) {
                    *min_flow = new_flow;
                    if self.dfs(graph, &to, sink, path, visited, min_flow, max_distance, scale) {
                        return true;
                    }
                }
            }
        }

        path.pop();
        false
    }
}

pub struct BiDFSPathSearch;

impl PathSearchStrategy for BiDFSPathSearch {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut forward_path = Vec::new();
        let mut backward_path = Vec::new();
        let mut forward_visited = HashSet::new();
        let mut backward_visited = HashSet::new();
        let mut forward_flow = requested_flow;
        let mut backward_flow = requested_flow;

        if self.bidirectional_dfs(
            graph,
            source,
            sink,
            &mut forward_path,
            &mut backward_path,
            &mut forward_visited,
            &mut backward_visited,
            &mut forward_flow,
            &mut backward_flow,
            max_distance,
            scale,
        ) {
            let meeting_point = forward_visited.intersection(&backward_visited).next().cloned();
            if let Some(meeting) = meeting_point {
                let mut complete_path = forward_path.clone();
                complete_path.extend(backward_path.iter().rev().skip_while(|&x| x != &meeting));
                let min_flow = std::cmp::min(forward_flow, backward_flow);
                return (min_flow, complete_path);
            }
        }

        (U256::from(0), Vec::new())
    }
}

impl BiDFSPathSearch {
    fn bidirectional_dfs(
        &self,
        graph: &FlowGraph,
        current_forward: &Address,
        current_backward: &Address,
        forward_path: &mut Vec<Address>,
        backward_path: &mut Vec<Address>,
        forward_visited: &mut HashSet<Address>,
        backward_visited: &mut HashSet<Address>,
        forward_flow: &mut U256,
        backward_flow: &mut U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> bool {
        if max_distance.map_or(false, |max| forward_path.len() + backward_path.len() > max as usize * 2) {
            return false;
        }

        forward_visited.insert(*current_forward);
        backward_visited.insert(*current_backward);
        forward_path.push(*current_forward);
        backward_path.push(*current_backward);

        if forward_visited.contains(current_backward) || backward_visited.contains(current_forward) {
            return true;
        }

        let forward_edges = graph.get_outgoing_edges(current_forward);
        let backward_edges = graph.get_incoming_edges(current_backward);

        for (to, _, capacity) in forward_edges {
            if !forward_visited.contains(&to) {
                let new_flow = std::cmp::min(*forward_flow, capacity);
                if new_flow > U256::from(0) && (scale.is_none() || new_flow >= scale.unwrap()) {
                    *forward_flow = new_flow;
                    if self.bidirectional_dfs(
                        graph, &to, current_backward,
                        forward_path, backward_path,
                        forward_visited, backward_visited,
                        forward_flow, backward_flow,
                        max_distance, scale,
                    ) {
                        return true;
                    }
                }
            }
        }

        for (from, _, capacity) in backward_edges {
            if !backward_visited.contains(&from) {
                let new_flow = std::cmp::min(*backward_flow, capacity);
                if new_flow > U256::from(0) && (scale.is_none() || new_flow >= scale.unwrap()) {
                    *backward_flow = new_flow;
                    if self.bidirectional_dfs(
                        graph, current_forward, &from,
                        forward_path, backward_path,
                        forward_visited, backward_visited,
                        forward_flow, backward_flow,
                        max_distance, scale,
                    ) {
                        return true;
                    }
                }
            }
        }

        forward_path.pop();
        backward_path.pop();
        false
    }
}


pub struct BFSPathSearch;

impl PathSearchStrategy for BFSPathSearch { 
    /* 
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut queue = VecDeque::new();
        let mut parent: BTreeMap<Address, (Address, Address, U256)> = BTreeMap::new();
        let mut flow: BTreeMap<Address, U256> = BTreeMap::new();

        queue.push_back(*source);
        flow.insert(*source, requested_flow);

        let min_flow = scale.unwrap_or(U256::from(1));

        while let Some(node) = queue.pop_front() {
            if node == *sink {
                break;
            }

            let current_flow = *flow.get(&node).unwrap();
            if current_flow < min_flow {
                continue;
            }

            let mut outgoing_edges: Vec<(Address, Address, U256)> = graph.get_outgoing_edges(&node);

            for (to, token, capacity) in outgoing_edges {
                if !parent.contains_key(&to) && capacity >= min_flow {
                    let new_flow = std::cmp::min(current_flow, capacity);
                    if new_flow >= min_flow && new_flow > U256::from(0) {
                        parent.insert(to, (node, token, new_flow));
                        flow.insert(to, new_flow);
                        queue.push_back(to);

                        if let Some(max_dist) = max_distance {
                            if parent.len() as u64 > max_dist {
                                return (U256::from(0), Vec::new());
                            }
                        }
                    }
                }
            }
        }

        if !parent.contains_key(sink) {
            return (U256::from(0), Vec::new());
        }

        let path = self.trace_path(&parent, source, sink);
        //let min_path_flow = path.windows(2).map(|window| {
        //    let (_, _, flow) = parent[&window[1]];
        //    flow
        //}).min().unwrap_or(U256::from(0));

        let min_path_flow = path.windows(2)
            .map(|window| parent[&window[1]].2)
            .min()
            .unwrap_or(U256::from(0));

        (min_path_flow, path)
    }
    */
  //  /* 
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut queue = VecDeque::new();
        let mut parent: BTreeMap<Address, (Address, Address, U256)> = BTreeMap::new();
        let mut flow: BTreeMap<Address, U256> = BTreeMap::new();

        queue.push_back(*source);
        flow.insert(*source, requested_flow);

        let min_flow = scale.unwrap_or(U256::from(1));

        while let Some(node) = queue.pop_front() {
            if node == *sink {
                break;
            }

            let current_flow = *flow.get(&node).unwrap();
            if current_flow < min_flow {
                continue;
            }

            let outgoing_edges: Vec<(Address, Address, U256)> = graph.get_outgoing_edges(&node);

            for (to, token, capacity) in outgoing_edges {
                if !parent.contains_key(&to) && capacity >= min_flow {
                    let new_flow = std::cmp::min(current_flow, capacity);
                    if new_flow >= min_flow && new_flow > U256::from(0) {
                        parent.insert(to, (node, token, new_flow));
                        flow.insert(to, new_flow);
                        queue.push_back(to);

                        if let Some(max_dist) = max_distance {
                            if parent.len() as u64 > max_dist {
                                return (U256::from(0), Vec::new());
                            }
                        }
                    }
                }
            }
        }

        if !parent.contains_key(sink) {
            return (U256::from(0), Vec::new());
        }

        let path = self.trace_path(&parent, source, sink);
        let min_path_flow = parent[sink].2;

        (min_path_flow, path)
    }
  //  */
    
}

impl BFSPathSearch {
    fn trace_path(&self, parent: &BTreeMap<Address, (Address, Address, U256)>, source: &Address, sink: &Address) -> Vec<Address> {
        let mut path = Vec::new();
        let mut current = *sink;
        while current != *source {
            path.push(current);
            let (prev, _, _) = parent[&current];
            current = prev;
        }
        path.push(*source);
        path.reverse();
        path
    }
}

pub struct BiBFSPathSearch;

impl PathSearchStrategy for BiBFSPathSearch {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        if source == sink {
            return (U256::default(), vec![]);
        }

        let mut forward_queue = VecDeque::new();
        let mut backward_queue = VecDeque::new();
        let mut forward_visited = HashMap::new();
        let mut backward_visited = HashMap::new();

        forward_queue.push_back((*source, requested_flow, 0));
        backward_queue.push_back((*sink, requested_flow, 0));
        forward_visited.insert(*source, (*source, requested_flow));
        backward_visited.insert(*sink, (*sink, requested_flow));

        while !forward_queue.is_empty() && !backward_queue.is_empty() {
            if let Some((meeting_node, flow)) = self.extend_search(graph, &mut forward_queue, &mut forward_visited, &backward_visited, true, max_distance, scale) {
                return self.construct_path(meeting_node, flow, &forward_visited, &backward_visited);
            }

            if let Some((meeting_node, flow)) = self.extend_search(graph, &mut backward_queue, &mut backward_visited, &forward_visited, false, max_distance, scale) {
                return self.construct_path(meeting_node, flow, &forward_visited, &backward_visited);
            }
        }

        (U256::from(0), Vec::new())
    }
}

impl BiBFSPathSearch {
    fn extend_search(
        &self,
        graph: &FlowGraph,
        queue: &mut VecDeque<(Address, U256, u64)>,
        visited: &mut HashMap<Address, (Address, U256)>,
        other_visited: &HashMap<Address, (Address, U256)>,
        is_forward: bool,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> Option<(Address, U256)> {
        if let Some((node, flow, depth)) = queue.pop_front() {
            if let Some(max) = max_distance {
                if depth >= max * 3 {
                    return None;
                }
            }

            let edges = if is_forward {
                graph.get_outgoing_edges(&node)
            } else {
                graph.get_incoming_edges(&node)
            };

            for &(next, _, capacity) in edges.iter() {
                let new_flow = std::cmp::min(flow, capacity);
                if new_flow > U256::from(0) && (scale.is_none() || new_flow >= scale.unwrap()) && !visited.contains_key(&next) {
                    visited.insert(next, (node, new_flow));
                    queue.push_back((next, new_flow, depth + 1));

                    if let Some(&(_, other_flow)) = other_visited.get(&next) {
                        return Some((next, std::cmp::min(new_flow, other_flow)));
                    }
                }
            }
        }
        None
    }

    fn construct_path(
        &self,
        meeting_node: Address,
        flow: U256,
        forward_visited: &HashMap<Address, (Address, U256)>,
        backward_visited: &HashMap<Address, (Address, U256)>,
    ) -> (U256, Vec<Address>) {
        let mut path = Vec::new();
        let mut current = meeting_node;

        // Construct forward path
        while let Some(&(parent, _)) = forward_visited.get(&current) {
            path.push(current);
            if current == parent {
                break;
            }
            current = parent;
        }
        path.reverse();

        // Construct backward path
        current = meeting_node;
        while let Some(&(parent, _)) = backward_visited.get(&current) {
            if current != meeting_node {
                path.push(current);
            }
            if current == parent {
                break;
            }
            current = parent;
        }

        (flow, path)
    }
}



pub struct AStarPathSearch;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: U256,
    node: Address,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}



impl PathSearchStrategy for AStarPathSearch {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        _requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut heap = BinaryHeap::new();
        let mut dist: HashMap<Address, U256> = HashMap::new();
        let mut prev: HashMap<Address, Address> = HashMap::new();

        heap.push(State { cost: U256::MAX, node: *source });
        dist.insert(*source, U256::MAX);

        let min_flow = scale.unwrap_or(U256::from(1));
        let mut max_dist = max_distance.unwrap_or(u64::MAX);

        while let Some(State { cost, node }) = heap.pop() {
            if node == *sink {
                return self.reconstruct_path(source, sink, &prev, cost);
            }

            if cost < *dist.get(&node).unwrap_or(&U256::from(0)) {
                continue;
            }

            if max_dist == 0 {
                break;
            }
            max_dist = max_dist.saturating_sub(1);

            for (next, _, capacity) in graph.get_outgoing_edges(&node) {
                let new_cost = std::cmp::min(cost, capacity);
                if new_cost >= min_flow && new_cost > *dist.get(&next).unwrap_or(&U256::from(0)) {
                    dist.insert(next, new_cost);
                    prev.insert(next, node);
                    let priority = new_cost + self.heuristic(graph, &next, sink);
                    heap.push(State { cost: priority, node: next });
                }
            }
        }

        (U256::from(0), Vec::new())
    }
}

impl AStarPathSearch {
    fn heuristic(&self, graph: &FlowGraph, node: &Address, goal: &Address) -> U256 {
        if node == goal {
            return U256::MAX;
        }

        let outgoing_edges = graph.get_outgoing_edges(node);
        
        if outgoing_edges.is_empty() {
            return U256::from(0);
        }

        // Simplified heuristic: use the maximum outgoing capacity
        outgoing_edges.iter()
            .map(|&(_, _, cap)| cap)
            .max()
            .unwrap_or(U256::from(0))
    }

    fn reconstruct_path(&self, source: &Address, sink: &Address, prev: &HashMap<Address, Address>, flow: U256) -> (U256, Vec<Address>) {
        let mut path = Vec::new();
        let mut current = *sink;
        while current != *source {
            path.push(current);
            current = prev[&current];
        }
        path.push(*source);
        path.reverse();
        (flow, path)
    }
}


pub struct BiAStarPathSearch;

impl PathSearchStrategy for BiAStarPathSearch {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        _requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut forward_heap = BinaryHeap::new();
        let mut backward_heap = BinaryHeap::new();
        let mut forward_dist: HashMap<Address, U256> = HashMap::new();
        let mut backward_dist: HashMap<Address, U256> = HashMap::new();
        let mut forward_prev: HashMap<Address, Address> = HashMap::new();
        let mut backward_prev: HashMap<Address, Address> = HashMap::new();

        forward_heap.push(State { cost: U256::MAX, node: *source });
        backward_heap.push(State { cost: U256::MAX, node: *sink });
        forward_dist.insert(*source, U256::MAX);
        backward_dist.insert(*sink, U256::MAX);

        let min_flow = scale.unwrap_or(U256::from(1));
        let max_dist = max_distance.unwrap_or(u64::MAX);
        let mut best_path: Option<(Address, U256)> = None;

        while !forward_heap.is_empty() && !backward_heap.is_empty() {
            if let Some((meeting_node, flow)) = self.extend_search(graph, &mut forward_heap, &mut forward_dist, &mut forward_prev, &backward_dist, true, source, sink, max_dist, min_flow) {
                best_path = Some((meeting_node, flow));
                break;
            }

            if let Some((meeting_node, flow)) = self.extend_search(graph, &mut backward_heap, &mut backward_dist, &mut backward_prev, &forward_dist, false, sink, source, max_dist, min_flow) {
                best_path = Some((meeting_node, flow));
                break;
            }
        }

        if let Some((meeting_node, flow)) = best_path {
            let path = self.reconstruct_path(source, sink, &meeting_node, &forward_prev, &backward_prev);
            (flow, path)
        } else {
            (U256::from(0), Vec::new())
        }
    }
}

impl BiAStarPathSearch {
    fn extend_search(
        &self,
        graph: &FlowGraph,
        heap: &mut BinaryHeap<State>,
        dist: &mut HashMap<Address, U256>,
        prev: &mut HashMap<Address, Address>,
        other_dist: &HashMap<Address, U256>,
        is_forward: bool,
        _start: &Address,
        goal: &Address,
        max_distance: u64,
        min_flow: U256,
    ) -> Option<(Address, U256)> {
        if let Some(State { cost, node }) = heap.pop() {
            if cost < *dist.get(&node).unwrap_or(&U256::from(0)) {
                return None;
            }

            let edges = if is_forward {
                graph.get_outgoing_edges(&node)
            } else {
                graph.get_incoming_edges(&node)
            };

            for (next, _, capacity) in edges {
                let new_cost = std::cmp::min(cost, capacity);
                if new_cost >= min_flow && new_cost > *dist.get(&next).unwrap_or(&U256::from(0)) {
                    dist.insert(next, new_cost);
                    prev.insert(next, node);
                    let priority = new_cost + self.heuristic(graph, &next, goal, is_forward);
                    heap.push(State { cost: priority, node: next });

                    if let Some(other_cost) = other_dist.get(&next) {
                        let total_flow = std::cmp::min(new_cost, *other_cost);
                        return Some((next, total_flow));
                    }
                }
            }

            if prev.len() as u64 > max_distance {
                return Some((node, U256::from(0)));
            }
        }

        None
    }

    fn heuristic(&self, graph: &FlowGraph, node: &Address, goal: &Address, is_forward: bool) -> U256 {
        if node == goal {
            return U256::MAX;
        }

        let edges = if is_forward {
            graph.get_outgoing_edges(node)
        } else {
            graph.get_incoming_edges(node)
        };
        
        edges.iter()
            .map(|&(_, _, cap)| cap)
            .max()
            .unwrap_or(U256::from(0))
    }

    fn reconstruct_path(
        &self,
        source: &Address,
        sink: &Address,
        meeting_node: &Address,
        forward_prev: &HashMap<Address, Address>,
        backward_prev: &HashMap<Address, Address>,
    ) -> Vec<Address> {
        let mut forward_path = Vec::new();
        let mut backward_path = Vec::new();

        let mut current = *meeting_node;
        while current != *source {
            forward_path.push(current);
            current = forward_prev[&current];
        }
        forward_path.push(*source);
        forward_path.reverse();

        current = *meeting_node;
        while current != *sink {
            if current != *meeting_node {
                backward_path.push(current);
            }
            current = backward_prev[&current];
        }
        backward_path.push(*sink);

        forward_path.extend(backward_path);
        forward_path
    }
}


pub struct IDDFSPathSearch;

impl PathSearchStrategy for IDDFSPathSearch {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let max_depth = max_distance.unwrap_or(u64::MAX);
        let min_flow = scale.unwrap_or(U256::from(1));

        for depth in 1..=max_depth {
            let mut path = Vec::new();
            let mut visited = HashMap::new();
            let (flow, found) = self.depth_limited_search(
                graph,
                source,
                sink,
                depth,
                requested_flow,
                min_flow,
                &mut path,
                &mut visited,
            );

            if found {
                return (flow, path);
            }
        }

        (U256::from(0), Vec::new())
    }
}

impl IDDFSPathSearch {
    fn depth_limited_search(
        &self,
        graph: &FlowGraph,
        current: &Address,
        sink: &Address,
        depth: u64,
        max_flow: U256,
        min_flow: U256,
        path: &mut Vec<Address>,
        visited: &mut HashMap<Address, U256>,
    ) -> (U256, bool) {
        if current == sink {
            return (max_flow, true);
        }

        if depth == 0 {
            return (U256::from(0), false);
        }

        path.push(*current);
        visited.insert(*current, max_flow);

        let edges = graph.get_outgoing_edges(current);
        for (to, _, capacity) in edges {
            if capacity < min_flow {
                continue;
            }

            let new_max_flow = std::cmp::min(max_flow, capacity);
            if let Some(&prev_flow) = visited.get(&to) {
                if prev_flow >= new_max_flow {
                    continue;
                }
            }

            let (flow, found) = self.depth_limited_search(
                graph,
                &to,
                sink,
                depth - 1,
                new_max_flow,
                min_flow,
                path,
                visited,
            );

            if found {
                return (flow, true);
            }
        }

        path.pop();
        (U256::from(0), false)
    }
}



pub struct DijkstraMaxFlow;



impl PathSearchStrategy for DijkstraMaxFlow {
    fn find_path(
        &self,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        _requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let mut heap = BinaryHeap::new();
        let mut max_flow_to = HashMap::new();
        let mut came_from = HashMap::new();
        
        heap.push(State { cost: U256::MAX, node: *source });
        max_flow_to.insert(*source, U256::MAX);

        let min_flow = scale.unwrap_or(U256::from(1));
        let max_dist = max_distance.unwrap_or(u64::MAX);

        while let Some(State { cost: current_flow, node }) = heap.pop() {
            if node == *sink {
                return (current_flow, self.reconstruct_path(source, sink, &came_from));
            }

            if current_flow < *max_flow_to.get(&node).unwrap_or(&U256::from(0)) {
                continue;
            }

            if came_from.len() as u64 > max_dist {
                break;
            }

            for (next, _, capacity) in graph.get_outgoing_edges(&node) {
                let new_flow = std::cmp::min(current_flow, capacity);
                if new_flow >= min_flow && new_flow > *max_flow_to.get(&next).unwrap_or(&U256::from(0)) {
                    heap.push(State { cost: new_flow, node: next });
                    max_flow_to.insert(next, new_flow);
                    came_from.insert(next, node);
                }
            }
        }

        (U256::from(0), Vec::new())
    }
}

impl DijkstraMaxFlow {
    fn reconstruct_path(&self, source: &Address, sink: &Address, came_from: &HashMap<Address, Address>) -> Vec<Address> {
        let mut path = Vec::new();
        let mut current = *sink;
        while current != *source {
            path.push(current);
            current = *came_from.get(&current).unwrap();
        }
        path.push(*source);
        path.reverse();
        path
    }
}






pub struct PathSearch;

impl PathSearch {
    pub fn find_path(
        algorithm: PathSearchAlgorithm,
        graph: &mut FlowGraph,
        source: &Address,
        sink: &Address,
        requested_flow: U256,
        max_distance: Option<u64>,
        scale: Option<U256>,
    ) -> (U256, Vec<Address>) {
        let strategy: Box<dyn PathSearchStrategy> = match algorithm {
            PathSearchAlgorithm::BFS => Box::new(BFSPathSearch),
            PathSearchAlgorithm::DFS => Box::new(DFSPathSearch),
            PathSearchAlgorithm::BiBFS => Box::new(BiBFSPathSearch),
            PathSearchAlgorithm::BiDFS => Box::new(BiDFSPathSearch),
            PathSearchAlgorithm::AStar => Box::new(AStarPathSearch),
            PathSearchAlgorithm::BiAStar => Box::new(BiAStarPathSearch),
            PathSearchAlgorithm::IDDFS => Box::new(IDDFSPathSearch),
            PathSearchAlgorithm::DijkstraMaxFlow => Box::new(DijkstraMaxFlow),
        };

        strategy.find_path(graph, source, sink, requested_flow, max_distance, scale)
    }
}
