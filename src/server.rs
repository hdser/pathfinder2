use crate::graph;
use crate::io::{import_from_safes_binary, read_edges_binary, read_edges_csv};
use crate::types::edge::EdgeDB;
use crate::types::{Address, Edge, U256};
use json::JsonValue;
use num_bigint::BigUint;
use regex::Regex;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::io::Read;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::Deref;
use std::str::FromStr;
use std::sync::mpsc::TrySendError;
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread;

use crate::graph_visualizer::GraphVisualizer;
use crate::safe_db::db::DB;

use crate::graph_new::graph::FlowGraph;
use crate::graph_new::graph_analytics::GraphAnalytics;
use crate::graph_new::network_flow::{NetworkFlow, NetworkFlowAlgorithm};
use crate::graph_new::ford_fulkerson::FordFulkerson;
use crate::graph_new::capacity_scaling::CapacityScaling;
use crate::graph_new::path_search::PathSearchAlgorithm;
use crate::graph_new::flow_recorder::FlowRecorder;
use crate::graph_new::dinic::DinicsAlgorithm;
use crate::graph_new::push_relabel::PushRelabelAlgorithm;

struct LoadedGraphData {
    edge_db: Arc<EdgeDB>,
    flow_graph: Arc<Mutex<FlowGraph>>,
}

struct JsonRpcRequest {
    id: JsonValue,
    method: String,
    params: JsonValue,
}

struct InputValidationError(String);
impl Error for InputValidationError {}

impl Debug for InputValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.0)
    }
}
impl Display for InputValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {}", self.0)
    }
}

fn validate_and_parse_ethereum_address(address: &str) -> Result<Address, Box<dyn Error>> {
    let re = Regex::new(r"^0x[0-9a-fA-F]{40}$").unwrap();
    if re.is_match(address) {
        Ok(Address::from(address))
    } else {
        Err(Box::new(InputValidationError(format!(
            "Invalid Ethereum address: {}",
            address
        ))))
    }
}

fn validate_and_parse_u256(value_str: &str) -> Result<U256, Box<dyn Error>> {
    match BigUint::from_str(value_str) {
        Ok(parsed_value) => {
            if parsed_value > U256::MAX.into() {
                Err(Box::new(InputValidationError(format!(
                    "Value {} is too large. Maximum value is {}.",
                    parsed_value,
                    U256::MAX
                ))))
            } else {
                Ok(U256::from_bigint_truncating(parsed_value))
            }
        }
        Err(e) => Err(Box::new(InputValidationError(format!(
            "Invalid value: {}. Couldn't parse value: {}",
            value_str, e
        )))),
    }
}

pub fn start_server(listen_at: &str, queue_size: usize, threads: u64) {
    let loaded_data: Arc<RwLock<Option<LoadedGraphData>>> = Arc::new(RwLock::new(None));

    let (sender, receiver) = mpsc::sync_channel(queue_size);
    let protected_receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..threads {
        let rec = protected_receiver.clone();
        let data = loaded_data.clone();
        thread::spawn(move || loop {
            let socket = rec.lock().unwrap().recv().unwrap();
            if let Err(e) = handle_connection(data.deref(), socket) {
                println!("Error handling connection: {e}");
            }
        });
    }
    let listener = TcpListener::bind(listen_at).expect("Could not create server.");
    loop {
        match listener.accept() {
            Ok((socket, _)) => match sender.try_send(socket) {
                Ok(()) => {}
                Err(TrySendError::Full(mut socket)) => {
                    let _ = socket.write_all(b"HTTP/1.1 503 Service Unavailable\r\n\r\n");
                }
                Err(TrySendError::Disconnected(_)) => {
                    panic!("Internal communication channel disconnected.");
                }
            },
            Err(e) => println!("Error accepting connection: {e}"),
        }
    }
}

fn handle_connection(
    loaded_data: &RwLock<Option<LoadedGraphData>>,
    mut socket: TcpStream,
) -> Result<(), Box<dyn Error>> {
    let request = read_request(&mut socket)?;
    match request.method.as_str() {
        "load_edges_binary" => {
            let response = match load_edges_binary(loaded_data, &request.params["file"].to_string()) {
                Ok(len) => jsonrpc_response(request.id, len),
                Err(e) => {
                    jsonrpc_error_response(request.id, -32000, &format!("Error loading edges: {e}"))
                }
            };
            socket.write_all(response.as_bytes())?;
        }
        "load_edges_csv" => {
            let response = match load_edges_csv(loaded_data, &request.params["file"].to_string()) {
                Ok(len) => jsonrpc_response(request.id, len),
                Err(e) => {
                    jsonrpc_error_response(request.id, -32000, &format!("Error loading edges: {e}"))
                }
            };
            socket.write_all(response.as_bytes())?;
        }
        "load_safes_binary" => {
            let response = match load_safes_binary(loaded_data, &request.params["file"].to_string()) {
                Ok(len) => jsonrpc_response(request.id, len),
                Err(e) => {
                    jsonrpc_error_response(request.id, -32000, &format!("Error loading edges: {e}"))
                }
            };
            socket.write_all(response.as_bytes())?;
        }
        "compute_transfer" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                compute_transfer(request, &data.edge_db, socket)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_FordFulkerson" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                generic_compute_transfer_new(&request, &data.edge_db, &data.flow_graph, socket, FordFulkerson)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_CapacityScaling" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                generic_compute_transfer_new(&request, &data.edge_db, &data.flow_graph, socket, CapacityScaling)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_Dinic" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                generic_compute_transfer_new(&request, &data.edge_db, &data.flow_graph, socket, DinicsAlgorithm)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_PushRelabel" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                generic_compute_transfer_new(&request, &data.edge_db, &data.flow_graph, socket, PushRelabelAlgorithm)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "visualize_flow_graph_FordFulkerson" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                let response = generic_visualize_flow_graph_new(&data.edge_db, &data.flow_graph, &request.params, FordFulkerson)?;
                socket.write_all(jsonrpc_response(request.id, response).as_bytes())?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        },
        "visualize_flow_graph_CapacityScaling" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                let response = generic_visualize_flow_graph_new(&data.edge_db, &data.flow_graph, &request.params, CapacityScaling)?;
                socket.write_all(jsonrpc_response(request.id, response).as_bytes())?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        },
        "visualize_flow_graph_Dinic" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                let response = generic_visualize_flow_graph_new(&data.edge_db, &data.flow_graph, &request.params, DinicsAlgorithm)?;
                socket.write_all(jsonrpc_response(request.id, response).as_bytes())?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        },
        "visualize_flow_graph_PushRelabel" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                let response = generic_visualize_flow_graph_new(&data.edge_db, &data.flow_graph, &request.params, PushRelabelAlgorithm)?;
                socket.write_all(jsonrpc_response(request.id, response).as_bytes())?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        },
        "compute_FordFulkerson_with_visualization" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                compute_transfer_with_visualization(&request, &data.edge_db, &data.flow_graph, socket, FordFulkerson)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_CapacityScaling_with_visualization" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                compute_transfer_with_visualization(&request, &data.edge_db, &data.flow_graph, socket, CapacityScaling)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_Dinic_with_visualization" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                compute_transfer_with_visualization(&request, &data.edge_db, &data.flow_graph, socket, DinicsAlgorithm)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "compute_PushRelabel_with_visualization" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                compute_transfer_with_visualization(&request, &data.edge_db, &data.flow_graph, socket, PushRelabelAlgorithm)?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        }
        "visualize_flow_graph" => {
            let data = loaded_data.read().unwrap();
            if let Some(ref data) = *data {
                let response = generic_visualize_flow_graph(&data.edge_db, &request.params, graph::compute_flow)?;
                socket.write_all(jsonrpc_response(request.id, response).as_bytes())?;
            } else {
                socket.write_all(jsonrpc_error_response(request.id, -32000, "No data loaded").as_bytes())?;
            }
        },
        "update_edges" => {
            let response = match request.params {
                JsonValue::Array(updates) => {
                    let mut data = loaded_data.write().unwrap();
                    if let Some(ref mut data) = *data {
                        match update_edges(&mut data.edge_db, &mut data.flow_graph, updates) {
                            Ok(len) => jsonrpc_response(request.id, len),
                            Err(e) => jsonrpc_error_response(
                                request.id,
                                -32000,
                                &format!("Error updating edges: {e}"),
                            ),
                        }
                    } else {
                        jsonrpc_error_response(request.id, -32000, "No data loaded")
                    }
                },
                _ => {
                    jsonrpc_error_response(request.id, -32602, "Invalid arguments: Expected array.")
                }
            };
            socket.write_all(response.as_bytes())?;
        }
        _ => socket
            .write_all(jsonrpc_error_response(request.id, -32601, "Method not found").as_bytes())?,
    };
    Ok(())
}

fn load_edges_binary(loaded_data: &RwLock<Option<LoadedGraphData>>, file: &str) -> Result<usize, Box<dyn Error>> {
    let db = import_from_safes_binary(file)?;
    let edge_db = Arc::new(db.edges().clone());
    let flow_graph = Arc::new(Mutex::new(edge_db_to_flow_graph(edge_db.clone(), &db)));
    let len = edge_db.edge_count();
    
    let mut data = loaded_data.write().unwrap();
    *data = Some(LoadedGraphData {
        edge_db,
        flow_graph,
    });
    
    Ok(len)
}

fn load_edges_csv(loaded_data: &RwLock<Option<LoadedGraphData>>, file: &str) -> Result<usize, Box<dyn Error>> {
    let updated_edges = Arc::new(read_edges_csv(&file.to_string())?);
    let len = updated_edges.edge_count();
    let flow_graph = Arc::new(Mutex::new(edge_db_to_flow_graph(updated_edges.clone(), &DB::default())));
    let mut data = loaded_data.write().unwrap();
    *data = Some(LoadedGraphData {
        edge_db: updated_edges,
        flow_graph,
    });
    Ok(len)
}

fn load_safes_binary(loaded_data: &RwLock<Option<LoadedGraphData>>, file: &str) -> Result<usize, Box<dyn Error>> {
    let db = import_from_safes_binary(file)?;
    let updated_edges = Arc::new(db.edges().clone());
    let len = updated_edges.edge_count();
    let flow_graph = Arc::new(Mutex::new(edge_db_to_flow_graph(updated_edges.clone(), &db)));
    let mut data = loaded_data.write().unwrap();
    *data = Some(LoadedGraphData {
        edge_db: updated_edges,
        flow_graph,
    });
    Ok(len)
}


fn compute_transfer(
    request: JsonRpcRequest,
    edges: &EdgeDB,
    mut socket: TcpStream,
) -> Result<(), Box<dyn Error>> {
    socket.write_all(chunked_header().as_bytes())?;

    let parsed_value_param = match request.params["value"].as_str() {
        Some(value_str) => validate_and_parse_u256(value_str)?,
        None => U256::MAX,
    };

    let from_address = validate_and_parse_ethereum_address(&request.params["from"].to_string())?;
    let to_address = validate_and_parse_ethereum_address(&request.params["to"].to_string())?;

    let max_distances = if request.params["iterative"].as_bool().unwrap_or_default() {
        vec![Some(1), Some(2), None]
    } else {
        vec![None]
    };

    let max_transfers = request.params["max_transfers"].as_u64();
    for max_distance in max_distances {
        let (flow, transfers) = graph::compute_flow(
            &from_address,
            &to_address,
            edges,
            parsed_value_param,
            max_distance,
            max_transfers,
        );
        println!("Computed flow with max distance {max_distance:?}: {}",flow.to_decimal());
        socket.write_all(
            chunked_response(
                &(jsonrpc_result(
                    request.id.clone(),
                    json::object! {
                        maxFlowValue: flow.to_decimal(),
                        final: max_distance.is_none(),
                        transferSteps: transfers.into_iter().map(|e| json::object! {
                            from: e.from.to_checksummed_hex(),
                            to: e.to.to_checksummed_hex(),
                            token_owner: e.token.to_checksummed_hex(),
                            value: e.capacity.to_decimal(),
                        }).collect::<Vec<_>>(),
                    },
                ) + "\r\n"),
            )
            .as_bytes(),
        )?;
    }
    socket.write_all(chunked_close().as_bytes())?;
    Ok(())
}


fn update_edges(
    edge_db: &mut Arc<EdgeDB>,
    flow_graph: &Arc<Mutex<FlowGraph>>,
    updates: Vec<JsonValue>,
) -> Result<usize, Box<dyn Error>> {
    let updates = updates
        .into_iter()
        .map(|e| Edge {
            from: Address::from(e["from"].to_string().as_str()),
            to: Address::from(e["to"].to_string().as_str()),
            token: Address::from(e["token_owner"].to_string().as_str()),
            capacity: U256::from(e["capacity"].to_string().as_str()),
        })
        .collect::<Vec<_>>();
    if updates.is_empty() {
        return Ok(edge_db.edge_count());
    }

    let mut updating_edges = (**edge_db).clone();
    let mut flow_graph = flow_graph.lock().unwrap();
    for update in &updates {
        updating_edges.update(update.clone());
        flow_graph.update_edge_capacity(&update.from, &update.to, &update.token, update.capacity);
    }
    let len = updating_edges.edge_count();
    *edge_db = Arc::new(updating_edges);
    Ok(len)
}

fn read_request(socket: &mut TcpStream) -> Result<JsonRpcRequest, Box<dyn Error>> {
    let payload = read_payload(socket)?;
    let mut request = json::parse(&String::from_utf8(payload)?)?;
    println!("Request: {request}");
    let id = request["id"].take();
    let params = request["params"].take();
    match request["method"].as_str() {
        Some(method) => Ok(JsonRpcRequest {
            id,
            method: method.to_string(),
            params,
        }),
        _ => Err(From::from("Invalid JSON-RPC request: {request}")),
    }
}

fn read_payload(socket: &mut TcpStream) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut reader = BufReader::new(socket);
    let mut length = 0;
    for result in reader.by_ref().lines() {
        let l = result?;
        if l.is_empty() {
            break;
        }

        let header = "content-length: ";
        if l.to_lowercase().starts_with(header) {
            length = l[header.len()..].parse::<usize>()?;
        }
    }
    let mut payload = vec![0u8; length];

    reader.read_exact(payload.as_mut_slice())?;
    Ok(payload)
}

fn jsonrpc_response(id: JsonValue, result: impl Into<json::JsonValue>) -> String {
    let payload = jsonrpc_result(id, result);
    format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
}

fn jsonrpc_result(id: JsonValue, result: impl Into<json::JsonValue>) -> String {
    json::object! {
        jsonrpc: "2.0",
        id: id,
        result: result.into(),
    }
    .dump()
}

fn jsonrpc_error_response(id: JsonValue, code: i64, message: &str) -> String {
    let payload = json::object! {
        jsonrpc: "2.0",
        id: id,
        error: {
            code: code,
            message: message
        }
    }
    .dump();
    format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
}

fn chunked_header() -> String {
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_string()
}

fn chunked_response(data: &str) -> String {
    if data.is_empty() {
        String::new()
    } else {
        format!("{:x}\r\n{}\r\n", data.len(), data)
    }
}

fn chunked_close() -> String {
    "0\r\n\r\n".to_string()
}

fn load_safes_and_convert_to_flow_graph(file: &str) -> Result<(Arc<FlowGraph>, Arc<EdgeDB>), Box<dyn Error>> {
    let db = import_from_safes_binary(file)?;
    let edge_db = Arc::new(db.edges().clone());
    let flow_graph = Arc::new(edge_db_to_flow_graph(edge_db.clone(), &db));
    Ok((flow_graph, edge_db))
}

fn edge_db_to_flow_graph(edge_db: Arc<EdgeDB>, db: &DB) -> FlowGraph {
    let mut flow_graph = FlowGraph::new();
    
    // Set balances
    for (address, safe) in db.safes() {
        for (token, balance) in &safe.balances {
            flow_graph.set_balance(*address, *token, *balance);
        }
    }

    // Add edges and set trust limits
    for edge in edge_db.edges() {
        flow_graph.add_edge(edge.from, edge.to, edge.token, edge.capacity);
        
        // Set trust limit
        let current_limit = flow_graph.get_trust_limit(&edge.from, &edge.to);
        flow_graph.set_trust_limit(edge.from, edge.to, std::cmp::max(current_limit, edge.capacity));
    }
    
    flow_graph
}

fn generic_compute_transfer_new<A: NetworkFlowAlgorithm + Clone>(
    request: &JsonRpcRequest,
    _edge_db: &EdgeDB,
    flow_graph: &Arc<Mutex<FlowGraph>>,
    mut socket: TcpStream,
    algorithm: A,
) -> Result<(), Box<dyn Error>> {
    socket.write_all(chunked_header().as_bytes())?;

    let parsed_value_param = match request.params["value"].as_str() {
        Some(value_str) => validate_and_parse_u256(value_str)?,
        None => U256::MAX,
    };

    let from_address = validate_and_parse_ethereum_address(&request.params["from"].to_string())?;
    let to_address = validate_and_parse_ethereum_address(&request.params["to"].to_string())?;

    let max_distances = if request.params["iterative"].as_bool().unwrap_or_default() {
        vec![Some(1), Some(2), None]
    } else {
        vec![None]
    };

    let path_search_algorithm = match request.params["path_search_algorithm"].as_str().unwrap_or("BFS") {
        "BFS" => PathSearchAlgorithm::BFS,
        "BiBFS" => PathSearchAlgorithm::BiBFS,
        _ => PathSearchAlgorithm::BFS, // Default to BFS if not specified or invalid
    };

    let flow_graph = flow_graph.lock().unwrap();
    let network_flow = NetworkFlow::new((*flow_graph).clone());

    for max_distance in max_distances {
        let (flow, transfers) = network_flow.compute_flow(
            algorithm.clone(),
            &from_address,
            &to_address,
            parsed_value_param,
            max_distance,
            None,  // max_transfers
            path_search_algorithm,
            None
        );
        println!("Computed flow with max distance {max_distance:?}: {flow}");
        socket.write_all(
            chunked_response(
                &(jsonrpc_result(
                    request.id.clone(),
                    json::object! {
                        maxFlowValue: flow.to_decimal(),
                        final: max_distance.is_none(),
                        transferSteps: transfers.into_iter().map(|e| json::object! {
                            from: e.from.to_checksummed_hex(),
                            to: e.to.to_checksummed_hex(),
                            token_owner: e.token.to_checksummed_hex(),
                            value: e.capacity.to_decimal(),
                        }).collect::<Vec<_>>(),
                    },
                ) + "\r\n"),
            )
            .as_bytes(),
        )?;
    }
    socket.write_all(chunked_close().as_bytes())?;
    Ok(())
}

type FlowComputationFn = fn(&Address, &Address, &EdgeDB, U256, Option<u64>, Option<u64>) -> (U256, Vec<Edge>);

fn generic_visualize_flow_graph_new<A: NetworkFlowAlgorithm + Clone>(
    edge_db: &EdgeDB,
    flow_graph: &Arc<Mutex<FlowGraph>>,
    params: &JsonValue,
    algorithm: A,
) -> Result<String, Box<dyn Error>> {
    let visualizer = GraphVisualizer::new(edge_db.clone());

    let from = Address::from(params["from"].as_str().unwrap());
    let to = Address::from(params["to"].to_string().as_str());
    let value = U256::from(params["value"].as_str().unwrap_or("0"));

    let path_search_algorithm = match params["path_search_algorithm"].as_str().unwrap_or("BFS") {
        "BFS" => PathSearchAlgorithm::BFS,
        "BiBFS" => PathSearchAlgorithm::BiBFS,
        _ => PathSearchAlgorithm::BFS, // Default to BFS if not specified or invalid
    };

    let flow_graph = flow_graph.lock().unwrap();
    let network_flow = NetworkFlow::new((*flow_graph).clone());
    let (_, flow) = network_flow.compute_flow(
        algorithm,
        &from,
        &to,
        value,
        None,  // max_distance
        None,   // max_transfers
        path_search_algorithm,
        None
    );
    
    Ok(visualizer.generate_flow_graph(&flow))
}

fn generic_visualize_flow_graph(
    edge_db: &Arc<EdgeDB>,
    params: &JsonValue,
    compute_fn: FlowComputationFn,
) -> Result<String, Box<dyn Error>> {
    let visualizer = GraphVisualizer::new((**edge_db).clone());

    let from = Address::from(params["from"].as_str().unwrap());
    let to = Address::from(params["to"].as_str().unwrap());
    let value = U256::from(params["value"].as_str().unwrap_or("0"));

    let (_, flow) = compute_fn(&from, &to, edge_db, value, None, None);
    Ok(visualizer.generate_flow_graph(&flow))
}

fn compute_transfer_with_visualization<A: NetworkFlowAlgorithm + Clone>(
    request: &JsonRpcRequest,
    _edge_db: &EdgeDB,
    flow_graph: &Arc<Mutex<FlowGraph>>,
    mut socket: TcpStream,
    algorithm: A,
) -> Result<(), Box<dyn Error>> {
    socket.write_all(chunked_header().as_bytes())?;

    let parsed_value_param = match request.params["value"].as_str() {
        Some(value_str) => validate_and_parse_u256(value_str)?,
        None => U256::MAX,
    };

    let from_address = validate_and_parse_ethereum_address(&request.params["from"].to_string())?;
    let to_address = validate_and_parse_ethereum_address(&request.params["to"].to_string())?;

    let path_search_algorithm = match request.params["path_search_algorithm"].as_str().unwrap_or("BFS") {
        "BFS" => PathSearchAlgorithm::BFS,
        "BiBFS" => PathSearchAlgorithm::BiBFS,
        _ => PathSearchAlgorithm::BFS,
    };

    let flow_graph = flow_graph.lock().unwrap();
    let network_flow = NetworkFlow::new((*flow_graph).clone());

    let mut recorder = FlowRecorder::new();

    let (flow, transfers) = network_flow.compute_flow(
        algorithm,
        &from_address,
        &to_address,
        parsed_value_param,
        None,  // max_distance
        None,  // max_transfers
        path_search_algorithm,
        Some(&mut recorder)
    );

    println!("Computed flow: {flow}");

    // Generate visualization
    let visualization_path = format!("flow_visualization_{}.gif", chrono::Utc::now().timestamp());
    recorder.generate_visualization(&visualization_path, from_address, to_address)?;

    socket.write_all(
        chunked_response(
            &(jsonrpc_result(
                request.id.clone(),
                json::object! {
                    maxFlowValue: flow.to_decimal(),
                    transferSteps: transfers.into_iter().map(|e| json::object! {
                        from: e.from.to_checksummed_hex(),
                        to: e.to.to_checksummed_hex(),
                        token_owner: e.token.to_checksummed_hex(),
                        value: e.capacity.to_decimal(),
                    }).collect::<Vec<_>>(),
                    visualizationPath: visualization_path,
                },
            ) + "\r\n"),
        )
        .as_bytes(),
    )?;

    socket.write_all(chunked_close().as_bytes())?;
    Ok(())
}
