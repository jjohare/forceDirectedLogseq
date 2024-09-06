// Import necessary modules
use warp::Filter;
use std::path::Path;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use warp::http::StatusCode;
use serde_json::{json, Value};
use base64::engine::general_purpose;
use regex::Regex;
use serde::{Serialize, Deserialize};
use wgpu::{
    Backends, Buffer, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    Device, DeviceDescriptor, Features, Instance, Limits, Queue, ShaderModuleDescriptor, ShaderSource,
};
use warp::ws::{WebSocket, Message};
use futures::{StreamExt, SinkExt};
use octocrab::Octocrab;
use anyhow::{Result, Context};
use dotenv::dotenv;
use std::env;
use log::{error, info};
use tokio::sync::mpsc;

// Module imports (place your module imports here, if applicable)
// Example:
// mod graph_simulation;
// mod github_integration;
// mod websocket_handler;
// mod file_metadata;
// mod utils;

// Define constants for file paths used in data storage
const MARKDOWN_STORAGE_PATH: &str = "/usr/src/app/data/processed_files/markdown";
const GRAPH_DATA_PATH: &str = "/usr/src/app/data/processed_files/graph-data.json";

// Simulation parameters for the graph algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub max_speed: f32,
    pub damping: f32,
    pub centering_force: f32,
    pub edge_distance: f32,
}

// Provide default values for the simulation parameters
impl Default for SimulationParams {
    fn default() -> Self {
        SimulationParams {
            repulsion_strength: 100.0,     // Default repulsion force between nodes
            attraction_strength: 0.05,     // Default attraction force between connected nodes
            max_speed: 5.0,                // Max speed nodes can move
            damping: 0.95,                 // How much velocity is reduced each frame
            centering_force: 0.01,         // Keeps nodes centered
            edge_distance: 50.0,           // Default distance for edges
        }
    }
}

// File metadata structure, representing information about markdown files from GitHub
#[derive(Debug, Clone, Serialize)]
pub struct FileMetadata {
    pub name: String,  // File name
    pub path: String,  // File path in the repository
    pub sha: String,   // SHA1 hash of the file content
    pub size: usize,   // Size of the file
    pub content: Option<String>, // File content (optional, only fetched when needed)
}

// === Data Management ===

/// Load the graph data from the specified file path.
pub async fn load_graph_data() -> Result<Value, Box<dyn Error>> {
    // Read the graph data from the JSON file at GRAPH_DATA_PATH
    let data = tokio::fs::read_to_string(GRAPH_DATA_PATH)
        .await
        .context("Failed to read graph data from file")?;
    
    // Parse the JSON data
    let graph_data: Value = serde_json::from_str(&data)
        .context("Failed to parse graph data")?;
    
    Ok(graph_data)
}

/// Save the updated graph data to the specified file path.
pub async fn save_graph_data(graph_data: &Value) -> Result<(), Box<dyn Error>> {
    // Serialize the graph data to a pretty JSON format
    let data = serde_json::to_string_pretty(graph_data)
        .context("Failed to serialize graph data to JSON")?;
    
    // Write the serialized data to the file at GRAPH_DATA_PATH
    tokio::fs::write(GRAPH_DATA_PATH, data)
        .await
        .context("Failed to write graph data to file")?;
    
    Ok(())
}

/// Build edges between graph nodes based on updated files.
///
/// This function updates the graph data with new nodes and edges based on the content
/// of the updated markdown files. It parses the content for references to other nodes
/// and adjusts edge weights accordingly.
pub async fn build_edges(graph_data: &mut Value, updated_files: &[FileMetadata]) -> Result<(), Box<dyn Error>> {
    // Extract nodes and edges from the graph data
    let nodes = graph_data["nodes"]
        .as_array_mut()
        .ok_or("Invalid graph data: 'nodes' not found")?;
    let node_names: Vec<String> = nodes.iter()
        .map(|node| node["name"].as_str().unwrap().to_string())
        .collect();
    
    let edges = graph_data["edges"]
        .as_array_mut()
        .ok_or("Invalid graph data: 'edges' not found")?;

    // Process each updated file
    for file in updated_files {
        let source = file.name.replace(".md", "");
        let content = file.content.as_ref().unwrap(); // File content must be present

        // Update or add the source node
        update_or_add_node(nodes, &source, content);

        // Update edges based on references in the file content
        update_edges(edges, &source, content, &node_names);
    }

    // Clean up any invalid edges (i.e., edges pointing to non-existent nodes)
    remove_invalid_edges(nodes, edges);

    Ok(())
}

/// Update an existing node or add a new node based on file content.
fn update_or_add_node(nodes: &mut Vec<Value>, source: &str, content: &str) {
    if let Some(node) = nodes.iter_mut().find(|n| n["name"] == source) {
        // Update the node's size and hyperlink count if it already exists
        node["size"] = json!(content.len());
        node["httpsLinksCount"] = json!(count_https_links(content));
    } else {
        // Add a new node if it doesn't exist
        nodes.push(json!({
            "name": source,
            "size": content.len(),
            "httpsLinksCount": count_https_links(content),
        }));
    }
}

/// Update edges between nodes based on file references.
fn update_edges(edges: &mut Vec<Value>, source: &str, content: &str, node_names: &[String]) {
    let references = extract_references(content, node_names);
    for (target, weight) in references {
        if target != source {
            if let Some(edge) = edges.iter_mut().find(|e| e["source"] == source && e["target"] == target) {
                // If an edge already exists, increase its weight
                edge["weight"] = json!(edge["weight"].as_f64().unwrap() + weight);
            } else {
                // Otherwise, add a new edge
                edges.push(json!({
                    "source": source,
                    "target": target,
                    "weight": weight,
                }));
            }
        }
    }
}

/// Remove edges that point to non-existent nodes.
fn remove_invalid_edges(nodes: &[Value], edges: &mut Vec<Value>) {
    edges.retain(|edge| {
        nodes.iter().any(|n| n["name"] == edge["source"])
            && nodes.iter().any(|n| n["name"] == edge["target"])
    });
}

/// Count the number of HTTPS links in the file content.
fn count_https_links(content: &str) -> usize {
    let re = Regex::new(r"https?://[^\s]+").unwrap();
    re.find_iter(content).count()
}

/// Extract references to other nodes from the file content.
fn extract_references(content: &str, node_names: &[String]) -> Vec<(String, f64)> {
    let mut references = Vec::new();
    for name in node_names {
        let escaped_name = regex::escape(&name.replace(".md", ""));
        let re = Regex::new(&format!(r"\b{}\b", escaped_name)).unwrap();
        let mut count = 0.0;
        for cap in re.captures_iter(content) {
            let start = cap.get(0).unwrap().start();
            let end = cap.get(0).unwrap().end();
            let surrounding_text = &content[start.saturating_sub(50)..end.saturating_add(50)];
            if surrounding_text.contains("](http") || surrounding_text.contains("](https") {
                count += 0.1; // Weigh hyperlink references less
            } else {
                count += 1.0; // Direct references have full weight
            }
        }
        if count > 0.0 {
            references.push((name.replace(".md", ""), count));
        }
    }
    references
}


// === GitHub Management ===

/// Structure to hold file metadata information fetched from GitHub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub name: String,
    pub path: String,
    pub sha: String,
    pub size: usize,
    pub content: Option<String>,
}

/// Fetches the metadata of markdown files from a GitHub repository directory.
///
/// This function connects to the GitHub API using the Octocrab client and retrieves the metadata
/// for markdown files within the specified directory of the repository.
pub async fn fetch_markdown_metadata(
    owner: &str,
    repo: &str,
    directory: &str,
) -> Result<Vec<FileMetadata>> {
    // Initialize Octocrab client to communicate with GitHub API
    let octocrab = Octocrab::builder().build()?;
    
    // Fetch the content of the repository directory
    let contents = octocrab
        .repos(owner, repo)
        .get_content()
        .path(directory)
        .send()
        .await
        .context("Failed to fetch repository contents")?;

    // Filter the fetched content to include only markdown files and return them as metadata
    let files = contents
        .items
        .into_iter()
        .filter(|item| item.name.ends_with(".md") && item.r#type == "file")
        .map(|item| FileMetadata {
            name: item.name,
            path: item.path,
            sha: item.sha,
            size: item.size as usize,
            content: None, // Content will be fetched later if needed
        })
        .collect::<Vec<_>>();

    Ok(files)
}

/// Compares and updates the local files with the GitHub files based on their SHA hashes.
///
/// This function checks the local files and compares them with the GitHub metadata. Files that
/// are outdated or missing locally are updated if marked `public:: true`. Obsolete files are removed.
pub async fn compare_and_update_files(
    github_files: Vec<FileMetadata>,
    local_directory: &Path,
    owner: &str,
    repo: &str,
) -> Result<Vec<FileMetadata>> {
    let mut files_to_update = Vec::new();
    let octocrab = Octocrab::builder().build()?;

    // Loop through each GitHub file
    for file in github_files {
        let local_path = local_directory.join(&file.name);
        let metadata_path = local_path.with_extension("md.meta.json");
        
        // Check if the local file needs updating
        let should_update = if metadata_path.exists() {
            let metadata = fs::read_to_string(&metadata_path)
                .await
                .context(format!("Failed to read metadata file: {}", metadata_path.display()))?;
            let metadata: serde_json::Value = serde_json::from_str(&metadata)
                .context("Failed to parse metadata JSON")?;
            metadata["sha"] != file.sha // Check if the SHA has changed
        } else {
            true // File is missing locally
        };

        if should_update {
            // Fetch the file content from GitHub
            let content_response = octocrab
                .repos(owner, repo)
                .get_content()
                .path(&file.path)
                .send()
                .await
                .context(format!("Failed to fetch content for file: {}", file.path))?;

            if let octocrab::models::repos::ContentItems::File { content: Some(content), .. } = content_response {
                let decoded_content = general_purpose::STANDARD
                    .decode(content.replace('\n', ""))
                    .context("Failed to decode base64 content")?;
                let content_str = String::from_utf8(decoded_content)
                    .context("Failed to convert content to UTF-8")?;

                // Only update the file if it is marked as `public:: true`
                if content_str.lines().next().map_or(false, |line| line.trim() == "public:: true") {
                    // Update the local file with the new content
                    fs::write(&local_path, &content_str)
                        .await
                        .context(format!("Failed to write file: {}", local_path.display()))?;

                    // Extract hyperlinks from the content for metadata
                    let metadata = serde_json::json!({
                        "sha": file.sha,
                        "hyperlinks": extract_hyperlinks(&content_str),
                    });

                    // Write metadata to a separate file
                    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)
                        .await
                        .context(format!("Failed to write metadata: {}", metadata_path.display()))?;

                    files_to_update.push(FileMetadata {
                        name: file.name,
                        path: file.path,
                        sha: file.sha,
                        size: file.size,
                        content: Some(content_str),
                    });

                    println!("Updated file and metadata: {}", file.path);
                } else {
                    println!("File {} is not public, skipping update", file.path);
                }
            }
        } else {
            println!("File {} is up to date, no changes needed", file.path);
        }
    }

    // Remove obsolete local files that no longer exist in the GitHub repository
    remove_obsolete_files(local_directory, &github_files).await?;

    Ok(files_to_update)
}

/// Extracts hyperlinks from the given file content.
fn extract_hyperlinks(content: &str) -> Vec<String> {
    let re = Regex::new(r"https?://[^\s)]+").unwrap();
    re.find_iter(content)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Removes local files that no longer exist in the GitHub repository.
///
/// This function compares the local directory's files with the GitHub repository files and
/// deletes any local files that are no longer present on GitHub.
async fn remove_obsolete_files(local_directory: &Path, github_files: &[FileMetadata]) -> Result<()> {
    let mut entries = fs::read_dir(local_directory).await?;
    let github_file_names: std::collections::HashSet<_> = github_files.iter().map(|f| f.name.clone()).collect();

    while let Some(entry) = entries.next_entry().await? {
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        // Check if the local file is a markdown file and not present in the GitHub repo
        if file_name_str.ends_with(".md") && !github_file_names.contains(&file_name_str.to_string()) {
            fs::remove_file(entry.path())
                .await
                .context(format!("Failed to remove obsolete file: {}", file_name_str))?;

            // Also remove the corresponding metadata file if it exists
            let metadata_path = entry.path().with_extension("md.meta.json");
            if metadata_path.exists() {
                fs::remove_file(&metadata_path)
                    .await
                    .context(format!("Failed to remove obsolete metadata file: {}", metadata_path.display()))?;
            }

            println!("Removed obsolete file and its metadata: {}", file_name_str);
        }
    }

    Ok(())
}

// === Graph Simulation ===

/// Structure representing the simulation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub max_speed: f32,
    pub damping: f32,
    pub centering_force: f32,
    pub edge_distance: f32,
}

impl Default for SimulationParams {
    fn default() -> Self {
        SimulationParams {
            repulsion_strength: 100.0,
            attraction_strength: 0.05,
            max_speed: 5.0,
            damping: 0.95,
            centering_force: 0.01,
            edge_distance: 50.0,
        }
    }
}

/// The main struct that handles the graph simulation using WebGPU.
pub struct ServerGraphSimulation {
    graph_data: Value,
    simulation_params: SimulationParams,
    is_simulation_running: bool,
    device: Device,
    queue: Queue,
    compute_pipeline: ComputePipeline,
    update_pipeline: ComputePipeline,
    buffers: HashMap<String, Buffer>,
    tx: mpsc::Sender<warp::ws::Message>,
}

impl ServerGraphSimulation {
    /// Initializes the graph simulation with initial graph data and a WebSocket sender.
    pub async fn new(graph_data: Value, tx: mpsc::Sender<warp::ws::Message>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable WebGPU adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Graph Simulation Device"),
                    features: Features::empty(),
                    limits: Limits::downlevel_webgl2_defaults(),
                },
                None,
            )
            .await?;

        // Load the shaders from the server/src/shaders directory
        let compute_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/compute_forces.wgsl").into()),
        });

        let update_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Update Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/update_positions.wgsl").into()),
        });

        // Create the compute pipeline
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "compute_forces", // The main function in the compute shader
        });

        // Create the update pipeline
        let update_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Update Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "update_positions", // The main function in the update shader
        });

        // Initialize buffers for positions, velocities, and edges
        let buffers = HashMap::new();
        let mut simulation = Self {
            graph_data,
            simulation_params: SimulationParams::default(),
            is_simulation_running: false,
            device,
            queue,
            compute_pipeline,
            update_pipeline,
            buffers,
            tx,
        };

        // Create necessary buffers
        simulation.create_buffers().await?;

        Ok(simulation)
    }

    /// Creates necessary buffers for positions, velocities, and forces.
    pub async fn create_buffers(&mut self) -> Result<()> {
        let nodes = self.graph_data["nodes"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid graph data: 'nodes' not found"))?;
        let edges = self.graph_data["edges"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid graph data: 'edges' not found"))?;

        let node_count = nodes.len() as u32;
        let edge_count = edges.len() as u32;

        // Create position buffer for nodes
        self.buffers.insert(
            "positions".to_string(),
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Positions Buffer"),
                contents: bytemuck::cast_slice(
                    &nodes.iter().flat_map(|node| {
                        let x = node.get("x").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                        let y = node.get("y").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                        [x, y]
                    }).collect::<Vec<f32>>(),
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            }),
        );

        // Create other buffers (velocities, edges, etc.)
        self.buffers.insert(
            "velocities".to_string(),
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Velocities Buffer"),
                size: node_count as u64 * 2 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        );

        self.buffers.insert(
            "forces".to_string(),
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Forces Buffer"),
                size: node_count as u64 * 2 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        );

        self.buffers.insert(
            "edges".to_string(),
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Edges Buffer"),
                contents: bytemuck::cast_slice(
                    &edges.iter().flat_map(|edge| {
                        let source_name = edge.get("source").and_then(Value::as_str).unwrap_or("");
                        let target_name = edge.get("target").and_then(Value::as_str).unwrap_or("");

                        let source_index = nodes
                            .iter()
                            .position(|node| node.get("name").and_then(Value::as_str).unwrap_or("") == source_name)
                            .unwrap_or(0) as u32;
                        let target_index = nodes
                            .iter()
                            .position(|node| node.get("name").and_then(Value::as_str).unwrap_or("") == target_name)
                            .unwrap_or(0) as u32;

                        [source_index, target_index]
                    }).collect::<Vec<u32>>(),
                ),
                usage: BufferUsages::STORAGE,
            }),
        );

        Ok(())
    }

    /// Starts the graph simulation at a specified interval (milliseconds).
    pub fn start_simulation(&mut self, interval_ms: u64) {
        if self.is_simulation_running {
            return;
        }
        self.is_simulation_running = true;

        let simulation = Arc::new(Mutex::new(self));
        let simulation_clone = Arc::clone(&simulation);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));
            loop {
                interval.tick().await;
                let mut simulation = simulation_clone.lock().unwrap();

                if simulation.is_simulation_running {
                    let delta_time = interval_ms as f32 / 1000.0;
                    simulation.simulate(delta_time).await;

                    if simulation.is_simulation_running {
                        simulation.broadcast_positions().await;
                    }
                } else {
                    break;
                }
            }
        });
    }

    /// Stops the ongoing simulation.
    pub fn stop_simulation(&mut self) {
        self.is_simulation_running = false;
    }

    /// Simulates the forces and updates positions for one time step.
    async fn simulate(&mut self, delta_time: f32) {
        // Update simulation parameters
        self.queue.write_buffer(
            &self.buffers["compute_params"],
            0,
            bytemuck::cast_slice(&[
                self.simulation_params.repulsion_strength,
                self.simulation_params.attraction_strength,
                self.simulation_params.centering_force,
                self.simulation_params.edge_distance,
            ]),
        );

        self.queue.write_buffer(
            &self.buffers["update_params"],
            0,
            bytemuck::cast_slice(&[
                delta_time,
                self.simulation_params.damping,
                self.simulation_params.max_speed,
            ]),
        );

        // Create the command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Simulation Encoder"),
        });

        {
            // Perform the compute pass to apply forces
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.dispatch_workgroups(256, 1, 1);
        }

        {
            // Perform the update pass to update positions
            let mut update_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Update Pass"),
            });
            update_pass.set_pipeline(&self.update_pipeline);
            update_pass.dispatch_workgroups(256, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Broadcasts the updated positions of nodes to connected WebSocket clients.
    async fn broadcast_positions(&mut self) {
        let buffer_slice = self.buffers["positions"].slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.await {
            let data = buffer_slice.get_mapped_range();
            let positions: &[[f32; 2]] = bytemuck::cast_slice(&data);

            // Update the graph data with the new positions
            if let Some(nodes) = self.graph_data.get_mut("nodes").and_then(Value::as_array_mut) {
                for (i, pos) in positions.iter().enumerate() {
                    if let Some(node) = nodes.get_mut(i) {
                        node["x"] = json!(pos[0] as f64);
                        node["y"] = json!(pos[1] as f64);
                    }
                }
            }

            // Send the updated positions via WebSocket
            let message = json!({
                "type": "nodePositions",
                "positions": json!(positions),
            });

            let message_str = serde_json::to_string(&message).unwrap();
            if let Err(e) = self.tx.send(warp::ws::Message::text(message_str)).await {
                eprintln!("Error sending WebSocket message: {}", e);
            }
        }

        self.buffers["positions"].unmap();
    }

    /// Updates the graph data and resets the simulation buffers.
    pub async fn update_graph_data(&mut self, new_graph_data: Value) {
        self.graph_data = new_graph_data;
        self.create_buffers().await.unwrap();
    }

    /// Updates the simulation parameters.
    pub fn set_simulation_parameters(&mut self, params: SimulationParams) {
        self.simulation_params = params;
    }

    /// Returns the current simulation parameters.
    pub fn get_simulation_parameters(&self) -> SimulationParams {
        self.simulation_params.clone()
    }

    /// Returns the current graph data.
    pub fn get_graph_data(&self) -> &Value {
        &self.graph_data
    }
}


// === Ragflow Integration ===

/// Sends a request to create a new conversation with the RAGFlow API.
/// Returns the conversation ID.
pub async fn create_conversation(user_id: &str) -> Result<String> {
    let client = Client::new();
    let ragflow_base_url = env::var("RAGFLOW_BASE_URL")?;
    let ragflow_api_key = env::var("RAGFLOW_API_KEY")?;

    // Make the request to create a new conversation
    let response = client
        .get(format!("{}/api/new_conversation", ragflow_base_url))
        .header("Authorization", format!("Bearer {}", ragflow_api_key))
        .query(&[("user_id", user_id)])
        .send()
        .await?;

    let response_json: serde_json::Value = response.json().await?;
    let conversation_id = response_json["data"]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Conversation ID not found"))?;

    Ok(conversation_id.to_string())
}

/// Sends a message to the active conversation in RAGFlow and receives the response.
/// The `conversation_id` identifies the active session and `message` is the user's input.
pub async fn send_message(conversation_id: &str, message: &str) -> Result<serde_json::Value> {
    let client = Client::new();
    let ragflow_base_url = env::var("RAGFLOW_BASE_URL")?;
    let ragflow_api_key = env::var("RAGFLOW_API_KEY")?;

    // Make the request to send the message
    let response = client
        .post(format!("{}/api/completion", ragflow_base_url))
        .header("Authorization", format!("Bearer {}", ragflow_api_key))
        .json(&json!({
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "content": message}],
            "stream": false
        }))
        .send()
        .await?;

    // Parse and return the result
    let result: serde_json::Value = response.json().await?;
    Ok(result)
}

/// Handles the initialization of a conversation and sends a message, including optional voice output.
/// If TTS is enabled, it sends the response back to the client as speech.
pub async fn handle_chat_message(
    conversation_id: &str,
    message: &str,
    tts_enabled: bool,
) -> Result<serde_json::Value> {
    let ragflow_response = send_message(conversation_id, message).await?;

    // If TTS is enabled, we could either trigger client-side TTS or process it server-side.
    if tts_enabled {
        // Future expansion: Send back text-to-speech audio, or instruct the client to render speech
        // For now, just log this as a placeholder.
        println!("Text-to-Speech processing could be triggered here");
    }

    Ok(ragflow_response)
}


/// Handles incoming WebSocket connections and forwards data to the graph simulation.
pub async fn handle_ws(
    ws: WebSocket,
    graph_simulation: Arc<Mutex<ServerGraphSimulation>>,
    mut rx: mpsc::Receiver<Message>,
) {
    let (mut ws_tx, mut ws_rx) = ws.split();

    // Forward messages from rx to the WebSocket
    let ws_tx_clone = ws_tx;
    tokio::task::spawn(async move {
        while let Some(message) = rx.recv().await {
            if let Err(e) = ws_tx_clone.send(message).await {
                eprintln!("Error sending WebSocket message: {}", e);
                break;
            }
        }
    });

    // Handle incoming WebSocket messages
    while let Some(result) = ws_rx.next().await {
        match result {
            Ok(message) => {
                if let Ok(text) = message.to_str() {
                    if let Err(e) = handle_message(text, &graph_simulation, &mut ws_tx).await {
                        eprintln!("Error handling message: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
        }
    }

    eprintln!("WebSocket connection closed");
}

/// Handles WebSocket messages and optionally integrates TTS for responses.
async fn handle_message(
    message: &str,
    graph_simulation: &Arc<Mutex<ServerGraphSimulation>>,
    ws_tx: &mut futures::stream::SplitSink<WebSocket, Message>,
) -> Result<()> {
    // Parse the message to check for commands like startSimulation, stopSimulation, etc.
    if let Ok(command) = serde_json::from_str::<Value>(message) {
        match command["type"].as_str() {
            Some("startSimulation") => {
                let mut simulation = graph_simulation.lock().unwrap();
                simulation.start_simulation(16); // 60 FPS
                println!("Simulation started");
            }
            Some("stopSimulation") => {
                let mut simulation = graph_simulation.lock().unwrap();
                simulation.stop_simulation();
                println!("Simulation stopped");
            }
            Some("updateParameters") => {
                let mut simulation = graph_simulation.lock().unwrap();
                if let Ok(params) = serde_json::from_value(command["params"].clone()) {
                    simulation.set_simulation_parameters(params);
                    println!("Simulation parameters updated");

                    // Send updated parameters back to the client
                    let message = json!({
                        "type": "parametersUpdated",
                        "params": json!(simulation.get_simulation_parameters()), 
                    });
                    let message_str = serde_json::to_string(&message)?;
                    ws_tx.send(Message::text(message_str)).await?;
                }
            }
            Some("chatMessage") => {
                let conversation_id = command["conversationId"].as_str().unwrap_or("");
                let user_message = command["message"].as_str().unwrap_or("");

                // Optionally enable TTS based on command input
                let tts_enabled = command["ttsEnabled"].as_bool().unwrap_or(false);

                let ragflow_response = handle_chat_message(conversation_id, user_message, tts_enabled).await?;

                // If TTS is enabled, we return an audio file, otherwise a text response
                if tts_enabled {
                    if let Some(audio_file_path) = generate_tts_audio(&ragflow_response["content"].as_str().unwrap()).await? {
                        // Read the audio file and send it as base64 over WebSocket
                        let mut file = File::open(&audio_file_path).await?;
                        let mut audio_data = Vec::new();
                        file.read_to_end(&mut audio_data).await?;

                        let message = json!({
                            "type": "ttsAudio",
                            "audio": encode(&audio_data)
                        });

                        let message_str = serde_json::to_string(&message)?;
                        ws_tx.send(Message::text(message_str)).await?;

                        println!("Sent TTS audio over WebSocket");
                    } else {
                        // If TTS generation fails, fall back to text response
                        send_fallback_text(ws_tx, &ragflow_response).await?;
                    }
                } else {
                    send_fallback_text(ws_tx, &ragflow_response).await?;
                }
            }
            _ => eprintln!("Unknown command: {}", message),
        }
    }
    Ok(())
}

/// Sends a fallback text message if TTS is not enabled or fails.
async fn send_fallback_text(ws_tx: &mut futures::stream::SplitSink<WebSocket, Message>, ragflow_response: &Value) -> Result<()> {
    let message = json!({
        "type": "textMessage",
        "content": ragflow_response["content"],
    });

    let message_str = serde_json::to_string(&message)?;
    ws_tx.send(Message::text(message_str)).await?;

    println!("Sent fallback text message over WebSocket");
    Ok(())
}

/// Generates TTS audio using an external service (e.g., Google Cloud, Amazon Polly).
/// Returns the path to the generated audio file.
async fn generate_tts_audio(text: &str) -> Result<Option<String>> {
    let tts_enabled = env::var("TTS_ENABLED").unwrap_or_else(|_| "false".to_string()) == "true";
    if !tts_enabled {
        return Ok(None);
    }

    // Replace this with actual TTS API integration (Google Cloud, Amazon Polly, etc.)
    let tts_api_key = env::var("TTS_API_KEY").ok();
    if tts_api_key.is_none() {
        eprintln!("TTS API key not found, skipping TTS generation");
        return Ok(None);
    }

    // Example: Placeholder for actual TTS generation logic
    let audio_file_path = format!("/tmp/tts_output_{}.mp3", uuid::Uuid::new_v4());
    let mut file = File::create(&Path::new(&audio_file_path)).await?;
    file.write_all(b"Fake audio content").await?;

    println!("Generated TTS audio file at: {}", audio_file_path);
    Ok(Some(audio_file_path))
}

// === File Serving (for the client) ===

/// Serves static files from the `public` directory (HTML, CSS, JS).
pub fn static_files() -> impl Filter<Extract = (File,), Error = Rejection> + Clone {
    warp::fs::dir("./client/public")
}

// === WebSocket & Simulation Handling ===

pub async fn handle_ws(
    ws: WebSocket,
    graph_simulation: Arc<Mutex<ServerGraphSimulation>>,
    mut rx: mpsc::Receiver<Message>,
) {
    let (mut ws_tx, mut ws_rx) = ws.split();

    // Forward messages from rx to the WebSocket
    let ws_tx_clone = ws_tx;
    tokio::task::spawn(async move {
        while let Some(message) = rx.recv().await {
            if let Err(e) = ws_tx_clone.send(message).await {
                eprintln!("Error sending WebSocket message: {}", e);
                break;
            }
        }
    });

    // Handle incoming WebSocket messages
    while let Some(result) = ws_rx.next().await {
        match result {
            Ok(message) => {
                if let Ok(text) = message.to_str() {
                    if let Err(e) = handle_ws_message(text, &graph_simulation, &mut ws_tx).await {
                        eprintln!("Error handling message: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
        }
    }

    eprintln!("WebSocket connection closed");
}

async fn handle_ws_message(
    message: &str,
    graph_simulation: &Arc<Mutex<ServerGraphSimulation>>,
    ws_tx: &mut futures::stream::SplitSink<WebSocket, Message>,
) -> Result<()> {
    // Handle WebSocket message commands (startSimulation, stopSimulation, chatMessage, etc.)
    let command: Value = serde_json::from_str(message)?;
    match command["type"].as_str() {
        Some("startSimulation") => {
            let mut simulation = graph_simulation.lock().unwrap();
            simulation.start_simulation(16); // 60 FPS
            println!("Simulation started");
        }
        Some("stopSimulation") => {
            let mut simulation = graph_simulation.lock().unwrap();
            simulation.stop_simulation();
            println!("Simulation stopped");
        }
        Some("chatMessage") => {
            let conversation_id = command["conversationId"].as_str().unwrap_or("");
            let user_message = command["message"].as_str().unwrap_or("");
            let tts_enabled = command["ttsEnabled"].as_bool().unwrap_or(false);

            // Handle chat and TTS via the RAGFlow API
            let response = handle_chat_message(conversation_id, user_message, tts_enabled).await?;
            send_response(ws_tx, response, tts_enabled).await?;
        }
        _ => {
            eprintln!("Unknown command received: {}", message);
        }
    }
    Ok(())
}

/// Sends either a TTS audio response or text fallback based on TTS availability.
async fn send_response(
    ws_tx: &mut futures::stream::SplitSink<WebSocket, Message>,
    response: Value,
    tts_enabled: bool,
) -> Result<()> {
    if tts_enabled {
        if let Some(audio_file_path) = generate_tts_audio(&response["content"].as_str().unwrap()).await? {
            let mut file = TokioFile::open(&audio_file_path).await?;
            let mut audio_data = Vec::new();
            file.read_to_end(&mut audio_data).await?;

            let message = json!({
                "type": "ttsAudio",
                "audio": base64::encode(&audio_data),
            });

            let message_str = serde_json::to_string(&message)?;
            ws_tx.send(Message::text(message_str)).await?;
        } else {
            send_fallback_text(ws_tx, &response).await?;
        }
    } else {
        send_fallback_text(ws_tx, &response).await?;
    }
    Ok(())
}

// === RAGFlow Integration ===

async fn handle_chat_message(
    conversation_id: &str,
    message: &str,
    tts_enabled: bool,
) -> Result<Value> {
    // Send message to RAGFlow and get response
    let ragflow_response = send_message(conversation_id, message).await?;

    // Optionally handle TTS response generation here
    if tts_enabled {
        // Trigger TTS or process response as text
        println!("TTS enabled, processing...");
    }

    Ok(ragflow_response)
}

// === Server Setup ===

/// Starts the Warp server to serve files and handle WebSocket connections.
pub async fn start_server(
    graph_simulation: Arc<Mutex<ServerGraphSimulation>>,
    rx: mpsc::Receiver<Message>,
) {
    // Serve static files (client assets)
    let static_files_route = warp::path("static").and(static_files());

    // WebSocket route
    let graph_sim = graph_simulation.clone();
    let ws_route = warp::path("ws")
        .and(warp::ws())
        .and(warp::any().map(move || graph_sim.clone()))
        .map(|ws: warp::ws::Ws, graph_sim: Arc<Mutex<ServerGraphSimulation>>| {
            ws.on_upgrade(move |socket| handle_ws(socket, graph_sim, rx.clone()))
        });

    // Combine the routes
    let routes = static_files_route.or(ws_route).recover(handle_rejection);

    // Start the Warp server
    info!("Starting Warp server at 0.0.0.0:8443");
    warp::serve(routes)
        .tls()
        .cert_path("cert.pem")
        .key_path("key.pem")
        .run(([0, 0, 0, 0], 8443))
        .await;
}

/// Error handling for rejections.
async fn handle_rejection(err: warp::Rejection) -> Result<impl Reply, warp::Rejection> {
    let code;
    let message;

    if err.is_not_found() {
        code = StatusCode::NOT_FOUND;
        message = "Not Found";
    } else {
        code = StatusCode::INTERNAL_SERVER_ERROR;
        message = "Internal Server Error";
    }

    Ok(warp::reply::with_status(message, code))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Initialize environment variables
    dotenv().ok();  // Load from .env file
    env_logger::init();  // Initialize logging

    // Retrieve environment variables
    let github_owner = env::var("GITHUB_OWNER").expect("GITHUB_OWNER not set");
    let github_repo = env::var("GITHUB_REPO").expect("GITHUB_REPO not set");
    let github_directory = env::var("GITHUB_DIRECTORY").expect("GITHUB_DIRECTORY not set");

    info!("Environment variables loaded successfully");

    // Step 2: Load initial graph data from file
    let initial_graph_data = match load_graph_data().await {
        Ok(data) => {
            info!("Graph data loaded successfully");
            data
        },
        Err(e) => {
            error!("Failed to load graph data: {:?}", e);
            return Err(e.into());
        }
    };
    
    // Step 3: Set up message channel for WebSocket communication
    let (tx, rx) = mpsc::channel(32);
    info!("WebSocket message channel set up");

    // Step 4: Initialize the graph simulation with the initial data
    let graph_simulation = Arc::new(Mutex::new(
        ServerGraphSimulation::new(initial_graph_data.clone(), tx.clone()).await?
    ));
    info!("Graph simulation initialized");

    // Step 5: Start a background task to refresh GitHub data periodically
    let graph_simulation_clone = graph_simulation.clone();
    tokio::spawn(async move {
        info!("Started background task for GitHub data refresh");
        if let Err(e) = refresh_graph_data_task(graph_simulation_clone, github_owner.clone(), github_repo.clone(), github_directory.clone()).await {
            eprintln!("Error in graph data refresh task: {}", e);
        }
    });

    // Step 6: Start the Warp server (serving static files and WebSocket)
    info!("Starting Warp server on 0.0.0.0:8443");
    start_server(graph_simulation, rx).await;

    Ok(())
}

/// Periodic task to refresh graph data from GitHub.
async fn refresh_graph_data_task(
    graph_simulation: Arc<Mutex<ServerGraphSimulation>>,
    github_owner: String,
    github_repo: String,
    github_directory: String,
) -> Result<()> {
    loop {
        info!("Refreshing graph data...");

        // Step 1: Fetch markdown metadata from GitHub
        let github_files = fetch_markdown_metadata(&github_owner, &github_repo, &github_directory).await?;
    
        // Step 2: Compare and update files if necessary
        let files_to_update = github_integration::compare_and_update_files(
            github_files, 
            std::path::Path::new(MARKDOWN_STORAGE_PATH), 
            &github_owner, 
            &github_repo,
        ).await?;

        // Step 3: If updates are found, rebuild the edges in the graph data
        if !files_to_update.is_empty() {
            let mut graph_data = load_graph_data().await?;
            build_edges(&mut graph_data, &files_to_update).await?;

            // Save the updated graph data
            save_graph_data(&graph_data).await?;

            // Update the graph simulation with the new data
            let mut simulation = graph_simulation.lock().unwrap();
            simulation.update_graph_data(graph_data).await;
        } else {
            info!("No updates needed for graph data.");
        }

        // Step 4: Sleep for 60 seconds before the next refresh
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    }
}

