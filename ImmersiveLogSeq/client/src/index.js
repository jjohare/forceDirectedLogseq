// Import necessary Three.js modules
import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';

// HID SpaceMouse support
import HID from 'hid';
import { SpaceMouse } from './spacemouse.js';

// Global variables for Three.js components
let renderer, scene, camera, controls, spaceMouse;

// Graph data: nodes and edges
let nodes = [], edges = [];

// WebSocket connection for real-time graph updates
let socket;

// Performance optimization: Use object pooling for nodes and edges
const nodePool = [], edgePool = [];

// Time tracking for animation
let lastTime = 0;

// Font for node labels
let font;

// Constants for graph visualization
const NODE_BASE_SIZE = 5;
const MAX_FILE_SIZE = 1000000;
const MAX_HYPERLINK_COUNT = 2000;
const MAX_EDGE_WEIGHT = 100;
const TEXT_VISIBILITY_THRESHOLD = 100;

// Chat window elements
const chatWindow = document.getElementById('chatWindow');
const questionInput = document.getElementById('questionInput');
const askButton = document.getElementById('askButton');
const smartPane = document.getElementById('smartPane');
const resizeHandle = document.getElementById('resizeHandle');

// Chat-related variables
let currentConversationId = null;

/**
 * Initializes the 3D scene, camera, renderer, and controls.
 */
function initScene() {
    updateStatus('Initializing Scene');

    try {
        // Create the WebGL renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Enable XR (VR/AR) support
        renderer.xr.enabled = true;
        document.body.appendChild(renderer.domElement);

        // Create the scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        // Set up camera
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, 200);

        // Orbit controls for camera manipulation
        controls = new OrbitControls(camera, renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 0);
        scene.add(ambientLight, directionalLight);

        // Handle window resizing
        window.addEventListener('resize', onWindowResize, false);

        console.log('Scene initialized');
    } catch (error) {
        console.error('Error initializing scene:', error);
        updateStatus('WebGL not supported or error initializing scene');
        throw new Error('WebGL not supported or error initializing scene');
    }
}

/**
 * Initializes HID SpaceMouse support.
 */
function initSpaceMouse() {
    try {
        // SpaceMouse setup
        spaceMouse = new SpaceMouse();
        spaceMouse.on('data', (data) => {
            // Handle 3D input from SpaceMouse, modifying camera accordingly
            camera.position.x += data.translation.x;
            camera.position.y += data.translation.y;
            camera.position.z += data.translation.z;
            camera.rotation.x += data.rotation.x;
            camera.rotation.y += data.rotation.y;
            camera.rotation.z += data.rotation.z;
        });

        console.log('SpaceMouse initialized');
    } catch (error) {
        console.error('Error initializing SpaceMouse:', error);
        updateStatus('SpaceMouse not detected or error initializing SpaceMouse');
    }
}

/**
 * Handles WebXR setup.
 */
async function setupXR() {
    updateStatus('Setting up XR');
    const button = VRButton.createButton(renderer);
    document.body.appendChild(button);
    updateStatus('VR button added');
}

/**
 * Loads initial graph data from the server and sets up WebSocket.
 */
async function loadData() {
    updateStatus('Loading graph data');
    try {
        const response = await fetch('/graph-data');
        const graphData = await response.json();

        updateGraphData(graphData);
        updateStatus('Graph data loaded');
        setupWebSocket();
    } catch (error) {
        console.error('Error loading graph data:', error);
        updateStatus('Error loading graph data');
    }
}

/**
 * Sets up WebSocket connection for real-time updates.
 */
function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}`);

    socket.onopen = () => {
        console.log('WebSocket connection established');
        updateStatus('WebSocket connected');
    };

    socket.onmessage = (event) => {
        const updatedGraphData = JSON.parse(event.data);
        updateGraphData(updatedGraphData);
        updateStatus('Graph data updated');
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('WebSocket error');
    };

    socket.onclose = () => {
        console.log('WebSocket connection closed');
        updateStatus('WebSocket disconnected');
    };
}

/**
 * Updates the graph data and re-creates graph objects.
 * @param {Object} graphData - The new graph data
 */
function updateGraphData(graphData) {
    if (!graphData || !graphData.nodes || !graphData.edges) {
        console.error('Invalid graph data received:', graphData);
        return;
    }

    nodes = graphData.nodes;
    edges = graphData.edges;

    console.log(`Updating graph with ${nodes.length} nodes and ${edges.length} edges`);
    updateGraphObjects();
}
/**
 * Updates 3D objects for nodes and edges in the scene.
 * 
 * This function iterates over the list of nodes and edges, updates their positions, colors, and 
 * other properties based on the current graph data. It makes use of object pooling for both nodes 
 * and edges to avoid unnecessary object creation, improving performance.
 */
function updateGraphObjects() {
    // Check if we have nodes and edges to update
    if (!nodes || !edges) {
        console.error('Graph data is missing nodes or edges');
        return;
    }

    // Check if font is loaded before proceeding with text labels
    if (!font) {
        console.error('Font not loaded. Skipping label generation');
    }

    // === Nodes Update ===
    nodes.forEach((node, index) => {
        let mesh = nodePool[index]; // Get a pooled mesh or create one if it doesn't exist

        // Create a new mesh if not available in the pool
        if (!mesh) {
            // Create a new icosahedron geometry for the node, which provides a spherical shape
            const geometry = new THREE.IcosahedronGeometry(NODE_BASE_SIZE, 1);

            // Create a Phong material for realistic lighting (better for nodes)
            const material = new THREE.MeshPhongMaterial();

            // Combine geometry and material into a mesh
            mesh = new THREE.Mesh(geometry, material);

            // Add the mesh to the pool for future use
            nodePool[index] = mesh;

            // Add the mesh to the scene so it's visible
            scene.add(mesh);

            // If the font is available, create a text label for the node
            if (font) {
                const textGeometry = new TextGeometry(node.name, {
                    font: font,
                    size: NODE_BASE_SIZE * 0.5, // Size of the label relative to the node size
                    height: 0.1,                // Flat text, extruded only slightly
                });

                // Create a basic material for the text label
                const textMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });

                // Create a mesh for the text and position it above the node
                const textMesh = new THREE.Mesh(textGeometry, textMaterial);
                textMesh.position.y = NODE_BASE_SIZE * 1.5; // Position the text slightly above the node
                textMesh.visible = false; // Initially hide the text label

                // Add the text mesh as a child of the node mesh
                mesh.add(textMesh);
            }
        }

        // Update node's position based on the latest graph data
        mesh.position.set(node.x, node.y, node.z);

        // Adjust the scale of the node based on its file size or other properties
        mesh.scale.setScalar(calculateNodeSize(node.size));

        // Update the node's color based on the number of hyperlinks it contains
        mesh.material.color.setHex(getNodeColor(node.httpsLinksCount));

        // Attach user-defined data to the node (for interaction or identification purposes)
        mesh.userData = {
            nodeId: node.name,
            name: node.name
        };

        // Make sure the node is visible in the scene
        mesh.visible = true;

        // Check if the node has a text label, and adjust its visibility based on camera distance
        const textMesh = mesh.children[0];
        if (textMesh) {
            const distanceToCamera = camera.position.distanceTo(mesh.position);
            textMesh.visible = distanceToCamera < TEXT_VISIBILITY_THRESHOLD;
        }
    });

    // Hide any extra nodes in the pool that are not used by the current graph data
    for (let i = nodes.length; i < nodePool.length; i++) {
        if (nodePool[i]) nodePool[i].visible = false;
    }

    // === Edges Update ===
    edges.forEach((edge, index) => {
        let line = edgePool[index]; // Get a pooled line or create one if it doesn't exist

        // Create a new line if not available in the pool
        if (!line) {
            // Create a basic material for the edge lines with some transparency
            const material = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.3 });

            // Create a new geometry for the edge
            const geometry = new THREE.BufferGeometry();

            // Create a line object with the geometry and material
            line = new THREE.Line(geometry, material);

            // Add the line to the pool for future use
            edgePool[index] = line;

            // Add the line to the scene so it's visible
            scene.add(line);
        }

        // Get the source and target nodes for the edge from the graph data
        const sourceIndex = nodes.findIndex(n => n.name === edge.source);
        const targetIndex = nodes.findIndex(n => n.name === edge.target);

        // Update the edge if both source and target nodes are found
        if (sourceIndex !== -1 && targetIndex !== -1) {
            // Get positions of source and target nodes
            const sourcePos = new THREE.Vector3(nodes[sourceIndex].x, nodes[sourceIndex].y, nodes[sourceIndex].z);
            const targetPos = new THREE.Vector3(nodes[targetIndex].x, nodes[targetIndex].y, nodes[targetIndex].z);

            // Update the geometry of the edge line with the new positions
            line.geometry.setFromPoints([sourcePos, targetPos]);

            // Update the edge color based on its weight (thickness of the line)
            line.material.color.set(getEdgeColor(edge.weight));

            // Make the edge visible in the scene
            line.visible = true;
        } else {
            // Hide the edge if source or target nodes are not found
            line.visible = false;
        }
    });

    // Hide any extra edges in the pool that are not used by the current graph data
    for (let i = edges.length; i < edgePool.length; i++) {
        if (edgePool[i]) edgePool[i].visible = false;
    }
}

/**
 * Calculates the size of a node based on its file size.
 * 
 * The node size is determined by normalizing the file size against a predefined maximum value.
 * 
 * @param {number} fileSize - Size of the file in bytes
 * @returns {number} Scaled size of the node
 */
function calculateNodeSize(fileSize) {
    // Normalize the file size to a value between 0 and 1, relative to the maximum file size
    const normalizedSize = Math.min(fileSize / MAX_FILE_SIZE, 1);

    // Scale the node size exponentially to provide a smoother size gradient
    return NODE_BASE_SIZE * Math.pow(normalizedSize, 0.5);
}

/**
 * Determines the color of a node based on its hyperlink count.
 * 
 * A gradient is applied from blue (few hyperlinks) to red (many hyperlinks).
 * 
 * @param {number} hyperlinkCount - Number of hyperlinks in the node
 * @returns {number} Hexadecimal color value for the node
 */
function getNodeColor(hyperlinkCount) {
    // Normalize the hyperlink count between 0 and 1, relative to the maximum allowed count
    const t = Math.min(hyperlinkCount / MAX_HYPERLINK_COUNT, 1);

    // Return a color that transitions from blue to red based on the hyperlink count
    return new THREE.Color(t, 0, 1 - t).getHex();
}

/**
 * Determines the color of an edge based on its weight.
 * 
 * The weight of the edge influences the color transition from green (low weight) to red (high weight).
 * 
 * @param {number} weight - Weight of the edge
 * @returns {THREE.Color} Color object for the edge
 */
function getEdgeColor(weight) {
    // Normalize the edge weight between 0 and 1, relative to the maximum edge weight
    const t = Math.min(weight / MAX_EDGE_WEIGHT, 1);

    // Return a color that transitions from green to red based on the weight
    return new THREE.Color(1 - t, t, 0);
}

/**
 * Handles window resize events to adjust the renderer and camera.
 * 
 * When the window is resized, this function ensures that the camera's aspect ratio is updated
 * and that the renderer adjusts to the new window dimensions.
 */
function onWindowResize() {
    // Update the camera's aspect ratio based on the new window dimensions
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix(); // Ensure the camera's projection is updated

    // Resize the renderer to match the new window dimensions
    renderer.setSize(window.innerWidth, window.innerHeight);
}
