/* ============================================================================
   SPIDER GRAPH VISUALIZER - APPLICATION LOGIC
   ============================================================================
   
   This file handles:
   1. WebSocket connection for realtime updates
   2. Sigma.js graph initialization and rendering
   3. ForceAtlas2 layout algorithm
   4. UI interactions (hover tooltips, zoom controls)
   
   ============================================================================ */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    // WebSocket server URL (same host, ws protocol)
    WS_URL: `ws://${window.location.host}/ws`,

    // REST API base URL
    API_URL: `http://${window.location.host}/api`,

    // Layout settings for ForceAtlas2
    LAYOUT: {
        iterations: 500,           // More iterations for better separation
        settings: {
            gravity: 1,            // Attraction to center
            scalingRatio: 50,      // More spread out (was 10)
            barnesHutOptimize: true, // Use Barnes-Hut optimization for large graphs
            barnesHutTheta: 0.5,
            strongGravityMode: false,
            slowDown: 1,
            adjustSizes: true,     // Prevent node overlap based on size
            outboundAttractionDistribution: true, // Better cluster separation
        }
    },

    // Node rendering settings
    NODE: {
        defaultSize: 3,      // Smaller default (was 5)
        minSize: 2,          // Smaller min (was 3)
        maxSize: 8,          // Smaller max (was 15)
        defaultColor: '#808080',   // Gray for unclustered nodes
    },

    // Edge rendering settings
    EDGE: {
        defaultColor: 'rgba(100, 100, 100, 0.3)',
        highlightColor: 'rgba(255, 255, 255, 0.6)',
    },

    // Zoom thresholds for label visibility
    ZOOM: {
        showLabelsRatio: 0.8,  // Only show labels when zoomed in past this ratio
    }
};

// ============================================================================
// GLOBAL STATE
// ============================================================================

let graph = null;           // Graphology graph instance
let renderer = null;        // Sigma.js renderer
let ws = null;              // WebSocket connection
let layoutRunning = false;  // Is layout algorithm currently running?

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize the application when DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🕸️ Spider Graph Visualizer initializing...');

    // Initialize the graph data structure
    initGraph();

    // Connect to WebSocket for realtime updates
    connectWebSocket();

    // Set up UI event handlers
    setupEventHandlers();
});

/**
 * Initialize the Graphology graph and Sigma.js renderer
 */
function initGraph() {
    // Create a new Graphology graph
    // Graphology is the data structure library that Sigma.js uses
    graph = new graphology.Graph();

    // Get the container element
    const container = document.getElementById('graph-container');

    // Create the Sigma.js renderer
    // This uses WebGL for high-performance rendering
    renderer = new Sigma(graph, container, {
        // Rendering settings
        renderEdgeLabels: false,
        allowInvalidContainer: true,

        // Node appearance
        defaultNodeColor: CONFIG.NODE.defaultColor,

        // Edge appearance
        defaultEdgeColor: CONFIG.EDGE.defaultColor,
        defaultEdgeType: 'line',

        // Label settings - only show when zoomed in
        labelSize: 11,
        labelColor: { color: '#ffffff' },
        labelRenderedSizeThreshold: 8, // Higher threshold = fewer labels shown

        // Interaction
        minCameraRatio: 0.1,
        maxCameraRatio: 10,
    });

    // Dynamic label visibility based on zoom level
    // Labels hidden when zoomed out, shown when zoomed in
    renderer.setSetting('labelRenderer', (context, data, settings) => {
        const camera = renderer.getCamera();
        const ratio = camera.ratio;

        // Only render labels when zoomed in enough
        if (ratio > CONFIG.ZOOM.showLabelsRatio) {
            return; // Don't render labels when zoomed out
        }

        // Default label rendering
        const size = settings.labelSize;
        const font = settings.labelFont;
        const weight = settings.labelWeight;
        const color = settings.labelColor.color;

        context.fillStyle = color;
        context.font = `${weight} ${size}px ${font}`;
        context.fillText(data.label, data.x + data.size + 3, data.y + size / 3);
    });

    // Set up hover effects
    setupHoverEffects();

    console.log('✅ Graph and renderer initialized');
}

// ============================================================================
// WEBSOCKET CONNECTION
// ============================================================================

/**
 * Connect to the WebSocket server for realtime updates
 */
function connectWebSocket() {
    console.log(`🔌 Connecting to WebSocket: ${CONFIG.WS_URL}`);

    // Create WebSocket connection
    ws = new WebSocket(CONFIG.WS_URL);

    // Connection opened
    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        updateConnectionStatus(true);
    };

    // Handle incoming messages
    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    // Connection closed
    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        updateConnectionStatus(false);

        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    // Connection error
    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        updateConnectionStatus(false);
    };
}

/**
 * Handle incoming WebSocket messages
 * Messages are tagged with a 'type' field
 */
function handleWebSocketMessage(message) {
    console.log('📥 Received:', message.type);

    switch (message.type) {
        case 'graph_snapshot':
            // Full graph data - replace everything
            handleGraphSnapshot(message.data);
            break;

        case 'node_added':
            // Single node added
            handleNodeAdded(message.data);
            break;

        case 'edge_added':
            // Single edge added
            handleEdgeAdded(message.data);
            break;

        case 'cluster_updated':
            // Cluster colors changed
            handleClusterUpdated(message.data);
            break;

        case 'refresh':
            // Full refresh requested
            loadGraphFromAPI();
            break;

        default:
            console.warn('Unknown message type:', message.type);
    }
}

// ============================================================================
// GRAPH DATA HANDLERS
// ============================================================================

/**
 * Handle a full graph snapshot
 * This clears the existing graph and loads new data
 */
function handleGraphSnapshot(data) {
    console.log(`📊 Loading graph: ${data.nodes.length} nodes, ${data.edges.length} edges`);

    // Clear existing graph
    graph.clear();

    // Add all nodes
    for (const node of data.nodes) {
        addNodeToGraph(node);
    }

    // Add all edges
    for (const edge of data.edges) {
        addEdgeToGraph(edge);
    }

    // Update UI
    updateStats(data.nodes.length, data.edges.length, Object.keys(data.clusters).length);
    updateLegend(data.clusters);

    // Run layout algorithm to position nodes
    runLayout();

    console.log('✅ Graph loaded');
}

/**
 * Add a single node to the graph
 */
function addNodeToGraph(node) {
    // Calculate node size based on significance (0-9)
    const size = CONFIG.NODE.minSize +
        (node.significance / 9) * (CONFIG.NODE.maxSize - CONFIG.NODE.minSize);

    // Add node to Graphology
    graph.addNode(node.id.toString(), {
        // Position (random if not provided, layout will fix it)
        x: node.x ?? Math.random() * 1000,
        y: node.y ?? Math.random() * 1000,

        // Visual properties
        size: size,
        color: node.color,

        // Data for tooltips
        label: node.label,
        content: node.content,
        significance: node.significance,
        clusterId: node.cluster_id,
    });
}

/**
 * Add a single edge to the graph
 */
function addEdgeToGraph(edge) {
    const sourceId = edge.source.toString();
    const targetId = edge.target.toString();

    // Only add if both nodes exist and edge doesn't already exist
    if (graph.hasNode(sourceId) && graph.hasNode(targetId)) {
        const edgeId = `${sourceId}-${targetId}`;
        if (!graph.hasEdge(edgeId)) {
            graph.addEdge(sourceId, targetId, {
                id: edgeId,
                color: CONFIG.EDGE.defaultColor,
            });
        }
    }
}

/**
 * Handle a single node being added (realtime update)
 */
function handleNodeAdded(node) {
    console.log(`➕ Node added: ${node.id}`);

    addNodeToGraph(node);

    // Update stats
    const nodeCount = graph.order;
    const edgeCount = graph.size;
    updateStats(nodeCount, edgeCount);
}

/**
 * Handle a single edge being added (realtime update)
 */
function handleEdgeAdded(edge) {
    console.log(`🔗 Edge added: ${edge.source} -> ${edge.target}`);

    addEdgeToGraph(edge);

    // Update stats
    const nodeCount = graph.order;
    const edgeCount = graph.size;
    updateStats(nodeCount, edgeCount);
}

/**
 * Handle cluster information update
 */
function handleClusterUpdated(clusterInfo) {
    console.log(`🎨 Cluster updated: ${clusterInfo.id}`);

    // Update node colors for this cluster
    graph.forEachNode((nodeId, attributes) => {
        if (attributes.clusterId === clusterInfo.id) {
            graph.setNodeAttribute(nodeId, 'color', clusterInfo.color);
        }
    });
}

// ============================================================================
// LAYOUT ALGORITHM - Cluster-Centric Circular Layout
// ============================================================================

/**
 * Run cluster-centric layout algorithm
 * 
 * This layout works in 3 steps:
 * 1. Position clusters in circular sectors around the center
 * 2. Position nodes within their cluster's sector
 * 3. Run NOverlap to prevent any remaining overlaps
 */
function runLayout() {
    if (layoutRunning) {
        console.log('⏳ Layout already running');
        return;
    }

    console.log('🔄 Running Cluster-Centric Layout...');
    layoutRunning = true;

    // ========================================================================
    // STEP 1: Collect nodes by cluster
    // ========================================================================
    console.log('   Step 1: Grouping nodes by cluster...');

    const clusterNodes = new Map();  // clusterId -> [nodeIds]
    const unclusteredNodes = [];

    graph.forEachNode((nodeId, attrs) => {
        const clusterId = attrs.clusterId;
        if (clusterId !== null && clusterId !== undefined) {
            if (!clusterNodes.has(clusterId)) {
                clusterNodes.set(clusterId, []);
            }
            clusterNodes.get(clusterId).push(nodeId);
        } else {
            unclusteredNodes.push(nodeId);
        }
    });

    console.log(`   Found ${clusterNodes.size} clusters, ${unclusteredNodes.length} unclustered`);

    // ========================================================================
    // STEP 2: Position clusters in circular sectors
    // ========================================================================
    console.log('   Step 2: Positioning clusters in circular layout...');

    const numClusters = clusterNodes.size;
    const centerX = 0;
    const centerY = 0;
    const clusterRadius = 300;  // Distance from center to cluster center
    const nodeSpread = 100;     // How spread out nodes are within a cluster

    let clusterIndex = 0;

    for (const [clusterId, nodes] of clusterNodes) {
        // Calculate angle for this cluster (evenly distributed around circle)
        const angle = (clusterIndex / numClusters) * 2 * Math.PI;

        // Cluster center position
        const clusterCenterX = centerX + clusterRadius * Math.cos(angle);
        const clusterCenterY = centerY + clusterRadius * Math.sin(angle);

        // Position each node within the cluster in a smaller circle
        const nodesInCluster = nodes.length;
        nodes.forEach((nodeId, nodeIndex) => {
            // Arrange nodes in a small circle within the cluster
            const nodeAngle = (nodeIndex / nodesInCluster) * 2 * Math.PI;
            const nodeRadius = Math.min(nodeSpread, nodesInCluster * 8);

            const x = clusterCenterX + nodeRadius * Math.cos(nodeAngle);
            const y = clusterCenterY + nodeRadius * Math.sin(nodeAngle);

            graph.setNodeAttribute(nodeId, 'x', x);
            graph.setNodeAttribute(nodeId, 'y', y);
        });

        clusterIndex++;
    }

    // Position unclustered nodes in the center
    unclusteredNodes.forEach((nodeId, i) => {
        const angle = (i / Math.max(unclusteredNodes.length, 1)) * 2 * Math.PI;
        const radius = 50;
        graph.setNodeAttribute(nodeId, 'x', centerX + radius * Math.cos(angle));
        graph.setNodeAttribute(nodeId, 'y', centerY + radius * Math.sin(angle));
    });

    // ========================================================================
    // STEP 3: Run ForceAtlas2 briefly to refine positions
    // ========================================================================
    console.log('   Step 3: Refining with ForceAtlas2...');

    const fa2Layout = graphologyLibrary.layoutForceAtlas2;
    fa2Layout.assign(graph, {
        iterations: 50,  // Less iterations since we have good initial positions
        settings: {
            gravity: 0.5,
            scalingRatio: 30,
            barnesHutOptimize: true,
            strongGravityMode: false,
            slowDown: 2,
        }
    });

    // ========================================================================
    // STEP 4: Run NOverlap to prevent any remaining overlaps
    // ========================================================================
    console.log('   Step 4: NOverlap (final separation)...');
    if (typeof graphologyLibrary.layoutNoverlap !== 'undefined') {
        graphologyLibrary.layoutNoverlap.assign(graph, {
            maxIterations: 50,
            ratio: 2.0,
            margin: 10,
            speed: 5,
        });
    }

    layoutRunning = false;
    console.log('✅ Cluster-centric layout complete');

    // Fit the camera to show all nodes
    fitView();
}

/**
 * Fit the camera to show all nodes
 */
function fitView() {
    const camera = renderer.getCamera();
    camera.animatedReset({ duration: 500 });
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

/**
 * Load graph data from REST API (fallback if WebSocket fails)
 */
async function loadGraphFromAPI() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/graph`);
        if (response.ok) {
            const data = await response.json();
            handleGraphSnapshot(data);
        }
    } catch (e) {
        console.error('Failed to load graph from API:', e);
    }
}

/**
 * Trigger a server-side notify (for testing)
 */
async function triggerNotify() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/notify`, { method: 'POST' });
        console.log('Notify response:', await response.json());
    } catch (e) {
        console.error('Failed to trigger notify:', e);
    }
}

// ============================================================================
// UI FUNCTIONS
// ============================================================================

/**
 * Update the connection status indicator
 */
function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    const textEl = statusEl.querySelector('.status-text');

    if (connected) {
        statusEl.className = 'status connected';
        textEl.textContent = 'Connected';
    } else {
        statusEl.className = 'status disconnected';
        textEl.textContent = 'Disconnected';
    }
}

/**
 * Update the statistics panel
 */
function updateStats(nodes, edges, clusters = null) {
    document.getElementById('node-count').textContent = nodes.toLocaleString();
    document.getElementById('edge-count').textContent = edges.toLocaleString();

    if (clusters !== null) {
        document.getElementById('cluster-count').textContent = clusters.toLocaleString();
    }
}

/**
 * Update the cluster legend
 */
function updateLegend(clusters) {
    const legendEl = document.getElementById('cluster-legend');
    legendEl.innerHTML = '';

    // Sort clusters by ID
    const sortedClusters = Object.values(clusters).sort((a, b) => a.id - b.id);

    for (const cluster of sortedClusters) {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <div class="legend-color" style="background: ${cluster.color}"></div>
            <span class="legend-label">Cluster ${cluster.id}</span>
            <span class="legend-count">${cluster.node_count}</span>
        `;
        legendEl.appendChild(item);
    }

    // Add unclustered item
    const unclustered = document.createElement('div');
    unclustered.className = 'legend-item';
    unclustered.innerHTML = `
        <div class="legend-color" style="background: ${CONFIG.NODE.defaultColor}"></div>
        <span class="legend-label">Unclustered</span>
    `;
    legendEl.appendChild(unclustered);
}

/**
 * Set up hover effects for nodes
 */
function setupHoverEffects() {
    const nodeInfoEl = document.getElementById('node-info');

    // Show tooltip on hover
    renderer.on('enterNode', ({ node }) => {
        const attrs = graph.getNodeAttributes(node);

        document.getElementById('node-info-title').textContent = `Node ${node}`;
        document.getElementById('node-info-content').textContent = attrs.content || attrs.label;
        document.getElementById('node-info-cluster').textContent = attrs.clusterId ?? 'None';
        document.getElementById('node-info-sig').textContent = `${attrs.significance}/9`;

        nodeInfoEl.classList.remove('hidden');
    });

    // Track mouse for tooltip position
    renderer.on('clickStage', () => {
        nodeInfoEl.classList.add('hidden');
    });

    // Update tooltip position
    renderer.getMouseCaptor().on('mousemove', (e) => {
        if (!nodeInfoEl.classList.contains('hidden')) {
            nodeInfoEl.style.left = `${e.x + 15}px`;
            nodeInfoEl.style.top = `${e.y + 15}px`;
        }
    });

    // Hide tooltip when leaving node
    renderer.on('leaveNode', () => {
        nodeInfoEl.classList.add('hidden');
    });
}

/**
 * Set up button event handlers
 */
function setupEventHandlers() {
    // Re-Layout button
    document.getElementById('btn-layout').addEventListener('click', () => {
        runLayout();
    });

    // Fit View button
    document.getElementById('btn-fit').addEventListener('click', () => {
        fitView();
    });
}
