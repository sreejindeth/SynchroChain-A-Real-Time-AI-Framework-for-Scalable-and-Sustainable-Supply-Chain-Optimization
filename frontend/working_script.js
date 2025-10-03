// Working JavaScript for SynchroChain Frontend
// Simplified version with proper login handling

// Global variables
let currentUser = null;
let currentSection = 'dashboard';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('SynchroChain frontend loaded');
    setupEventListeners();
    
    // Check if user is already logged in
    const savedUser = localStorage.getItem('synchrochain_user');
    if (savedUser) {
        currentUser = JSON.parse(savedUser);
        showMainApp();
    }
});

// Login function
async function login() {
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();
    const role = document.getElementById('userRole').value.trim();
    
    console.log('Login attempt:', { username, password, role });
    
    if (!username || !password || !role) {
        alert('Please fill in all fields');
        return;
    }
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                password: password,
                role: role
            })
        });
        
        const result = await response.json();
        console.log('Login response:', result);
        
        if (result.success) {
            currentUser = result.user;
            localStorage.setItem('synchrochain_user', JSON.stringify(currentUser));
            showMainApp();
            console.log('Login successful:', currentUser);
        } else {
            alert('Login failed: ' + result.error);
            console.error('Login failed:', result.error);
        }
        
    } catch (error) {
        console.error('Login error:', error);
        alert('Login error: ' + error.message);
    }
}

// Show main application
function showMainApp() {
    document.getElementById('loginModal').style.display = 'none';
    document.getElementById('mainApp').style.display = 'block';
    
    // Update user display
    document.getElementById('currentUser').textContent = currentUser.username;
    
    // Show/hide admin section based on role
    const adminSection = document.getElementById('adminSection');
    if (currentUser.role === 'admin') {
        adminSection.style.display = 'block';
    } else {
        adminSection.style.display = 'none';
    }
    
    // Initialize dashboard
    initializeDashboard();
}

// Logout function
function logout() {
    currentUser = null;
    localStorage.removeItem('synchrochain_user');
    document.getElementById('loginModal').style.display = 'block';
    document.getElementById('mainApp').style.display = 'none';
}

// Section navigation
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show selected section
    document.getElementById(sectionId).style.display = 'block';
    
    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`[href="#${sectionId}"]`).classList.add('active');
    
    currentSection = sectionId;
    
    // Initialize section-specific content
    if (sectionId === 'dashboard') {
        initializeDashboard();
    } else if (sectionId === 'ai-pipeline') {
        initializePipelineView();
    } else if (sectionId === 'predictions') {
        initializePredictions();
    } else if (sectionId === 'analytics') {
        initializeAnalytics();
    } else if (sectionId === 'admin') {
        initializeAdmin();
    }
}

// Initialize dashboard
function initializeDashboard() {
    updateMetrics();
    initializeCharts();
}

// Update metrics
async function updateMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const result = await response.json();
        
        if (result.success) {
            const metrics = result.metrics;
            
            document.getElementById('active-orders').textContent = metrics.active_orders.toLocaleString();
            document.getElementById('ai-accuracy').textContent = metrics.ai_accuracy + '%';
            document.getElementById('processing-speed').textContent = metrics.processing_speed_ms + 'ms';
            document.getElementById('cost-savings').textContent = metrics.cost_savings_percent + '%';
        }
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

// Initialize charts
function initializeCharts() {
    // Order trend chart
    const orderTrendData = [{
        x: generateDateRange(30),
        y: generateRandomData(30, 800, 1500),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Orders',
        line: { color: '#1f77b4', width: 3 }
    }];
    
    const orderTrendLayout = {
        title: '',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Orders' },
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot('order-trend-chart', orderTrendData, orderTrendLayout, {responsive: true});
    
    // Decision pie chart
    const decisionPieData = [{
        values: [45, 30, 25],
        labels: ['Standard', 'Expedited', 'Priority'],
        type: 'pie',
        marker: { colors: ['#28a745', '#ffc107', '#dc3545'] },
        textinfo: 'label+percent'
    }];
    
    const decisionPieLayout = {
        title: '',
        margin: { t: 20, r: 20, b: 20, l: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot('decision-pie-chart', decisionPieData, decisionPieLayout, {responsive: true});
}

// Initialize pipeline view
function initializePipelineView() {
    console.log('Pipeline view initialized');
    
    // Load pipeline status
    loadPipelineStatus();
    
    // Create pipeline flow visualization
    createPipelineFlow();
}

// Load pipeline status
async function loadPipelineStatus() {
    try {
        const response = await fetch('/api/pipeline/status');
        const result = await response.json();
        
        if (result.success) {
            updatePipelineStatus(result.pipeline_status);
        }
    } catch (error) {
        console.error('Error loading pipeline status:', error);
    }
}

// Update pipeline status
function updatePipelineStatus(status) {
    // Update status badges
    updateStatusBadge('intent-status', status.intent_transformer.status);
    updateStatusBadge('gnn-status', status.gnn_module.status);
    updateStatusBadge('rl-status', status.rl_agent.status);
    updateStatusBadge('orchestrator-status', status.orchestrator.status);
    
    // Update metrics
    updatePipelineMetrics(status);
}

// Update status badge
function updateStatusBadge(elementId, status) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        element.className = `badge ${status === 'online' ? 'bg-success' : 'bg-danger'}`;
    }
}

// Update pipeline metrics
function updatePipelineMetrics(status) {
    // Update processing times
    updateMetric('intent-processing-time', status.intent_transformer.processing_time_ms + 'ms');
    updateMetric('gnn-processing-time', status.gnn_module.processing_time_ms + 'ms');
    updateMetric('rl-processing-time', status.rl_agent.processing_time_ms + 'ms');
    updateMetric('orchestrator-throughput', status.orchestrator.throughput_per_second + ' req/s');
}

// Update metric
function updateMetric(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

// Create pipeline flow visualization
function createPipelineFlow() {
    // This would create an interactive pipeline flow diagram
    console.log('Pipeline flow visualization created');
}

// Initialize predictions
function initializePredictions() {
    console.log('Predictions view initialized');
    
    // Set up event listeners for prediction form
    setupPredictionForm();
    
    // Initialize sliders
    setupSliders();
}

// Setup prediction form
function setupPredictionForm() {
    // Session duration slider
    const sessionSlider = document.getElementById('session-duration');
    if (sessionSlider) {
        sessionSlider.addEventListener('input', function(e) {
            const minutes = e.target.value;
            document.getElementById('session-duration-value').textContent = minutes + ' minutes';
        });
    }
}

// Setup sliders
function setupSliders() {
    // Inventory slider
    const inventorySlider = document.getElementById('inventory-slider');
    if (inventorySlider) {
        inventorySlider.addEventListener('input', function(e) {
            document.getElementById('inventory-value').textContent = e.target.value;
        });
    }
    
    // Delay slider
    const delaySlider = document.getElementById('delay-slider');
    if (delaySlider) {
        delaySlider.addEventListener('input', function(e) {
            document.getElementById('delay-value').textContent = e.target.value + '%';
        });
    }
    
    // Carbon slider
    const carbonSlider = document.getElementById('carbon-slider');
    if (carbonSlider) {
        carbonSlider.addEventListener('input', function(e) {
            const value = parseInt(e.target.value);
            let category = 'Low';
            if (value > 75) category = 'Very High';
            else if (value > 50) category = 'High';
            else if (value > 25) category = 'Medium';
            document.getElementById('carbon-value').textContent = category;
        });
    }
}

// Initialize analytics
function initializeAnalytics() {
    console.log('Analytics view initialized');
    
    // Load performance data
    loadPerformanceData();
    
    // Update model performance cards
    updateModelPerformanceCards();
}

// Load performance data
async function loadPerformanceData() {
    try {
        const response = await fetch('/api/performance?days=90');
        const result = await response.json();
        
        if (result.success) {
            createPerformanceChart(result.data);
        }
    } catch (error) {
        console.error('Error loading performance data:', error);
    }
}

// Create performance chart
function createPerformanceChart(data) {
    const performanceData = [
        {
            x: data.dates,
            y: data.intent_accuracy,
            type: 'scatter',
            mode: 'lines',
            name: 'Intent Accuracy',
            line: { color: '#1f77b4', width: 2 }
        },
        {
            x: data.dates,
            y: data.gnn_performance,
            type: 'scatter',
            mode: 'lines',
            name: 'GNN Performance',
            line: { color: '#ffc107', width: 2 }
        },
        {
            x: data.dates,
            y: data.ppo_efficiency,
            type: 'scatter',
            mode: 'lines',
            name: 'PPO Efficiency',
            line: { color: '#28a745', width: 2 }
        }
    ];
    
    const performanceLayout = {
        title: '',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Performance %' },
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif' }
    };
    
    Plotly.newPlot('performance-chart', performanceData, performanceLayout, {responsive: true});
}

// Update model performance cards
async function updateModelPerformanceCards() {
    try {
        const response = await fetch('/api/pipeline/status');
        const result = await response.json();
        
        if (result.success) {
            const status = result.pipeline_status;
            
            // Update progress bars based on status
            updateProgressBar('intent-progress', status.intent_transformer.accuracy * 100);
            updateProgressBar('gnn-progress', (1 - status.gnn_module.mae) * 100);
            updateProgressBar('ppo-progress', status.rl_agent.improvement * 100);
            updateProgressBar('orchestrator-progress', 90); // Fixed high performance
        }
    } catch (error) {
        console.error('Error updating model performance:', error);
    }
}

// Update progress bar
function updateProgressBar(elementId, percentage) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.width = percentage + '%';
    }
}

// Initialize admin
function initializeAdmin() {
    if (currentUser.role !== 'admin') {
        alert('Access denied. Admin role required.');
        showSection('dashboard');
        return;
    }
    console.log('Admin view initialized');
}

// Run AI prediction
async function runFullAIPipeline() {
    const button = document.querySelector('button[onclick="runFullAIPipeline()"]');
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="loading"></span> Processing...';
    button.disabled = true;
    
    try {
        // Get input values
        const products = Array.from(document.getElementById('product-select').selectedOptions).map(option => option.value);
        const warehouse = document.getElementById('warehouse-select').value;
        const inventory = parseInt(document.getElementById('inventory-slider').value);
        const delayRisk = parseInt(document.getElementById('delay-slider').value);
        const sessionDuration = parseInt(document.getElementById('session-duration').value);
        
        // Get user actions
        const addToCart = document.getElementById('action-cart').checked;
        const searchAction = document.getElementById('action-search').checked;
        
        console.log('Running prediction with:', {
            products, warehouse, inventory, delayRisk, sessionDuration, addToCart, searchAction
        });
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                products: products,
                warehouse: warehouse,
                inventory: inventory,
                delay_risk: delayRisk,
                session_duration: sessionDuration,
                add_to_cart: addToCart,
                search_queries: searchAction ? 1 : 0
            })
        });
        
        const result = await response.json();
        console.log('Prediction result:', result);
        
        if (result.success) {
            displayDetailedPredictionResults(result);
            document.getElementById('pipeline-results').style.display = 'block';
            
            // Scroll to results
            document.getElementById('pipeline-results').scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Prediction failed: ' + result.error);
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Prediction error: ' + error.message);
    } finally {
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

// Display detailed prediction results
function displayDetailedPredictionResults(result) {
    console.log('Displaying prediction results:', result);
    
    // Intent Transformer Results
    document.getElementById('intent-score').textContent = result.intent_score;
    document.getElementById('urgency-level').textContent = result.urgency_score > 0.7 ? 'High' : 
                                                      result.urgency_score > 0.4 ? 'Medium' : 'Low';
    
    // GNN Module Results
    document.getElementById('delay-risk').textContent = result.risk_assessment;
    document.getElementById('inventory-status').textContent = result.inventory_level > 200 ? 'Good' : 
                                                           result.inventory_level > 100 ? 'Medium' : 'Low';
    document.getElementById('carbon-cost').textContent = result.carbon_cost || 'Medium';
    
    // RL Agent Results
    document.getElementById('final-action').textContent = result.final_decision;
    document.getElementById('confidence-score').textContent = result.confidence;
    document.getElementById('expected-reward').textContent = '+' + (result.confidence * 15).toFixed(1);
    
    // Orchestrator Results
    document.getElementById('final-decision').textContent = result.final_decision + ' Shipping';
    document.getElementById('processing-time').textContent = result.processing_time_ms + 'ms';
    document.getElementById('feature-store').textContent = 'Updated';
    
    // Create visual charts
    createPredictionCharts(result);
}

// Create prediction charts
function createPredictionCharts(result) {
    // Intent Chart
    const intentData = [{
        values: [result.intent_score, 1 - result.intent_score],
        labels: ['Purchase Intent', 'Remaining'],
        type: 'pie',
        hole: 0.4,
        marker: { colors: ['#17a2b8', '#e9ecef'] },
        textinfo: 'none',
        showlegend: false
    }];
    
    const intentLayout = {
        title: `${(result.intent_score * 100).toFixed(1)}%`,
        margin: { t: 40, r: 20, b: 20, l: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif', size: 16 },
        annotations: [{
            text: `${(result.intent_score * 100).toFixed(1)}%`,
            x: 0.5, y: 0.5,
            font: { size: 20, color: '#17a2b8' },
            showarrow: false
        }]
    };
    
    Plotly.newPlot('intent-chart', intentData, intentLayout, {responsive: true});
    
    // Risk Chart
    const riskData = [{
        values: [result.risk_assessment, 1 - result.risk_assessment],
        labels: ['Risk Level', 'Safe Zone'],
        type: 'pie',
        hole: 0.4,
        marker: { colors: ['#ffc107', '#e9ecef'] },
        textinfo: 'none',
        showlegend: false
    }];
    
    const riskLayout = {
        title: `${(result.risk_assessment * 100).toFixed(1)}%`,
        margin: { t: 40, r: 20, b: 20, l: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif', size: 16 },
        annotations: [{
            text: `${(result.risk_assessment * 100).toFixed(1)}%`,
            x: 0.5, y: 0.5,
            font: { size: 20, color: '#ffc107' },
            showarrow: false
        }]
    };
    
    Plotly.newPlot('risk-chart', riskData, riskLayout, {responsive: true});
    
    // RL Agent Chart
    const rlData = [{
        values: [result.confidence, 1 - result.confidence],
        labels: ['Confidence', 'Uncertainty'],
        type: 'pie',
        hole: 0.4,
        marker: { colors: ['#28a745', '#e9ecef'] },
        textinfo: 'none',
        showlegend: false
    }];
    
    const rlLayout = {
        title: `${(result.confidence * 100).toFixed(1)}%`,
        margin: { t: 40, r: 20, b: 20, l: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif', size: 16 },
        annotations: [{
            text: `${(result.confidence * 100).toFixed(1)}%`,
            x: 0.5, y: 0.5,
            font: { size: 20, color: '#28a745' },
            showarrow: false
        }]
    };
    
    Plotly.newPlot('rl-chart', rlData, rlLayout, {responsive: true});
    
    // Orchestrator Chart
    const orchData = [{
        values: [1],
        labels: ['Processing Complete'],
        type: 'pie',
        hole: 0.4,
        marker: { colors: ['#6f42c1'] },
        textinfo: 'none',
        showlegend: false
    }];
    
    const orchLayout = {
        title: 'Complete',
        margin: { t: 40, r: 20, b: 20, l: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif', size: 16 },
        annotations: [{
            text: '✓',
            x: 0.5, y: 0.5,
            font: { size: 20, color: '#6f42c1' },
            showarrow: false
        }]
    };
    
    Plotly.newPlot('orchestrator-chart', orchData, orchLayout, {responsive: true});
}

// Setup event listeners
function setupEventListeners() {
    // Login form
    document.getElementById('loginForm').addEventListener('submit', function(e) {
        e.preventDefault();
        login();
    });
    
    // Inventory slider
    const inventorySlider = document.getElementById('inventory-slider');
    if (inventorySlider) {
        inventorySlider.addEventListener('input', function(e) {
            document.getElementById('inventory-value').textContent = e.target.value;
        });
    }
    
    // Delay slider
    const delaySlider = document.getElementById('delay-slider');
    if (delaySlider) {
        delaySlider.addEventListener('input', function(e) {
            document.getElementById('delay-value').textContent = e.target.value + '%';
        });
    }
}

// Utility functions
function generateDateRange(days) {
    const dates = [];
    const today = new Date();
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        dates.push(date.toISOString().split('T')[0]);
    }
    return dates;
}

function generateRandomData(length, min, max) {
    const data = [];
    for (let i = 0; i < length; i++) {
        data.push(Math.floor(Math.random() * (max - min + 1)) + min);
    }
    return data;
}

// Profile function
function showProfile() {
    alert('Profile: ' + currentUser.username + ' (' + currentUser.role + ')');
}
