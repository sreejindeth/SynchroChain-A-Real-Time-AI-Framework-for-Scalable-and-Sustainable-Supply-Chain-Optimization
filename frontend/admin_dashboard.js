// Admin Dashboard JavaScript
let orders = [];
let currentSection = 'overview';
let realtimeInterval;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    checkAdminAuth();
    loadDashboardData();
    setupEventListeners();
    startRealtimeUpdates();
});

// Check admin authentication
function checkAdminAuth() {
    const userData = localStorage.getItem('currentUser');
    if (!userData) {
        window.location.href = 'enhanced_index.html';
        return;
    }
    
    const user = JSON.parse(userData);
    if (user.role !== 'admin') {
        alert('Access denied. Admin role required.');
        window.location.href = 'enhanced_index.html';
        return;
    }
}

// Setup event listeners
function setupEventListeners() {
    // Threshold sliders
    document.getElementById('intentThreshold').addEventListener('input', function(e) {
        document.getElementById('intentThresholdValue').textContent = e.target.value;
    });
    
    document.getElementById('riskThreshold').addEventListener('input', function(e) {
        document.getElementById('riskThresholdValue').textContent = e.target.value;
    });
}

// Load dashboard data
async function loadDashboardData() {
    await loadOrders();
    await loadSystemMetrics();
    await loadAIPipelineStatus();
    createCharts();
}

// Load orders
async function loadOrders() {
    try {
        const response = await fetch('/api/admin/orders');
        const result = await response.json();
        
        if (result.success) {
            orders = result.orders;
            displayOrders(orders);
        } else {
            console.error('Failed to load orders:', result.error);
            loadMockOrders();
        }
    } catch (error) {
        console.error('Error loading orders:', error);
        loadMockOrders();
    }
}

// Load mock orders if API fails
function loadMockOrders() {
    orders = [
        {
            id: 'ORD-001',
            user_id: 'user123',
            items: [
                { name: 'Smart Watch', price: 327.75, quantity: 1 },
                { name: 'Wireless Headphones', price: 199.99, quantity: 1 }
            ],
            total: 527.74,
            status: 'processing',
            created_at: new Date().toISOString(),
            ai_timeline: {
                intent_score: 0.75,
                urgency_score: 0.68,
                risk_assessment: 0.25,
                final_decision: 'expedited',
                confidence: 0.82,
                processing_time: 4.2
            }
        },
        {
            id: 'ORD-002',
            user_id: 'user456',
            items: [
                { name: 'Fitness Tracker', price: 149.99, quantity: 2 }
            ],
            total: 299.98,
            status: 'completed',
            created_at: new Date(Date.now() - 3600000).toISOString(),
            ai_timeline: {
                intent_score: 0.45,
                urgency_score: 0.35,
                risk_assessment: 0.15,
                final_decision: 'standard',
                confidence: 0.68,
                processing_time: 3.1
            }
        }
    ];
    displayOrders(orders);
}

// Display orders
function displayOrders(ordersToShow) {
    const ordersList = document.getElementById('ordersList');
    ordersList.innerHTML = '';
    
    ordersToShow.forEach(order => {
        const orderCard = createOrderCard(order);
        ordersList.appendChild(orderCard);
    });
}

// Create order card
function createOrderCard(order) {
    const col = document.createElement('div');
    col.className = 'col-md-6 col-lg-4';
    
    const statusClass = {
        'pending': 'warning',
        'processing': 'info',
        'completed': 'success',
        'cancelled': 'danger'
    }[order.status] || 'secondary';
    
    col.innerHTML = `
        <div class="card order-card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h6 class="mb-0">Order ${order.id}</h6>
                <span class="badge bg-${statusClass}">${order.status.toUpperCase()}</span>
            </div>
            <div class="card-body">
                <div class="row mb-2">
                    <div class="col-6">
                        <small class="text-muted">User ID:</small>
                        <div>${order.user_id}</div>
                    </div>
                    <div class="col-6">
                        <small class="text-muted">Total:</small>
                        <div>$${order.total.toFixed(2)}</div>
                    </div>
                </div>
                <div class="mb-2">
                    <small class="text-muted">Items:</small>
                    <div>${order.items.map(item => `${item.name} x${item.quantity}`).join(', ')}</div>
                </div>
                <div class="mb-3">
                    <small class="text-muted">Created:</small>
                    <div>${new Date(order.created_at).toLocaleString()}</div>
                </div>
                ${order.ai_timeline ? `
                    <div class="ai-timeline-small">
                        <div class="row text-center">
                            <div class="col-3">
                                <div class="ai-score-small">${(order.ai_timeline.intent_score * 100).toFixed(0)}%</div>
                                <small>Intent</small>
                            </div>
                            <div class="col-3">
                                <div class="ai-score-small">${(order.ai_timeline.urgency_score * 100).toFixed(0)}%</div>
                                <small>Urgency</small>
                            </div>
                            <div class="col-3">
                                <div class="ai-score-small">${(order.ai_timeline.risk_assessment * 100).toFixed(0)}%</div>
                                <small>Risk</small>
                            </div>
                            <div class="col-3">
                                <div class="ai-score-small">${order.ai_timeline.final_decision}</div>
                                <small>Decision</small>
                            </div>
                        </div>
                    </div>
                ` : ''}
                <button class="btn btn-primary btn-sm w-100" onclick="viewOrderDetails('${order.id}')">
                    <i class="fas fa-eye me-1"></i>View AI Timeline
                </button>
            </div>
        </div>
    `;
    
    return col;
}

// View order details
function viewOrderDetails(orderId) {
    const order = orders.find(o => o.id === orderId);
    if (!order) return;
    
    const orderDetails = document.getElementById('orderDetails');
    orderDetails.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h5>Order Information</h5>
                <table class="table table-sm">
                    <tr><td><strong>Order ID:</strong></td><td>${order.id}</td></tr>
                    <tr><td><strong>User ID:</strong></td><td>${order.user_id}</td></tr>
                    <tr><td><strong>Status:</strong></td><td><span class="badge bg-${order.status === 'completed' ? 'success' : order.status === 'processing' ? 'info' : 'warning'}">${order.status.toUpperCase()}</span></td></tr>
                    <tr><td><strong>Total:</strong></td><td>$${order.total.toFixed(2)}</td></tr>
                    <tr><td><strong>Created:</strong></td><td>${new Date(order.created_at).toLocaleString()}</td></tr>
                </table>
                
                <h6>Items</h6>
                <ul class="list-group">
                    ${order.items.map(item => `
                        <li class="list-group-item d-flex justify-content-between">
                            <span>${item.name} x${item.quantity}</span>
                            <span>$${(item.price * item.quantity).toFixed(2)}</span>
                        </li>
                    `).join('')}
                </ul>
            </div>
            <div class="col-md-6">
                <h5>AI Pipeline Timeline</h5>
                ${order.ai_timeline ? createAITimeline(order.ai_timeline) : '<p class="text-muted">No AI data available</p>'}
            </div>
        </div>
    `;
    
    new bootstrap.Modal(document.getElementById('orderModal')).show();
}

// Create AI timeline
function createAITimeline(timeline) {
    return `
        <div class="timeline">
            <div class="timeline-step completed">
                <div class="d-flex justify-content-between">
                    <span><i class="fas fa-eye me-2"></i>Intent Analysis</span>
                    <span class="badge bg-success">${(timeline.intent_score * 100).toFixed(1)}%</span>
                </div>
                <small class="text-muted">Analyzed user behavior and purchase intent</small>
            </div>
            
            <div class="timeline-step completed">
                <div class="d-flex justify-content-between">
                    <span><i class="fas fa-project-diagram me-2"></i>GNN Risk Assessment</span>
                    <span class="badge bg-warning">${(timeline.risk_assessment * 100).toFixed(1)}%</span>
                </div>
                <small class="text-muted">Evaluated supply chain risks and inventory levels</small>
            </div>
            
            <div class="timeline-step completed">
                <div class="d-flex justify-content-between">
                    <span><i class="fas fa-robot me-2"></i>RL Decision Making</span>
                    <span class="badge bg-info">${timeline.final_decision.toUpperCase()}</span>
                </div>
                <small class="text-muted">Reinforcement learning agent made shipping decision</small>
            </div>
            
            <div class="timeline-step completed">
                <div class="d-flex justify-content-between">
                    <span><i class="fas fa-cogs me-2"></i>Orchestrator Processing</span>
                    <span class="badge bg-primary">${timeline.processing_time}ms</span>
                </div>
                <small class="text-muted">Coordinated all models and updated feature store</small>
            </div>
        </div>
        
        <div class="mt-3">
            <h6>AI Scores Summary</h6>
            <div class="row text-center">
                <div class="col-3">
                    <div class="ai-score-small">${(timeline.intent_score * 100).toFixed(0)}%</div>
                    <small>Intent Score</small>
                </div>
                <div class="col-3">
                    <div class="ai-score-small">${(timeline.urgency_score * 100).toFixed(0)}%</div>
                    <small>Urgency Level</small>
                </div>
                <div class="col-3">
                    <div class="ai-score-small">${(timeline.confidence * 100).toFixed(0)}%</div>
                    <small>Confidence</small>
                </div>
                <div class="col-3">
                    <div class="ai-score-small">${timeline.processing_time}ms</div>
                    <small>Processing Time</small>
                </div>
            </div>
        </div>
    `;
}

// Load system metrics
async function loadSystemMetrics() {
    try {
        const response = await fetch('/api/admin/metrics');
        const result = await response.json();
        
        if (result.success) {
            updateSystemMetrics(result.metrics);
        } else {
            updateSystemMetrics({
                total_orders: orders.length,
                total_revenue: orders.reduce((sum, order) => sum + order.total, 0),
                ai_accuracy: 0.75,
                avg_processing_time: 3.5
            });
        }
    } catch (error) {
        console.error('Error loading metrics:', error);
        updateSystemMetrics({
            total_orders: orders.length,
            total_revenue: orders.reduce((sum, order) => sum + order.total, 0),
            ai_accuracy: 0.75,
            avg_processing_time: 3.5
        });
    }
}

// Update system metrics
function updateSystemMetrics(metrics) {
    document.getElementById('totalOrders').textContent = metrics.total_orders || 0;
    document.getElementById('totalRevenue').textContent = `$${(metrics.total_revenue || 0).toFixed(2)}`;
    document.getElementById('aiAccuracy').textContent = `${((metrics.ai_accuracy || 0) * 100).toFixed(1)}%`;
    document.getElementById('avgProcessing').textContent = `${(metrics.avg_processing_time || 0).toFixed(1)}ms`;
}

// Load AI pipeline status
async function loadAIPipelineStatus() {
    try {
        const response = await fetch('/api/pipeline/status');
        const result = await response.json();
        
        if (result.success) {
            updateAIPipelineStatus(result.pipeline_status);
        }
    } catch (error) {
        console.error('Error loading AI pipeline status:', error);
    }
}

// Update AI pipeline status
function updateAIPipelineStatus(status) {
    // Update scores
    document.getElementById('intentScore').textContent = (status.intent_transformer.accuracy || 0.69).toFixed(2);
    document.getElementById('urgencyScore').textContent = (status.intent_transformer.urgency || 0.65).toFixed(2);
    document.getElementById('riskScore').textContent = (status.gnn_module.risk || 0.25).toFixed(2);
    document.getElementById('confidenceScore').textContent = (status.rl_agent.confidence || 0.78).toFixed(2);
    
    // Update model statuses
    updateModelStatus('intent', status.intent_transformer);
    updateModelStatus('gnn', status.gnn_module);
    updateModelStatus('rl', status.rl_agent);
    updateModelStatus('orchestrator', status.orchestrator);
}

// Update model status
function updateModelStatus(model, data) {
    const statusElement = document.getElementById(`${model}Status`);
    const accuracyElement = document.getElementById(`${model}Accuracy`) || document.getElementById(`${model}MAE`) || document.getElementById(`${model}Improvement`);
    const processingElement = document.getElementById(`${model}ProcessingTime`);
    const progressElement = document.getElementById(`${model}Progress`);
    
    if (statusElement) {
        statusElement.textContent = data.status || 'Online';
        statusElement.className = `model-status status-${data.status || 'online'}`;
    }
    
    if (accuracyElement) {
        accuracyElement.textContent = data.accuracy || data.mae || data.improvement || 'N/A';
    }
    
    if (processingElement) {
        processingElement.textContent = `${data.processing_time_ms || 0}ms`;
    }
    
    if (progressElement) {
        const progress = data.accuracy || (1 - data.mae) || data.improvement || 0.8;
        progressElement.style.width = `${(progress * 100).toFixed(1)}%`;
    }
}

// Create charts
function createCharts() {
    createActivityChart();
    createPerformanceChart();
    createDecisionChart();
}

// Create activity chart
function createActivityChart() {
    const data = [
        {
            x: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            y: [5, 12, 8, 25, 18, 15],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Orders',
            line: { color: '#1f77b4', width: 3 }
        },
        {
            x: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            y: [2, 8, 5, 18, 12, 10],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'AI Processing',
            line: { color: '#ff7f0e', width: 3 }
        }
    ];
    
    const layout = {
        title: '',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Count' },
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif' }
    };
    
    Plotly.newPlot('activityChart', data, layout, {responsive: true});
}

// Create performance chart
function createPerformanceChart() {
    const data = [
        {
            x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            y: [65, 68, 72, 69, 75, 78],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Intent Accuracy',
            line: { color: '#1f77b4', width: 3 }
        },
        {
            x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            y: [70, 73, 76, 74, 78, 82],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'GNN Performance',
            line: { color: '#ff7f0e', width: 3 }
        },
        {
            x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            y: [60, 65, 68, 72, 75, 78],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'RL Efficiency',
            line: { color: '#2ca02c', width: 3 }
        }
    ];
    
    const layout = {
        title: '',
        xaxis: { title: 'Month' },
        yaxis: { title: 'Performance %' },
        margin: { t: 20, r: 20, b: 40, l: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif' }
    };
    
    Plotly.newPlot('performanceChart', data, layout, {responsive: true});
}

// Create decision chart
function createDecisionChart() {
    const data = [{
        values: [45, 30, 25],
        labels: ['Standard', 'Expedited', 'Priority'],
        type: 'pie',
        hole: 0.4,
        marker: { colors: ['#1f77b4', '#ff7f0e', '#2ca02c'] }
    }];
    
    const layout = {
        title: '',
        margin: { t: 20, r: 20, b: 20, l: 20 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Segoe UI, sans-serif' }
    };
    
    Plotly.newPlot('decisionChart', data, layout, {responsive: true});
}

// Show section
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show selected section
    document.getElementById(`${sectionName}-section`).style.display = 'block';
    
    // Update navigation
    document.querySelectorAll('.list-group-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.classList.add('active');
    
    currentSection = sectionName;
    
    // Load section-specific data
    if (sectionName === 'orders') {
        loadOrders();
    } else if (sectionName === 'ai-pipeline') {
        loadAIPipelineStatus();
    } else if (sectionName === 'analytics') {
        createCharts();
    }
}

// Filter orders
function filterOrders() {
    const statusFilter = document.getElementById('statusFilter').value;
    const dateFilter = document.getElementById('dateFilter').value;
    
    let filteredOrders = orders;
    
    if (statusFilter !== 'all') {
        filteredOrders = filteredOrders.filter(order => order.status === statusFilter);
    }
    
    if (dateFilter) {
        const filterDate = new Date(dateFilter);
        filteredOrders = filteredOrders.filter(order => {
            const orderDate = new Date(order.created_at);
            return orderDate.toDateString() === filterDate.toDateString();
        });
    }
    
    displayOrders(filteredOrders);
}

// Refresh orders
function refreshOrders() {
    loadOrders();
}

// Save settings
function saveSettings() {
    const intentThreshold = document.getElementById('intentThreshold').value;
    const riskThreshold = document.getElementById('riskThreshold').value;
    
    // Send settings to API
    fetch('/api/admin/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            intent_threshold: parseFloat(intentThreshold),
            risk_threshold: parseFloat(riskThreshold)
        })
    }).then(response => response.json())
    .then(result => {
        if (result.success) {
            alert('Settings saved successfully!');
        } else {
            alert('Failed to save settings: ' + result.error);
        }
    }).catch(error => {
        console.error('Error saving settings:', error);
        alert('Error saving settings. Please try again.');
    });
}

// Start real-time updates
function startRealtimeUpdates() {
    realtimeInterval = setInterval(() => {
        if (currentSection === 'overview') {
            loadSystemMetrics();
        } else if (currentSection === 'ai-pipeline') {
            loadAIPipelineStatus();
        }
    }, 5000); // Update every 5 seconds
}

// Stop real-time updates
function stopRealtimeUpdates() {
    if (realtimeInterval) {
        clearInterval(realtimeInterval);
    }
}

// Logout
function logout() {
    localStorage.removeItem('currentUser');
    window.location.href = 'enhanced_index.html';
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopRealtimeUpdates();
});

