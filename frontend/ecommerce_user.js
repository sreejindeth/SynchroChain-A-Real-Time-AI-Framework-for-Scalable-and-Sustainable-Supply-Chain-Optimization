// E-commerce User Interface JavaScript
let products = [];
let cart = [];
let currentCategory = 'all';
let currentUser = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadProducts();
    loadUserSession();
    setupEventListeners();
});

// Load user session
function loadUserSession() {
    const userData = localStorage.getItem('currentUser');
    if (userData) {
        currentUser = JSON.parse(userData);
        console.log('User logged in:', currentUser);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Search functionality
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchProducts();
        }
    });
}

// Load products from API
async function loadProducts() {
    try {
        const response = await fetch('/api/products');
        const result = await response.json();
        
        if (result.success) {
            products = result.products;
            displayProducts(products);
        } else {
            console.error('Failed to load products:', result.error);
            loadMockProducts();
        }
    } catch (error) {
        console.error('Error loading products:', error);
        loadMockProducts();
    }
}

// Load mock products if API fails
function loadMockProducts() {
    products = [
        {
            id: 1,
            name: 'Smart Watch',
            category: 'Electronics',
            price: 327.75,
            description: 'Advanced smartwatch with health monitoring',
            image: 'fas fa-clock',
            stock: 50
        },
        {
            id: 2,
            name: 'Wireless Headphones',
            category: 'Electronics',
            price: 199.99,
            description: 'Premium wireless headphones with noise cancellation',
            image: 'fas fa-headphones',
            stock: 30
        },
        {
            id: 3,
            name: 'Sports Jacket',
            category: 'Clothing',
            price: 89.99,
            description: 'High-performance sports jacket for all weather',
            image: 'fas fa-tshirt',
            stock: 25
        },
        {
            id: 4,
            name: 'Fitness Tracker',
            category: 'Sports',
            price: 149.99,
            description: 'Track your fitness goals with precision',
            image: 'fas fa-running',
            stock: 40
        },
        {
            id: 5,
            name: 'Home Speaker',
            category: 'Home',
            price: 79.99,
            description: 'Smart home speaker with voice control',
            image: 'fas fa-volume-up',
            stock: 35
        },
        {
            id: 6,
            name: 'Gaming Mouse',
            category: 'Electronics',
            price: 59.99,
            description: 'High-precision gaming mouse with RGB lighting',
            image: 'fas fa-mouse',
            stock: 60
        }
    ];
    displayProducts(products);
}

// Display products in grid
function displayProducts(productsToShow) {
    const grid = document.getElementById('productsGrid');
    grid.innerHTML = '';
    
    productsToShow.forEach(product => {
        const productCard = createProductCard(product);
        grid.appendChild(productCard);
    });
}

// Create product card element
function createProductCard(product) {
    const col = document.createElement('div');
    col.className = 'col-md-4 col-lg-3 mb-4';
    
    col.innerHTML = `
        <div class="card product-card h-100">
            <div class="product-image">
                <i class="${product.image}"></i>
            </div>
            <div class="card-body d-flex flex-column">
                <h6 class="card-title">${product.name}</h6>
                <p class="text-muted small">${product.category}</p>
                <p class="card-text flex-grow-1">${product.description}</p>
                <div class="d-flex justify-content-between align-items-center">
                    <h5 class="text-primary mb-0">$${product.price}</h5>
                    <div class="btn-group">
                        <button class="btn btn-outline-primary btn-sm" onclick="viewProduct(${product.id})">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-primary btn-sm" onclick="addToCart(${product.id})">
                            <i class="fas fa-cart-plus"></i>
                        </button>
                    </div>
                </div>
                <small class="text-muted">Stock: ${product.stock}</small>
            </div>
        </div>
    `;
    
    return col;
}

// Search products
function searchProducts() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const filteredProducts = products.filter(product => 
        product.name.toLowerCase().includes(searchTerm) ||
        product.description.toLowerCase().includes(searchTerm) ||
        product.category.toLowerCase().includes(searchTerm)
    );
    displayProducts(filteredProducts);
}

// Filter by category
function filterByCategory(category) {
    currentCategory = category;
    
    // Update active category
    document.querySelectorAll('.category-filter').forEach(filter => {
        filter.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Filter products
    const filteredProducts = category === 'all' ? 
        products : 
        products.filter(product => product.category === category);
    
    displayProducts(filteredProducts);
}

// View product details
function viewProduct(productId) {
    const product = products.find(p => p.id === productId);
    if (!product) return;
    
    document.getElementById('productModalTitle').textContent = product.name;
    document.getElementById('productModalName').textContent = product.name;
    document.getElementById('productModalCategory').textContent = product.category;
    document.getElementById('productModalPrice').textContent = `$${product.price}`;
    document.getElementById('productModalDescription').textContent = product.description;
    document.getElementById('productModalImage').innerHTML = `<i class="${product.image}"></i>`;
    document.getElementById('productQuantity').value = 1;
    
    new bootstrap.Modal(document.getElementById('productModal')).show();
}

// Add to cart
function addToCart(productId, quantity = 1) {
    const product = products.find(p => p.id === productId);
    if (!product) return;
    
    const existingItem = cart.find(item => item.id === productId);
    
    if (existingItem) {
        existingItem.quantity += quantity;
    } else {
        cart.push({
            ...product,
            quantity: quantity
        });
    }
    
    updateCartBadge();
    showCartNotification(product.name);
}

// Add to cart from modal
function addToCartFromModal() {
    const productId = parseInt(document.getElementById('productModalTitle').textContent);
    const quantity = parseInt(document.getElementById('productQuantity').value);
    
    // Find product by name since we don't have ID in modal
    const product = products.find(p => p.name === document.getElementById('productModalName').textContent);
    if (product) {
        addToCart(product.id, quantity);
        bootstrap.Modal.getInstance(document.getElementById('productModal')).hide();
    }
}

// Show cart notification
function showCartNotification(productName) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-success border-0 position-fixed top-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-check-circle me-2"></i>
                ${productName} added to cart!
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        document.body.removeChild(toast);
    });
}

// Toggle cart modal
function toggleCart() {
    updateCartDisplay();
    new bootstrap.Modal(document.getElementById('cartModal')).show();
}

// Update cart display
function updateCartDisplay() {
    const cartItems = document.getElementById('cartItems');
    const cartTotal = document.getElementById('cartTotal');
    
    if (cart.length === 0) {
        cartItems.innerHTML = '<p class="text-center text-muted">Your cart is empty</p>';
        cartTotal.textContent = '0.00';
        return;
    }
    
    cartItems.innerHTML = '';
    let total = 0;
    
    cart.forEach(item => {
        const itemTotal = item.price * item.quantity;
        total += itemTotal;
        
        const cartItem = document.createElement('div');
        cartItem.className = 'cart-item d-flex justify-content-between align-items-center';
        cartItem.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="${item.image} me-3"></i>
                <div>
                    <h6 class="mb-0">${item.name}</h6>
                    <small class="text-muted">$${item.price} x ${item.quantity}</small>
                </div>
            </div>
            <div class="d-flex align-items-center">
                <span class="me-3">$${itemTotal.toFixed(2)}</span>
                <button class="btn btn-sm btn-outline-danger" onclick="removeFromCart(${item.id})">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        cartItems.appendChild(cartItem);
    });
    
    cartTotal.textContent = total.toFixed(2);
}

// Remove from cart
function removeFromCart(productId) {
    cart = cart.filter(item => item.id !== productId);
    updateCartBadge();
    updateCartDisplay();
}

// Update cart badge
function updateCartBadge() {
    const totalItems = cart.reduce((sum, item) => sum + item.quantity, 0);
    document.getElementById('cartBadge').textContent = totalItems;
}

// Proceed to checkout
async function proceedToCheckout() {
    if (cart.length === 0) {
        alert('Your cart is empty!');
        return;
    }
    
    if (!currentUser) {
        alert('Please log in to proceed with checkout');
        return;
    }
    
    try {
        // Create order
        const orderData = {
            user_id: currentUser.id,
            items: cart,
            total: cart.reduce((sum, item) => sum + (item.price * item.quantity), 0)
        };
        
        const response = await fetch('/api/orders', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(orderData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Show order tracking
            showOrderTracking(result.order_id);
            
            // Clear cart
            cart = [];
            updateCartBadge();
            updateCartDisplay();
            
            // Close cart modal
            bootstrap.Modal.getInstance(document.getElementById('cartModal')).hide();
            
            // Track user behavior for AI
            trackUserBehavior('checkout', orderData);
        } else {
            alert('Order failed: ' + result.error);
        }
    } catch (error) {
        console.error('Checkout error:', error);
        alert('Checkout failed. Please try again.');
    }
}

// Show order tracking
function showOrderTracking(orderId) {
    document.getElementById('orderId').textContent = orderId;
    document.getElementById('orderTracking').style.display = 'block';
    
    // Simulate order processing
    let progress = 0;
    const progressBar = document.getElementById('orderProgress');
    
    const interval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress > 100) progress = 100;
        
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                document.getElementById('orderTracking').style.display = 'none';
            }, 3000);
        }
    }, 500);
}

// Track user behavior for AI analysis
function trackUserBehavior(action, data) {
    const behaviorData = {
        user_id: currentUser?.id || 'anonymous',
        action: action,
        timestamp: new Date().toISOString(),
        data: data
    };
    
    // Send to AI system
    fetch('/api/track-behavior', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(behaviorData)
    }).catch(error => {
        console.error('Behavior tracking error:', error);
    });
}

// Track search behavior
function searchProducts() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    
    // Track search behavior
    trackUserBehavior('search', { query: searchTerm });
    
    const filteredProducts = products.filter(product => 
        product.name.toLowerCase().includes(searchTerm) ||
        product.description.toLowerCase().includes(searchTerm) ||
        product.category.toLowerCase().includes(searchTerm)
    );
    displayProducts(filteredProducts);
}

// Track view behavior
function viewProduct(productId) {
    const product = products.find(p => p.id === productId);
    if (product) {
        trackUserBehavior('view', { product_id: productId, product_name: product.name });
    }
    
    // Show product modal
    document.getElementById('productModalTitle').textContent = product.name;
    document.getElementById('productModalName').textContent = product.name;
    document.getElementById('productModalCategory').textContent = product.category;
    document.getElementById('productModalPrice').textContent = `$${product.price}`;
    document.getElementById('productModalDescription').textContent = product.description;
    document.getElementById('productModalImage').innerHTML = `<i class="${product.image}"></i>`;
    document.getElementById('productQuantity').value = 1;
    
    new bootstrap.Modal(document.getElementById('productModal')).show();
}

