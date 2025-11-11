"""
User Simulation Module for SynchroChain AI Dashboard
E-commerce browsing demo with real-time AI predictions
"""
import streamlit as st
import random
from typing import Dict, List
from datetime import datetime

# Product catalog (8 products as mentioned in docs)
PRODUCTS = [
    {'id': 1, 'name': 'Laptop', 'category': 'Electronics', 'price': 999, 'stock': 50, 'image': '💻'},
    {'id': 2, 'name': 'Headphones', 'category': 'Electronics', 'price': 199, 'stock': 100, 'image': '🎧'},
    {'id': 3, 'name': 'Smartphone', 'category': 'Electronics', 'price': 699, 'stock': 75, 'image': '📱'},
    {'id': 4, 'name': 'Watch', 'category': 'Fashion', 'price': 299, 'stock': 60, 'image': '⌚'},
    {'id': 5, 'name': 'Shoes', 'category': 'Fashion', 'price': 149, 'stock': 120, 'image': '👟'},
    {'id': 6, 'name': 'Backpack', 'category': 'Fashion', 'price': 79, 'stock': 90, 'image': '🎒'},
    {'id': 7, 'name': 'Table Lamp', 'category': 'Home', 'price': 49, 'stock': 80, 'image': '💡'},
    {'id': 8, 'name': 'Coffee Maker', 'category': 'Home', 'price': 129, 'stock': 45, 'image': '☕'}
]

def get_product_by_id(product_id: int) -> Dict:
    """Get product details by ID."""
    for product in PRODUCTS:
        if product['id'] == product_id:
            return product
    return None

def track_action(action: str, product_id: int = None):
    """Track user action and update session state."""
    if 'user_session' not in st.session_state:
        st.session_state.user_session = []
    
    action_entry = {
        'action': action,
        'product_id': product_id,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.user_session.append(action_entry)

def create_order_context(cart_items: List[Dict]) -> Dict:
    """Create order context for GNN prediction from cart items."""
    if not cart_items:
        return {
            'category_name': 'Unknown',
            'order_quantity': 0,
            'order_value': 0,
            'shipping_mode': 'Standard Class',
            'product_price': 0,
            'order_item_quantity': 0,
            'order_item_discount_rate': 0.0,
            'order_item_profit_ratio': 0.1,
            'sales': 0,
            'days_for_shipment': 3,
            'market': 'Pacific Asia',
            'customer_segment': 'Consumer'
        }
    
    # Calculate order totals
    total_value = sum(item['price'] * item['quantity'] for item in cart_items)
    total_quantity = sum(item['quantity'] for item in cart_items)
    
    # Get most common category
    categories = [item['category'] for item in cart_items for _ in range(item['quantity'])]
    most_common_category = max(set(categories), key=categories.count) if categories else 'Unknown'
    
    # Get average price (this will be used as Product Price)
    avg_price = total_value / total_quantity if total_quantity > 0 else 0
    
    # Create order context matching GNN training features
    # Features: Product Price, Order Item Quantity, Order Item Discount Rate, 
    # Order Item Profit Ratio, Sales, Days for shipment (scheduled),
    # Customer Segment, Shipping Mode, Market
    
    # Add variation based on order characteristics
    # Shipping mode varies based on order value and urgency
    if total_value > 500:
        shipping_modes = ['First Class', 'Second Class', 'Standard Class']
        shipping_mode = random.choice(shipping_modes) if total_value > 1000 else 'Standard Class'
    else:
        shipping_mode = 'Standard Class'
    
    # Days for shipment varies based on shipping mode
    shipping_days_map = {
        'Standard Class': 5,
        'Second Class': 3,
        'First Class': 2,
        'Same Day': 1
    }
    days_for_shipment = shipping_days_map.get(shipping_mode, 3)
    
    # Discount varies with order value
    if total_value > 300:
        discount_rate = random.uniform(0.05, 0.25)
    else:
        discount_rate = random.uniform(0.0, 0.15)
    
    # Profit ratio varies
    profit_ratio = random.uniform(0.10, 0.30)
    
    # Customer segment varies
    customer_segments = ['Consumer', 'Corporate', 'Home Office']
    customer_segment = random.choice(customer_segments)
    
    # Market varies
    markets = ['Pacific Asia', 'Southeast Asia', 'South Asia', 'Oceania', 'Europe', 'Africa', 'Middle East', 'Latin America']
    market = random.choice(markets)
    
    return {
        'category_name': most_common_category,  # For display/context
        'order_quantity': total_quantity,
        'order_value': total_value,
        # GNN model features (matching training script feature_cols)
        'product_price': avg_price,  # Maps to 'Product Price'
        'order_item_quantity': total_quantity,  # Maps to 'Order Item Quantity'
        'order_item_discount_rate': discount_rate,  # Varied discount
        'order_item_profit_ratio': profit_ratio,  # Varied profit ratio
        'sales': total_value,  # Maps to 'Sales'
        'days_for_shipment': days_for_shipment,  # Varied based on shipping mode
        'customer_segment': customer_segment,  # Varied segment
        'shipping_mode': shipping_mode,  # Varied shipping mode
        'market': market,  # Varied market
        'Product Price': avg_price,  # Alternative key
        'Order Item Quantity': total_quantity,  # Alternative key
        'Order Item Discount Rate': discount_rate,  # Alternative key
        'Order Item Profit Ratio': profit_ratio,  # Alternative key
        'Sales': total_value,  # Alternative key
        'Days for shipment (scheduled)': days_for_shipment,  # Alternative key
        'Customer Segment': customer_segment,  # Alternative key
        'Shipping Mode': shipping_mode,  # Alternative key
        'Market': market  # Alternative key
    }

def display_ai_predictions(predictions: Dict):
    """Display AI predictions with visual indicators."""
    st.markdown("---")
    st.markdown("### 🤖 AI Predictions (Real-time)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent Score
        intent_score = predictions.get('intent_score', 0.0)
        st.markdown("#### 💡 Intent Score")
        st.progress(intent_score)
        st.metric("Score", f"{intent_score:.2%}", help="Probability of high purchase intent")
        
        # Urgency Level
        urgency = predictions.get('urgency', 0.0)
        st.markdown("#### ⚡ Urgency Level")
        st.progress(urgency)
        st.metric("Urgency", f"{urgency:.2%}", help="How soon the purchase might happen")
    
    with col2:
        # Delay Risk
        delay_risk = predictions.get('delay_risk', 0.0)
        st.markdown("#### ⚠️ Delay Risk")
        
        # Color-coded risk indicator
        if delay_risk < 0.3:
            risk_color = "🟢 Low"
            risk_level = "Low"
        elif delay_risk < 0.6:
            risk_color = "🟡 Medium"
            risk_level = "Medium"
        else:
            risk_color = "🔴 High"
            risk_level = "High"
        
        st.progress(delay_risk)
        st.metric("Risk Level", risk_color)
        st.caption(f"Risk Score: {delay_risk:.2%}")
        
        # PPO Decision
        rl_decision = predictions.get('rl_decision', {})
        st.markdown("#### 🎯 Supply Chain Decision")
        
        if isinstance(rl_decision, str):
            decision_text = rl_decision
            decision_details = ""
        else:
            # Handle Dict format
            shipping_mode = rl_decision.get('shipping_mode', 'Standard Class')
            priority = rl_decision.get('priority', 'Normal')
            pre_allocate = rl_decision.get('pre_allocate', False)
            restock = rl_decision.get('restock', False)
            
            # Get context for decision explanation
            state = predictions.get('state', {})
            delay_risk = state.get('delay_risk', 0)
            urgency = state.get('urgency', state.get('urgency_level', 0))
            intent_score = state.get('intent_score', 0)
            carbon_cost = state.get('carbon_cost', 0.5)
            order_value = state.get('order_value', 0)
            
            # Determine primary decision
            decision_text = "Normal Operation"
            decision_details = []
            
            if pre_allocate:
                decision_text = "Pre-allocate Inventory"
                decision_details.append(f"High intent ({intent_score:.1%}) and urgency ({urgency:.1%})")
            elif restock:
                decision_text = "Restock Items"
                decision_details.append("Low inventory detected")
            elif shipping_mode in ['Same Day', 'First Class']:
                decision_text = "Expedite Shipping"
                decision_details.append(f"Delay risk: {delay_risk:.1%}")
                if delay_risk > 0.6:
                    decision_details.append("High risk - expediting to prevent delays")
                elif urgency > 0.6:
                    decision_details.append("High urgency - priority shipping")
            elif shipping_mode == 'Second Class':
                decision_text = "Standard Shipping"
                decision_details.append("Balancing cost and delivery time")
            else:
                decision_text = "Normal Operation"
                decision_details.append("Standard processing")
            
            # Add context information
            if priority != 'Normal':
                decision_details.append(f"Priority: {priority}")
            
            # Carbon cost consideration
            if shipping_mode in ['Same Day', 'First Class']:
                if carbon_cost > 0.6:
                    decision_details.append(f"⚠️ Higher carbon footprint ({carbon_cost:.1%})")
                else:
                    decision_details.append(f"Carbon cost: {carbon_cost:.1%}")
            else:
                decision_details.append(f"Carbon cost: {carbon_cost:.1%}")
        
        # Display decision
        st.info(f"**{decision_text}**")
        if decision_details:
            st.caption(" | ".join(decision_details))

def display_product_catalog():
    """Display product catalog with action buttons."""
    st.markdown("### 🛍️ Product Catalog")
    
    # Category filter
    categories = ['All'] + list(set([p['category'] for p in PRODUCTS]))
    selected_category = st.selectbox("Filter by Category", categories)
    
    # Filter products
    if selected_category == 'All':
        display_products = PRODUCTS
    else:
        display_products = [p for p in PRODUCTS if p['category'] == selected_category]
    
    # Display products in grid
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, product in enumerate(display_products):
        col_idx = idx % num_cols
        with cols[col_idx]:
            st.markdown(f"#### {product['image']} {product['name']}")
            st.write(f"**Category:** {product['category']}")
            st.write(f"**Price:** ${product['price']}")
            st.write(f"**Stock:** {product['stock']} available")
            
            # Action buttons
            button_col1, button_col2 = st.columns(2)
            
            with button_col1:
                if st.button("View", key=f"view_{product['id']}", use_container_width=True):
                    track_action('view', product['id'])
                    st.rerun()
            
            with button_col2:
                if st.button("Add to Cart", key=f"add_{product['id']}", use_container_width=True):
                    track_action('add_to_cart', product['id'])
                    # Add to cart
                    if 'cart' not in st.session_state:
                        st.session_state.cart = []
                    
                    # Check if already in cart
                    cart_item = next((item for item in st.session_state.cart if item['id'] == product['id']), None)
                    if cart_item:
                        cart_item['quantity'] += 1
                    else:
                        st.session_state.cart.append({
                            'id': product['id'],
                            'name': product['name'],
                            'category': product['category'],
                            'price': product['price'],
                            'quantity': 1
                        })
                    st.rerun()
            
            # Like button
            if st.button("❤️ Like", key=f"like_{product['id']}", use_container_width=True):
                track_action('like', product['id'])
                st.rerun()
            
            st.markdown("---")

def display_shopping_cart():
    """Display shopping cart and checkout."""
    st.markdown("### 🛒 Shopping Cart")
    
    if 'cart' not in st.session_state or len(st.session_state.cart) == 0:
        st.info("Your cart is empty. Browse products and add items to see AI predictions!")
        return []
    
    # Display cart items
    total = 0
    for item in st.session_state.cart:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.write(f"**{item['name']}** ({item['category']})")
        
        with col2:
            st.write(f"${item['price']}")
        
        with col3:
            st.write(f"Qty: {item['quantity']}")
        
        with col4:
            if st.button("Remove", key=f"remove_{item['id']}", use_container_width=True):
                track_action('remove', item['id'])
                st.session_state.cart = [i for i in st.session_state.cart if i['id'] != item['id']]
                st.rerun()
        
        total += item['price'] * item['quantity']
        st.markdown("---")
    
    # Cart summary
    st.markdown(f"### **Total: ${total:.2f}**")
    
    # Checkout button
    if st.button("💳 Checkout", use_container_width=True, type="primary"):
        track_action('checkout')
        st.success("Order placed successfully!")
        # Clear cart after checkout
        if 'cart' in st.session_state:
            st.session_state.cart = []
        st.rerun()
    
    return st.session_state.cart

def display_session_history():
    """Display user session history."""
    st.markdown("### 📜 Session History")
    
    if 'user_session' not in st.session_state or len(st.session_state.user_session) == 0:
        st.info("No actions yet. Start browsing to see your session history!")
        return
    
    # Show last 10 actions
    recent_actions = st.session_state.user_session[-10:]
    
    for action in reversed(recent_actions):
        action_text = action['action']
        if action['product_id']:
            product = get_product_by_id(action['product_id'])
            if product:
                action_text += f" - {product['name']}"
        
        st.text(f"• {action_text} ({action['timestamp'][:19]})")
    
    # Clear session button
    if st.button("🗑️ Clear Session History"):
        st.session_state.user_session = []
        st.rerun()

def user_simulation():
    """Main user simulation function - e-commerce browsing demo."""
    st.markdown("# 🛒 E-Commerce Store Simulation")
    st.markdown("Browse products, add items to cart, and see real-time AI predictions!")
    st.markdown("---")
    
    # Initialize cart if not exists
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    
    # Initialize session if not exists
    if 'user_session' not in st.session_state:
        st.session_state.user_session = []
    
    # Get ModelManager (should be initialized in app.py)
    if 'model_manager' not in st.session_state or st.session_state.model_manager is None:
        st.error("⚠️ ModelManager not initialized. Please refresh the page.")
        return
    
    model_manager = st.session_state.model_manager
    
    # Main layout: Sidebar for cart, main area for products
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_main:
        display_product_catalog()
    
    with col_sidebar:
        cart_items = display_shopping_cart()
        
        # Real-time AI predictions
        if cart_items and len(st.session_state.user_session) > 0:
            # Create order context from cart (fresh each time with new random variation)
            order_context = create_order_context(cart_items)
            
            # Get user session actions as list of strings
            session_actions = [action['action'] for action in st.session_state.user_session]
            
            # Get AI predictions
            try:
                predictions = model_manager.process_user_session(session_actions, order_context)
                
                # Debug: Show actual prediction values in expander
                with st.expander("🔍 Debug: Prediction Values", expanded=False):
                    st.write("**State Values:**")
                    st.json(predictions.get('state', {}))
                    st.write("**Raw Predictions:**")
                    st.write(f"- Delay Risk: {predictions.get('delay_risk', 0):.4f}")
                    st.write(f"- Intent Score: {predictions.get('intent_score', 0):.4f}")
                    st.write(f"- Urgency: {predictions.get('urgency', 0):.4f}")
                    st.write(f"- Order Value (raw): {order_context.get('order_value', 0):.2f}")
                    st.write(f"- Order Value (normalized): {predictions.get('state', {}).get('order_value', 0):.4f}")
                    st.write("**RL Decision:**")
                    st.json(predictions.get('rl_decision', {}))
                
                # Display predictions
                display_ai_predictions(predictions)
            except Exception as e:
                st.error(f"Error getting predictions: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Session history
        st.markdown("---")
        display_session_history()

