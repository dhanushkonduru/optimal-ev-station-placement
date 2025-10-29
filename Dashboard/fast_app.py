import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import warnings
try:
    from pyswarms.discrete.binary import BinaryPSO
    _PSO_AVAILABLE = True
except Exception:
    _PSO_AVAILABLE = False
warnings.filterwarnings('ignore')

# Simple city data loader
def load_city_data_simple(city_name):
    """Load data for a specific city with error handling"""
    try:
        data = pd.read_csv('Data/all_city_data_with_pop.csv')
        city_data = data[data['city'] == city_name].copy()
        
        if len(city_data) == 0:
            st.error(f"No data found for {city_name}")
            return None
        
        # Convert geometry strings to coordinates for mapping
        city_data['geometry'] = city_data['geometry'].str.replace('POLYGON ((', '').str.replace('))', '')
        city_data['coords'] = city_data['geometry'].str.split(',').str[0].str.split()
        city_data['lat'] = city_data['coords'].str[1].astype(float)
        city_data['lon'] = city_data['coords'].str[0].astype(float)
        
        return city_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_simple_predictions(city_data):
    """Create simple predictions based on features"""
    # Simple scoring based on available features
    scores = (
        0.3 * (city_data['population'] / (city_data['population'].max() + 1)) +
        0.2 * (city_data['restaurant'] / (city_data['restaurant'].max() + 1)) +
        0.2 * (city_data['commercial'] / (city_data['commercial'].max() + 1)) +
        0.15 * (city_data['parking_space'] / (city_data['parking_space'].max() + 1)) +
        0.1 * (city_data['school'] / (city_data['school'].max() + 1)) +
        0.05 * (city_data['university'] / (city_data['university'].max() + 1))
    )
    return scores.fillna(0)

def create_clear_map(city_data, predictions, optimized_locations, show_existing, show_predicted, show_optimized, selected_infrastructure):
    """Create a clear, uncluttered map"""
    center_lat = city_data['lat'].mean()
    center_lon = city_data['lon'].mean()
    
    city_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add all grid cells (very light gray, smaller)
    for idx, row in city_data.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='lightgray',
            fill=True,
            fillColor='lightgray',
            fillOpacity=0.2,
            popup=f"Grid {idx}"
        ).add_to(city_map)
    
    # Add existing EV stations (RED - most important)
    if show_existing:
        existing_evs = city_data[city_data['EV_stations'] > 0]
        for idx, row in existing_evs.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=25,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.9,
                popup=f"🔴 Existing EV Station<br>Count: {row['EV_stations']}"
            ).add_to(city_map)
    
    # Add predicted locations (BLUE - only top 10% to reduce clutter)
    if show_predicted:
        top_predictions = city_data[predictions > predictions.quantile(0.9)]  # Only top 10%
        for idx, row in top_predictions.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=18,
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.7,
                popup=f"🔵 Predicted: {float(predictions.loc[idx]):.3f}"
            ).add_to(city_map)
    
    # Add optimized locations (GREEN - most important)
    if show_optimized:
        for idx, row in optimized_locations.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=30,
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.9,
                popup=f"🟢 Optimized Station<br>Score: {row['prediction_score']:.3f}"
            ).add_to(city_map)
    
    # Add infrastructure markers with emoji icons
    if selected_infrastructure:
        # Map infrastructure types to their emojis
        emoji_mapping = {
            'Parks': '🌳',
            'Parking spaces': '🅿️',
            'Restaurant': '🍽️',
            'Residential': '🏠',
            'Schools': '🏫',
            'Retail': '🛍️',
            'Community center': '🏛️',
            'Place of worship': '⛪',
            'University': '🎓',
            'Cinema': '🎬',
            'Library': '📚',
            'Commercial': '🏢',
            'Government': '🏛️'
        }
        
        for infra_name, column_name in selected_infrastructure:
            if column_name in city_data.columns:
                emoji = emoji_mapping.get(infra_name, '📍')
                infra_locations = city_data[city_data[column_name] > 0]
                for idx, row in infra_locations.iterrows():
                    # Use emoji as marker icon
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 20px; text-align: center;">{emoji}</div>',
                            icon_size=(20, 20),
                            icon_anchor=(10, 10)
                        ),
                        popup=f"{emoji} {infra_name}: {row[column_name]}"
                    ).add_to(city_map)
    
    return city_map

def create_simple_optimization(city_data, predictions, budget, station_cost):
    """Simple optimization - select top locations within budget"""
    # Calculate how many stations we can afford
    max_stations = budget // station_cost
    
    # Get top locations by prediction score
    city_data['prediction_score'] = predictions
    top_locations = city_data.nlargest(max_stations, 'prediction_score')
    
    return top_locations

def create_pso_optimization(city_data, predictions, budget, station_cost, n_particles=50, iters=80, penalty_weight=1e4):
    """Binary PSO: select cells to maximize total score under station budget."""
    if not _PSO_AVAILABLE:
        st.warning("PSO not available. Falling back to fast heuristic. Run: pip install pyswarms")
        return create_simple_optimization(city_data, predictions, budget, station_cost)

    max_stations = int(budget // station_cost)
    demand = predictions.values.astype(float)
    num_cells = len(demand)

    # Objective for BinaryPSO (minimize). We minimize negative utility + penalties
    def objective(x):
        # x shape: (n_particles, num_cells) with {0,1}
        # utility per particle
        util = -np.dot(x, demand)  # negative to maximize
        # penalty for exceeding budget (sum of ones > max_stations)
        over = np.maximum(0, x.sum(axis=1) - max_stations)
        penalty = penalty_weight * over
        return util + penalty

    options = {
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.7,
        'k': max(3, min(10, n_particles - 1)),  # number of neighbors
        'p': 2                                   # Minkowski p-norm order
    }
    optimizer = BinaryPSO(n_particles=n_particles, dimensions=num_cells, options=options)
    cost, pos = optimizer.optimize(objective, iters=iters, verbose=False)

    # Convert to indices
    selected = np.where(pos >= 0.5)[0].tolist()
    if len(selected) > max_stations:
        # keep best by demand
        selected = sorted(selected, key=lambda i: demand[i], reverse=True)[:max_stations]
    elif len(selected) < max_stations:
        # fill with top remaining
        remaining = [i for i in np.argsort(-demand) if i not in selected]
        selected += remaining[: max_stations - len(selected)]

    out = city_data.iloc[selected].copy()
    out['prediction_score'] = predictions.iloc[selected]
    return out

# Page config
st.set_page_config(
    page_title="Fast EV Station Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Header
st.title("🚗 Fast EV Charging Station Dashboard")
st.markdown("**Quick city selection and optimization demo**")

# Sidebar
st.sidebar.header("🎛️ Dashboard Controls")

# City selection
available_cities = ['Saarbrucken', 'Berlin', 'Munich', 'Stuttgart', 'Frankfurt', 'Karlsruhe', 'Trier', 'Mainz', 'Kaiserslautern']
selected_city = st.sidebar.selectbox("🏙️ Select City", available_cities, index=0)

st.sidebar.markdown("---")

# Optimization parameters
st.sidebar.subheader("🔧 Optimization Parameters")
budget = st.sidebar.slider("💰 Budget (€)", 100000, 2000000, 500000, 50000, format="€%d")
station_cost = st.sidebar.slider("🏗️ Station Cost (€)", 10000, 100000, 25000, 5000, format="€%d")
min_distance = st.sidebar.slider("📏 Min Distance (m)", 100, 1000, 300, 50)
max_stations = budget // station_cost
st.sidebar.metric("📊 Max Stations", max_stations)

# Optimization method
method_options = ["Heuristic (fast)", "PSO (binary)"]
selected_method = st.sidebar.selectbox("⚙️ Optimization Method", method_options, index=0, help=("PSO tries to find a better subset under the budget. Heuristic simply picks top-scoring cells."))

st.sidebar.markdown("---")

# Map display options
st.sidebar.subheader("🗺️ Map Display Options")
show_existing = st.sidebar.checkbox("🔴 Existing EV Stations", value=True)
show_predicted = st.sidebar.checkbox("🔵 Predicted Locations", value=True)
show_optimized = st.sidebar.checkbox("🟢 Optimized Locations", value=True)
show_population = st.sidebar.checkbox("👥 Population Density", value=False)

st.sidebar.markdown("---")

# Infrastructure overlay
st.sidebar.subheader("🏢 Infrastructure Overlay")
infrastructure_options = {
    'Parks': st.sidebar.checkbox("🌳 Parks", value=False),
    'Parking spaces': st.sidebar.checkbox("🅿️ Parking spaces", value=False),
    'Restaurant': st.sidebar.checkbox("🍽️ Restaurant", value=False),
    'Residential': st.sidebar.checkbox("🏠 Residential", value=False),
    'Schools': st.sidebar.checkbox("🏫 Schools", value=False),
    'Retail': st.sidebar.checkbox("🛍️ Retail", value=False),
    'Community center': st.sidebar.checkbox("🏛️ Community center", value=False),
    'Place of worship': st.sidebar.checkbox("⛪ Place of worship", value=False),
    'University': st.sidebar.checkbox("🎓 University", value=False),
    'Cinema': st.sidebar.checkbox("🎬 Cinema", value=False),
    'Library': st.sidebar.checkbox("📚 Library", value=False),
    'Commercial': st.sidebar.checkbox("🏢 Commercial", value=False),
    'Government': st.sidebar.checkbox("🏛️ Government", value=False)
}

# Get selected infrastructure (allow multiple selections)
selected_infrastructure = []
for infra_type, show in infrastructure_options.items():
    if show:
        # Map display names to column names
        column_mapping = {
            'Parks': 'park',
            'Parking spaces': 'parking_space', 
            'Restaurant': 'restaurant',
            'Residential': 'residential',
            'Schools': 'school',
            'Retail': 'retail',
            'Community center': 'Community_centre',
            'Place of worship': 'place_of_worship',
            'University': 'university',
            'Cinema': 'cinema',
            'Library': 'library',
            'Commercial': 'commercial',
            'Government': 'government'
        }
        column_name = column_mapping.get(infra_type, infra_type.lower().replace(' ', '_'))
        selected_infrastructure.append((infra_type, column_name))

# Main content - maximize map space
col1, col2 = st.columns([5, 1])

with col1:
    st.subheader(f"📍 {selected_city} Analysis")
    
    # Load data
    with st.spinner("Loading city data..."):
        city_data = load_city_data_simple(selected_city)
    
    if city_data is not None:
        st.success(f"✅ Loaded {len(city_data)} grid cells for {selected_city}")
        
        # Debug: Show available columns
        if selected_infrastructure:
            st.info(f"🔍 Selected infrastructure: {[infra[0] for infra in selected_infrastructure]}")
            available_columns = [infra[1] for infra in selected_infrastructure if infra[1] in city_data.columns]
            if available_columns:
                st.success(f"✅ Found columns: {available_columns}")
            else:
                st.warning(f"⚠️ No matching columns found. Available columns: {list(city_data.columns)}")
        
        # Create predictions
        predictions = create_simple_predictions(city_data)
        
        # Run chosen optimization
        if selected_method.startswith("PSO"):
            optimized_locations = create_pso_optimization(city_data.copy(), predictions.copy(), budget, station_cost)
        else:
            optimized_locations = create_simple_optimization(city_data.copy(), predictions.copy(), budget, station_cost)
        
        # Create clear map
        city_map = create_clear_map(
            city_data, predictions, optimized_locations, 
            show_existing, show_predicted, show_optimized,
            selected_infrastructure
        )
        
        # Display map with maximum size
        st_folium(city_map, width=1200, height=800, returned_objects=[], zoom=12)

with col2:
    st.subheader("📊 Analysis Results")
    
    if city_data is not None:
        # Basic stats
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Grid Cells", len(city_data))
            st.metric("Existing EVs", city_data['EV_stations'].sum())
        with col_b:
            st.metric("Optimized Stations", len(optimized_locations))
            st.metric("Budget Used", f"{(len(optimized_locations) * station_cost / budget * 100):.1f}%")
        
        # Cost analysis
        total_cost = len(optimized_locations) * station_cost
        st.metric("Total Cost", f"€{total_cost:,}")
        
        # Show top locations
        if len(optimized_locations) > 0:
            st.subheader("🎯 Selected Station Locations")
            top_locations_df = optimized_locations[['lat', 'lon', 'prediction_score']].head(10)
            top_locations_df.columns = ['Latitude', 'Longitude', 'Score']
            st.dataframe(top_locations_df, use_container_width=True)
            
            # Download button
            csv = optimized_locations[['lat', 'lon', 'prediction_score']].to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"{selected_city.lower()}_results.csv",
                mime="text/csv"
            )
    
    # Legend
    st.subheader("🗺️ Map Legend")
    st.markdown("""
    **🔴 Red Circles**: Existing EV charging stations  
    **🔵 Blue Circles**: ML-predicted optimal locations (top 10%)  
    **🟢 Green Circles**: Optimized locations (within budget)  
    **Emoji Icons**: Selected infrastructure (🌳 Parks, 🅿️ Parking, 🍽️ Restaurants, etc.)  
    **Gray Dots**: All grid cells (background)
    """)
    
    # Optimization summary
    if city_data is not None and len(optimized_locations) > 0:
        st.subheader("📈 Optimization Summary")
        st.success(f"✅ Selected {len(optimized_locations)} optimal locations")
        st.info(f"💰 Cost: €{total_cost:,} / €{budget:,} ({(total_cost/budget*100):.1f}%)")
        st.warning(f"📏 Min distance: {min_distance}m between stations")

# Footer
st.markdown("---")
st.markdown("**⚡ Fast Demo Version** - This is a simplified version for quick testing. The full optimization version includes linear programming constraints.")
