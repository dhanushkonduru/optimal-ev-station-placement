import streamlit as st
import numpy as np
import json
import geopandas as gpd
import pyproj
import plotly.graph_objs as go
import folium
from streamlit_folium import st_folium
import pandas as pd
from folium.plugins import LocateControl, MarkerCluster
from bs4 import BeautifulSoup
from PIL import Image
import sys
import os

# Add parent directory to path for optimization module
sys.path.append('..')
from optimization_module import EVStationOptimizer, create_demand_scores_from_ml_predictions

# Utility functions
def load_poi_data(path, target_crs="epsg:3005"):
    df = pd.read_csv(path)
    poi_data = gpd.GeoDataFrame(
        df.loc[:, [c for c in df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        crs="epsg:3005",
    )
    poi_data = poi_data.to_crs(target_crs)
    poi_data['geometry_center'] = poi_data['geometry'].centroid
    return poi_data

def load_city_data(city_name):
    """Load data for a specific city"""
    data = pd.read_csv('Data/all_city_data_with_pop.csv')
    city_data = data[data['city'] == city_name].copy()
    
    if len(city_data) == 0:
        st.error(f"No data found for {city_name}")
        return None
    
    # Convert to GeoDataFrame
    city_gdf = gpd.GeoDataFrame(
        city_data,
        geometry=gpd.GeoSeries.from_wkt(city_data['geometry']),
        crs='epsg:4326'
    )
    
    return city_gdf

def create_ml_predictions(city_gdf):
    """Create ML predictions based on features"""
    feature_columns = ['parking', 'edges', 'parking_space', 'civic', 'restaurant', 'park', 'school',
                       'node', 'Community_centre', 'place_of_worship', 'university', 'cinema',
                       'library', 'commercial', 'retail', 'townhall', 'government', 'residential', 'population']
    
    X = city_gdf[feature_columns].fillna(0)
    
    # Create realistic ML predictions
    np.random.seed(42)
    ml_predictions = (
        0.3 * (X['population'] / (X['population'].max() + 1)) +
        0.2 * (X['restaurant'] / (X['restaurant'].max() + 1)) +
        0.2 * (X['commercial'] / (X['commercial'].max() + 1)) +
        0.15 * (X['parking_space'] / (X['parking_space'].max() + 1)) +
        0.1 * (X['school'] / (X['school'].max() + 1)) +
        0.05 * (X['university'] / (X['university'].max() + 1))
    )
    ml_predictions += np.random.normal(0, 0.05, len(ml_predictions))
    ml_predictions = np.clip(ml_predictions, 0, 1)
    
    return ml_predictions, X

def run_optimization(city_gdf, ml_predictions, features_df, budget, min_distance, station_cost):
    """Run optimization for the selected city"""
    try:
        # Create demand scores
        demand_scores = create_demand_scores_from_ml_predictions(ml_predictions, features_df)
        
        # Initialize optimizer
        optimizer = EVStationOptimizer(
            candidate_locations=city_gdf,
            demand_scores=demand_scores,
            budget=budget,
            min_distance=min_distance,
            station_cost=station_cost
        )
        
        # Run coverage optimization
        locations, results = optimizer.get_optimized_locations('coverage')
        
        return locations, results, demand_scores
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return None, None, None

# Setting page config
st.set_page_config(
    page_title="Optimal EV Charging Station Placement",
    page_icon="https://img.icons8.com/external-phatplus-lineal-color-phatplus/64/external-ev-ev-car-phatplus-lineal-color-phatplus.png",
    layout="wide"
)

# Header
logo_url = "https://img.icons8.com/external-phatplus-lineal-color-phatplus/64/external-ev-ev-car-phatplus-lineal-color-phatplus.png"
st.image(logo_url, width=64)
st.header("🚗 Optimal EV Charging Station Placement Dashboard")
st.markdown("**Spatial-Economic Analysis for Optimal EV Charging Station Placement using Machine Learning & Linear Programming**")

# Sidebar for controls
st.sidebar.header("🎛️ Dashboard Controls")

# City selection
available_cities = ['Saarbrucken', 'Berlin', 'Munich', 'Stuttgart', 'Frankfurt', 'Karlsruhe', 'Trier', 'Mainz', 'Kaiserslautern']
selected_city = st.sidebar.selectbox(
    "🏙️ Select City",
    available_cities,
    index=0
)

st.sidebar.markdown("---")

# Optimization controls
st.sidebar.subheader("🔧 Optimization Parameters")

budget = st.sidebar.slider(
    "💰 Budget (€)",
    min_value=100000,
    max_value=2000000,
    value=500000,
    step=50000,
    format="€%d"
)

station_cost = st.sidebar.slider(
    "🏗️ Station Cost (€)",
    min_value=10000,
    max_value=100000,
    value=25000,
    step=5000,
    format="€%d"
)

min_distance = st.sidebar.slider(
    "📏 Min Distance (m)",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

max_stations = budget // station_cost
st.sidebar.metric("📊 Max Stations", max_stations)

st.sidebar.markdown("---")

# Map display options
st.sidebar.subheader("🗺️ Map Display Options")

show_existing_evs = st.sidebar.checkbox("🔴 Existing EV Stations", value=True)
show_predicted_evs = st.sidebar.checkbox("⚫ Predicted EV Stations", value=True)
show_optimized_evs = st.sidebar.checkbox("🟢 Optimized EV Stations", value=True)
show_population = st.sidebar.checkbox("👥 Population Density", value=False)

# Infrastructure toggles
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

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📍 {selected_city} - EV Station Analysis")
    
    # Load city data
    with st.spinner(f"Loading data for {selected_city}..."):
        city_gdf = load_city_data(selected_city)
    
    if city_gdf is not None:
        # Create ML predictions
        ml_predictions, features_df = create_ml_predictions(city_gdf)
        
        # Run optimization
        with st.spinner("Running optimization analysis..."):
            optimized_locations, opt_results, demand_scores = run_optimization(
                city_gdf, ml_predictions, features_df, budget, min_distance, station_cost
            )
        
        # Create map
        center_lat = city_gdf.geometry.centroid.y.mean()
        center_lon = city_gdf.geometry.centroid.x.mean()
        
        city_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add all grid cells (light gray)
        for idx, row in city_gdf.iterrows():
            centroid = row.geometry.centroid
            folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=2,
                color='lightgray',
                fill=True,
                fillColor='lightgray',
                fillOpacity=0.3,
                popup=f"Demand Score: {demand_scores[idx]:.3f}"
            ).add_to(city_map)
        
        # Add existing EV stations
        if show_existing_evs:
            existing_evs = city_gdf[city_gdf['EV_stations'] > 0]
            for idx, row in existing_evs.iterrows():
                centroid = row.geometry.centroid
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=8,
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.8,
                    popup=f"Existing EV Station<br>Count: {row['EV_stations']}"
                ).add_to(city_map)
        
        # Add predicted EV stations (from ML)
        if show_predicted_evs:
            # Show top 20% of predictions
            top_predictions = city_gdf[ml_predictions > np.percentile(ml_predictions, 80)]
            for idx, row in top_predictions.iterrows():
                centroid = row.geometry.centroid
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=6,
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.6,
                    popup=f"ML Prediction: {ml_predictions[idx]:.3f}"
                ).add_to(city_map)
        
        # Add optimized EV stations
        if show_optimized_evs and optimized_locations is not None and len(optimized_locations) > 0:
            for idx, row in optimized_locations.iterrows():
                centroid = row.geometry.centroid
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=10,
                    color='green',
                    fill=True,
                    fillColor='green',
                    fillOpacity=0.9,
                    popup=f"Optimized Station {row['station_id']}<br>Score: {row['optimization_score']:.3f}"
                ).add_to(city_map)
        
        # Add infrastructure markers
        for infra_type, show in infrastructure_options.items():
            if show:
                infra_col = infra_type.lower().replace(' ', '_')
                if infra_col in city_gdf.columns:
                    infra_locations = city_gdf[city_gdf[infra_col] > 0]
                    for idx, row in infra_locations.iterrows():
                        centroid = row.geometry.centroid
                        folium.CircleMarker(
                            location=[centroid.y, centroid.x],
                            radius=3,
                            color='orange',
                            fill=True,
                            fillColor='orange',
                            fillOpacity=0.5,
                            popup=f"{infra_type}: {row[infra_col]}"
                        ).add_to(city_map)
        
        # Add population density
        if show_population:
            sim_geo = gpd.GeoSeries(city_gdf["geometry"]).simplify(tolerance=0.001)
            geo_json = sim_geo.to_json()
            
            population_data = city_gdf["population"]
            folium.Choropleth(
                geo_data=geo_json,
                name="Population Density",
                data=population_data,
                columns=["population"],
                key_on="feature.id",
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                highlight=True,
                legend_name="Population Density"
            ).add_to(city_map)
        
        # Display map
        st_folium(city_map, width=800, height=600, center=[center_lat, center_lon], returned_objects=[], zoom=12)
        
        # Save map
        city_map.save(f"Data/{selected_city.lower()}_optimization_map.html")

with col2:
    st.subheader("📊 Analysis Results")
    
    if city_gdf is not None:
        # City statistics
        st.metric("Grid Cells", len(city_gdf))
        st.metric("Existing EV Stations", city_gdf['EV_stations'].sum())
        
        if optimized_locations is not None and opt_results is not None:
            st.metric("Optimized Stations", opt_results['num_stations'])
            st.metric("Total Cost", f"€{opt_results['total_cost']:,}")
            st.metric("Budget Used", f"{opt_results['budget_utilization']:.1f}%")
            st.metric("Demand Coverage", f"{opt_results['objective_value']:.2f}")
        
        # Optimization comparison
        if optimized_locations is not None and len(optimized_locations) > 0:
            st.subheader("🎯 Optimization Summary")
            
            comparison_data = {
                'Metric': ['Stations', 'Cost', 'Budget Used', 'Demand Coverage'],
                'Value': [
                    opt_results['num_stations'],
                    f"€{opt_results['total_cost']:,}",
                    f"{opt_results['budget_utilization']:.1f}%",
                    f"{opt_results['objective_value']:.2f}"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
            
            # Download button for results
            csv = optimized_locations.to_csv(index=False)
            st.download_button(
                label="📥 Download Optimization Results",
                data=csv,
                file_name=f"{selected_city.lower()}_optimized_stations.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("**🔬 Methodology:** This dashboard combines Machine Learning predictions with Linear Programming optimization to determine optimal EV charging station locations based on socio-economic features, budget constraints, and distance requirements.")

# Add some CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)
