"""
City Selector Component for EV Station Dashboard
===============================================

This component provides a reusable city selection interface that can be easily
integrated into your existing Streamlit dashboard.

Usage:
    from city_selector_component import create_city_selector, load_city_data, create_city_map
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from streamlit_folium import st_folium
import sys
import os

# Add parent directory to path for optimization module
sys.path.append('..')
from optimization_module import EVStationOptimizer, create_demand_scores_from_ml_predictions

def create_city_selector():
    """
    Create a city selection interface in the sidebar.
    
    Returns:
    --------
    str: Selected city name
    """
    st.sidebar.header("🏙️ City Selection")
    
    available_cities = [
        'Saarbrucken', 'Berlin', 'Munich', 'Stuttgart', 
        'Frankfurt', 'Karlsruhe', 'Trier', 'Mainz', 'Kaiserslautern'
    ]
    
    selected_city = st.sidebar.selectbox(
        "Select City for Analysis",
        available_cities,
        index=0,
        help="Choose a city to analyze for optimal EV charging station placement"
    )
    
    return selected_city

def create_optimization_controls():
    """
    Create optimization parameter controls in the sidebar.
    
    Returns:
    --------
    tuple: (budget, station_cost, min_distance)
    """
    st.sidebar.header("🔧 Optimization Parameters")
    
    budget = st.sidebar.slider(
        "Budget (€)",
        min_value=100000,
        max_value=2000000,
        value=500000,
        step=50000,
        format="€%d"
    )
    
    station_cost = st.sidebar.slider(
        "Station Cost (€)",
        min_value=10000,
        max_value=100000,
        value=25000,
        step=5000,
        format="€%d"
    )
    
    min_distance = st.sidebar.slider(
        "Min Distance (m)",
        min_value=100,
        max_value=1000,
        value=300,
        step=50
    )
    
    max_stations = budget // station_cost
    st.sidebar.metric("Max Stations", max_stations)
    
    return budget, station_cost, min_distance

def load_city_data(city_name):
    """
    Load data for a specific city.
    
    Parameters:
    -----------
    city_name : str
        Name of the city to load data for
    
    Returns:
    --------
    GeoDataFrame or None: City data if found, None otherwise
    """
    try:
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
    except Exception as e:
        st.error(f"Error loading data for {city_name}: {str(e)}")
        return None

def create_ml_predictions(city_gdf):
    """
    Create ML predictions based on city features.
    
    Parameters:
    -----------
    city_gdf : GeoDataFrame
        City data with features
    
    Returns:
    --------
    tuple: (ml_predictions, features_df)
    """
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
    """
    Run optimization for the selected city.
    
    Parameters:
    -----------
    city_gdf : GeoDataFrame
        City data
    ml_predictions : array
        ML predictions
    features_df : DataFrame
        Feature data
    budget : float
        Budget constraint
    min_distance : float
        Minimum distance constraint
    station_cost : float
        Cost per station
    
    Returns:
    --------
    tuple: (optimized_locations, opt_results, demand_scores)
    """
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

def create_city_map(city_gdf, ml_predictions, optimized_locations=None, opt_results=None, 
                   show_existing=True, show_predicted=True, show_optimized=True):
    """
    Create an interactive map for the selected city.
    
    Parameters:
    -----------
    city_gdf : GeoDataFrame
        City data
    ml_predictions : array
        ML predictions
    optimized_locations : GeoDataFrame, optional
        Optimized station locations
    opt_results : dict, optional
        Optimization results
    show_existing : bool
        Show existing EV stations
    show_predicted : bool
        Show predicted EV stations
    show_optimized : bool
        Show optimized EV stations
    
    Returns:
    --------
    folium.Map: Interactive map
    """
    # Get map center
    center_lat = city_gdf.geometry.centroid.y.mean()
    center_lon = city_gdf.geometry.centroid.x.mean()
    
    # Create base map
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
            popup=f"Grid Cell {idx}"
        ).add_to(city_map)
    
    # Add existing EV stations
    if show_existing:
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
    if show_predicted:
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
    if show_optimized and optimized_locations is not None and len(optimized_locations) > 0:
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
    
    return city_map

def display_optimization_results(opt_results, optimized_locations, city_name):
    """
    Display optimization results in a formatted way.
    
    Parameters:
    -----------
    opt_results : dict
        Optimization results
    optimized_locations : GeoDataFrame
        Optimized station locations
    city_name : str
        Name of the city
    """
    if opt_results is None:
        st.warning("No optimization results available")
        return
    
    st.subheader("📊 Optimization Results")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stations", opt_results['num_stations'])
    
    with col2:
        st.metric("Total Cost", f"€{opt_results['total_cost']:,}")
    
    with col3:
        st.metric("Budget Used", f"{opt_results['budget_utilization']:.1f}%")
    
    with col4:
        st.metric("Demand Coverage", f"{opt_results['objective_value']:.2f}")
    
    # Results table
    if optimized_locations is not None and len(optimized_locations) > 0:
        st.subheader("🎯 Selected Station Locations")
        
        # Create a summary table
        summary_data = {
            'Station ID': range(len(optimized_locations)),
            'Latitude': [loc.geometry.centroid.y for loc in optimized_locations.geometry],
            'Longitude': [loc.geometry.centroid.x for loc in optimized_locations.geometry],
            'Optimization Score': optimized_locations['optimization_score'].values
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download button
        csv = optimized_locations.to_csv(index=False)
        st.download_button(
            label="📥 Download Optimization Results",
            data=csv,
            file_name=f"{city_name.lower()}_optimized_stations.csv",
            mime="text/csv"
        )

# Example usage function
def run_city_analysis():
    """
    Example function showing how to use the city selector component.
    """
    # Create city selector
    selected_city = create_city_selector()
    
    # Create optimization controls
    budget, station_cost, min_distance = create_optimization_controls()
    
    # Load city data
    city_gdf = load_city_data(selected_city)
    
    if city_gdf is not None:
        # Create ML predictions
        ml_predictions, features_df = create_ml_predictions(city_gdf)
        
        # Run optimization
        with st.spinner("Running optimization analysis..."):
            optimized_locations, opt_results, demand_scores = run_optimization(
                city_gdf, ml_predictions, features_df, budget, min_distance, station_cost
            )
        
        # Create and display map
        city_map = create_city_map(
            city_gdf, ml_predictions, optimized_locations, opt_results
        )
        
        st_folium(city_map, width=800, height=600, returned_objects=[], zoom=12)
        
        # Display results
        display_optimization_results(opt_results, optimized_locations, selected_city)

if __name__ == "__main__":
    st.title("🚗 EV Station Placement Analysis")
    run_city_analysis()
