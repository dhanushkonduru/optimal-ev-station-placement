#!/usr/bin/env python3
"""
EV Station Optimization Demo
============================

This script demonstrates the linear programming optimization for EV charging station placement.
It shows how to integrate the optimization module with your existing ML predictions.

Usage:
    python demo_optimization.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from optimization_module import EVStationOptimizer, create_demand_scores_from_ml_predictions
import warnings
warnings.filterwarnings('ignore')


def main():
    print("🚗 EV Station Placement Optimization Demo")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading data...")
    data = pd.read_csv('data/processed/all_city_data_with_pop.csv')
    saarbrucken_data = data[data['city'] == 'Saarbrucken'].copy()
    
    # Convert to GeoDataFrame
    saarbrucken_gdf = gpd.GeoDataFrame(
        saarbrucken_data,
        geometry=gpd.GeoSeries.from_wkt(saarbrucken_data['geometry']),
        crs='epsg:4326'
    )
    
    print(f"✅ Loaded {len(saarbrucken_gdf)} grid cells for Saarbrücken")
    
    # Create feature matrix
    feature_columns = ['parking', 'edges', 'parking_space', 'civic', 'restaurant', 'park', 'school',
                       'node', 'Community_centre', 'place_of_worship', 'university', 'cinema',
                       'library', 'commercial', 'retail', 'townhall', 'government', 'residential', 'population']
    
    X = saarbrucken_gdf[feature_columns].fillna(0)
    
    # Create realistic ML predictions
    print("\n🤖 Creating ML predictions...")
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
    
    print(f"✅ ML predictions created (range: {ml_predictions.min():.3f} to {ml_predictions.max():.3f})")
    
    # Create demand scores
    print("\n📈 Creating demand scores...")
    demand_scores = create_demand_scores_from_ml_predictions(ml_predictions, X)
    print(f"✅ Demand scores created (range: {demand_scores.min():.3f} to {demand_scores.max():.3f})")
    
    # Set optimization parameters
    BUDGET = 500000  # 500k budget
    MIN_DISTANCE = 300  # 300m minimum distance
    STATION_COST = 25000  # 25k per station
    
    print(f"\n💰 Optimization Parameters:")
    print(f"   Budget: €{BUDGET:,}")
    print(f"   Station Cost: €{STATION_COST:,}")
    print(f"   Min Distance: {MIN_DISTANCE}m")
    print(f"   Max Stations: {BUDGET // STATION_COST}")
    
    # Initialize optimizer
    print("\n🔧 Initializing optimizer...")
    optimizer = EVStationOptimizer(
        candidate_locations=saarbrucken_gdf,
        demand_scores=demand_scores,
        budget=BUDGET,
        min_distance=MIN_DISTANCE,
        station_cost=STATION_COST
    )
    
    # Run different optimization strategies
    print("\n🎯 Running optimization strategies...")
    
    strategies = ['coverage', 'efficiency', 'coverage_requirement']
    results = {}
    
    for strategy in strategies:
        print(f"\n   Running {strategy} optimization...")
        try:
            locations, opt_results = optimizer.get_optimized_locations(strategy)
            results[strategy] = (locations, opt_results)
            
            print(f"   ✅ {strategy.title()}: {opt_results['num_stations']} stations, "
                  f"€{opt_results['total_cost']:,} cost, "
                  f"{opt_results['budget_utilization']:.1f}% budget used")
        except Exception as e:
            print(f"   ❌ {strategy.title()}: Failed - {str(e)}")
    
    # Display results comparison
    print("\n📊 OPTIMIZATION RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Stations':<8} {'Cost (€)':<12} {'Budget %':<10} {'Objective':<12}")
    print("-" * 60)
    
    for strategy, (locations, opt_results) in results.items():
        if len(opt_results) > 0:
            print(f"{strategy.replace('_', ' ').title():<20} "
                  f"{opt_results['num_stations']:<8} "
                  f"{opt_results['total_cost']:<12,} "
                  f"{opt_results['budget_utilization']:<10.1f} "
                  f"{opt_results.get('objective_value', 0):<12.2f}")
    
    # Budget sensitivity analysis
    print("\n📈 BUDGET SENSITIVITY ANALYSIS")
    print("=" * 50)
    budgets = [250000, 500000, 750000, 1000000]
    
    print(f"{'Budget (€)':<12} {'Stations':<8} {'Cost (€)':<12} {'Utilization %':<15}")
    print("-" * 50)
    
    for budget in budgets:
        temp_optimizer = EVStationOptimizer(
            candidate_locations=saarbrucken_gdf,
            demand_scores=demand_scores,
            budget=budget,
            min_distance=MIN_DISTANCE,
            station_cost=STATION_COST
        )
        
        try:
            _, temp_results = temp_optimizer.get_optimized_locations('coverage')
            print(f"{budget:<12,} {temp_results['num_stations']:<8} "
                  f"{temp_results['total_cost']:<12,} {temp_results['budget_utilization']:<15.1f}")
        except:
            print(f"{budget:<12,} {'Error':<8} {'Error':<12} {'Error':<15}")
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save coverage optimization results
    if 'coverage' in results:
        locations, _ = results['coverage']
        if len(locations) > 0:
            locations.to_csv('data/processed/saarbrucken_optimized_stations_coverage.csv', index=False)
            print("✅ Coverage optimization results saved")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run the full notebook: notebooks/optimization_integration.ipynb")
    print("2. Integrate optimization into your dashboard")
    print("3. Experiment with different budget and distance constraints")


if __name__ == "__main__":
    main()
