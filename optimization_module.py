"""
EV Station Placement Optimization Module
=======================================

This module provides linear programming optimization for EV charging station placement.
It integrates with the existing machine learning predictions to provide optimal
station placement considering budget, distance, and coverage constraints.

Author: AI Assistant
Date: 2024
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class EVStationOptimizer:
    """
    Linear Programming optimizer for EV charging station placement.
    
    This class solves the facility location problem to determine optimal
    EV charging station locations given budget constraints, distance requirements,
    and demand coverage objectives.
    """
    
    def __init__(self, candidate_locations, demand_scores, budget=1000000, 
                 min_distance=500, station_cost=50000):
        """
        Initialize the EV Station Optimizer.
        
        Parameters:
        -----------
        candidate_locations : GeoDataFrame
            Grid cells with ML predictions for potential EV station locations
        demand_scores : array-like
            Demand scores for each candidate location (from ML model)
        budget : float
            Total budget available for station installation
        min_distance : float
            Minimum distance between stations (in meters)
        station_cost : float
            Cost per EV charging station
        """
        self.candidate_locations = candidate_locations
        self.demand_scores = np.array(demand_scores)
        self.budget = budget
        self.min_distance = min_distance
        self.station_cost = station_cost
        self.n_candidates = len(candidate_locations)
        
        # Calculate distance matrix between all candidate locations
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self):
        """Calculate pairwise distances between all candidate locations."""
        # Get centroids of all grid cells
        centroids = self.candidate_locations.geometry.centroid
        coords = np.array([[point.x, point.y] for point in centroids])
        
        # Calculate pairwise distances
        distances = pdist(coords)
        distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def optimize_coverage(self):
        """
        Optimize for maximum demand coverage.
        
        This method solves the following optimization problem:
        Maximize: Σ(demand_i * x_i)
        Subject to:
        - Σ(cost * x_i) ≤ budget
        - x_i + x_j ≤ 1 for all i,j where distance(i,j) < min_distance
        - x_i ∈ {0,1} for all i
        
        Returns:
        --------
        dict: Optimization results including selected locations and objective value
        """
        # Decision variables: x[i] = 1 if station at location i, 0 otherwise
        x = cp.Variable(self.n_candidates, boolean=True)
        
        # Objective: Maximize total demand coverage
        objective = cp.Maximize(self.demand_scores @ x)
        
        # Constraints
        constraints = []
        
        # Budget constraint
        constraints.append(cp.sum(x) * self.station_cost <= self.budget)
        
        # Distance constraints: no two stations within min_distance
        for i in range(self.n_candidates):
            for j in range(i + 1, self.n_candidates):
                if self.distance_matrix[i, j] < self.min_distance:
                    constraints.append(x[i] + x[j] <= 1)
        
        # Binary constraints
        constraints.extend([x >= 0, x <= 1])
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            selected_indices = np.where(x.value > 0.5)[0]
            total_demand = np.sum(self.demand_scores[selected_indices])
            total_cost = len(selected_indices) * self.station_cost
            
            return {
                'status': problem.status,
                'selected_locations': selected_indices,
                'objective_value': total_demand,
                'total_cost': total_cost,
                'num_stations': len(selected_indices),
                'budget_utilization': total_cost / self.budget * 100
            }
        else:
            return {
                'status': problem.status,
                'selected_locations': [],
                'objective_value': 0,
                'total_cost': 0,
                'num_stations': 0,
                'budget_utilization': 0
            }
    
    def optimize_cost_efficiency(self):
        """
        Optimize for cost efficiency (demand per unit cost).
        
        This method solves a modified optimization problem that considers
        both demand coverage and cost efficiency.
        """
        # Decision variables
        x = cp.Variable(self.n_candidates, boolean=True)
        
        # Objective: Maximize demand per unit cost
        # We'll use a weighted objective that balances demand and cost
        demand_weight = 1.0
        cost_weight = 0.001  # Small weight to prefer lower costs
        
        objective = cp.Maximize(
            demand_weight * (self.demand_scores @ x) - 
            cost_weight * (cp.sum(x) * self.station_cost)
        )
        
        # Constraints
        constraints = []
        
        # Budget constraint
        constraints.append(cp.sum(x) * self.station_cost <= self.budget)
        
        # Distance constraints
        for i in range(self.n_candidates):
            for j in range(i + 1, self.n_candidates):
                if self.distance_matrix[i, j] < self.min_distance:
                    constraints.append(x[i] + x[j] <= 1)
        
        # Binary constraints
        constraints.extend([x >= 0, x <= 1])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            selected_indices = np.where(x.value > 0.5)[0]
            total_demand = np.sum(self.demand_scores[selected_indices])
            total_cost = len(selected_indices) * self.station_cost
            efficiency = total_demand / total_cost if total_cost > 0 else 0
            
            return {
                'status': problem.status,
                'selected_locations': selected_indices,
                'objective_value': total_demand,
                'total_cost': total_cost,
                'num_stations': len(selected_indices),
                'efficiency': efficiency,
                'budget_utilization': total_cost / self.budget * 100
            }
        else:
            return {
                'status': problem.status,
                'selected_locations': [],
                'objective_value': 0,
                'total_cost': 0,
                'num_stations': 0,
                'efficiency': 0,
                'budget_utilization': 0
            }
    
    def optimize_with_coverage_requirement(self, min_coverage=0.8):
        """
        Optimize with minimum coverage requirement.
        
        Parameters:
        -----------
        min_coverage : float
            Minimum fraction of total demand that must be covered
        """
        total_demand = np.sum(self.demand_scores)
        min_demand_coverage = min_coverage * total_demand
        
        # Decision variables
        x = cp.Variable(self.n_candidates, boolean=True)
        
        # Objective: Minimize total cost while meeting coverage requirement
        objective = cp.Minimize(cp.sum(x) * self.station_cost)
        
        # Constraints
        constraints = []
        
        # Coverage requirement
        constraints.append(self.demand_scores @ x >= min_demand_coverage)
        
        # Budget constraint
        constraints.append(cp.sum(x) * self.station_cost <= self.budget)
        
        # Distance constraints
        for i in range(self.n_candidates):
            for j in range(i + 1, self.n_candidates):
                if self.distance_matrix[i, j] < self.min_distance:
                    constraints.append(x[i] + x[j] <= 1)
        
        # Binary constraints
        constraints.extend([x >= 0, x <= 1])
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        if problem.status not in ["infeasible", "unbounded"]:
            selected_indices = np.where(x.value > 0.5)[0]
            total_demand = np.sum(self.demand_scores[selected_indices])
            total_cost = len(selected_indices) * self.station_cost
            coverage_achieved = total_demand / total_demand if total_demand > 0 else 0
            
            return {
                'status': problem.status,
                'selected_locations': selected_indices,
                'objective_value': total_cost,
                'total_demand_covered': total_demand,
                'total_cost': total_cost,
                'num_stations': len(selected_indices),
                'coverage_achieved': coverage_achieved,
                'budget_utilization': total_cost / self.budget * 100
            }
        else:
            return {
                'status': problem.status,
                'selected_locations': [],
                'objective_value': 0,
                'total_demand_covered': 0,
                'total_cost': 0,
                'num_stations': 0,
                'coverage_achieved': 0,
                'budget_utilization': 0
            }
    
    def get_optimized_locations(self, optimization_type='coverage'):
        """
        Get optimized station locations based on specified optimization type.
        
        Parameters:
        -----------
        optimization_type : str
            Type of optimization: 'coverage', 'efficiency', or 'coverage_requirement'
        
        Returns:
        --------
        GeoDataFrame: Selected locations with optimization results
        """
        if optimization_type == 'coverage':
            results = self.optimize_coverage()
        elif optimization_type == 'efficiency':
            results = self.optimize_cost_efficiency()
        elif optimization_type == 'coverage_requirement':
            results = self.optimize_with_coverage_requirement()
        else:
            raise ValueError("optimization_type must be 'coverage', 'efficiency', or 'coverage_requirement'")
        
        if results['status'] in ["infeasible", "unbounded"]:
            print(f"Optimization failed with status: {results['status']}")
            return gpd.GeoDataFrame()
        
        # Create GeoDataFrame with selected locations
        selected_locations = self.candidate_locations.iloc[results['selected_locations']].copy()
        selected_locations['optimization_score'] = self.demand_scores[results['selected_locations']]
        selected_locations['station_id'] = range(len(selected_locations))
        
        return selected_locations, results


def create_demand_scores_from_ml_predictions(ml_predictions, features_df):
    """
    Create demand scores from ML predictions and feature data.
    
    Parameters:
    -----------
    ml_predictions : array-like
        ML model predictions (probability scores)
    features_df : DataFrame
        Feature data used for ML predictions
    
    Returns:
    --------
    array: Demand scores for optimization
    """
    # Weight different features to create demand scores
    weights = {
        'population': 0.3,
        'restaurant': 0.15,
        'commercial': 0.15,
        'parking_space': 0.1,
        'school': 0.1,
        'university': 0.1,
        'retail': 0.1
    }
    
    # Calculate weighted demand scores
    demand_scores = np.zeros(len(features_df))
    
    for feature, weight in weights.items():
        if feature in features_df.columns:
            # Normalize feature values
            feature_values = features_df[feature].values
            if feature_values.max() > 0:
                normalized_values = feature_values / feature_values.max()
                demand_scores += weight * normalized_values
    
    # Combine with ML predictions
    ml_scores = np.array(ml_predictions)
    if ml_scores.max() > 0:
        ml_scores = ml_scores / ml_scores.max()
    
    # Final demand score: 70% ML prediction + 30% feature-based score
    final_scores = 0.7 * ml_scores + 0.3 * demand_scores
    
    return final_scores


def run_optimization_analysis(candidate_locations, ml_predictions, features_df, 
                            budget=1000000, min_distance=500, station_cost=50000):
    """
    Run comprehensive optimization analysis.
    
    Parameters:
    -----------
    candidate_locations : GeoDataFrame
        Grid cells with geometry information
    ml_predictions : array-like
        ML model predictions
    features_df : DataFrame
        Feature data
    budget : float
        Total budget
    min_distance : float
        Minimum distance between stations
    station_cost : float
        Cost per station
    
    Returns:
    --------
    dict: Results from all optimization approaches
    """
    # Create demand scores
    demand_scores = create_demand_scores_from_ml_predictions(ml_predictions, features_df)
    
    # Initialize optimizer
    optimizer = EVStationOptimizer(
        candidate_locations=candidate_locations,
        demand_scores=demand_scores,
        budget=budget,
        min_distance=min_distance,
        station_cost=station_cost
    )
    
    # Run different optimization approaches
    results = {}
    
    # 1. Maximize coverage
    results['coverage'] = optimizer.get_optimized_locations('coverage')
    
    # 2. Cost efficiency
    results['efficiency'] = optimizer.get_optimized_locations('efficiency')
    
    # 3. Coverage requirement (80% coverage)
    results['coverage_requirement'] = optimizer.get_optimized_locations('coverage_requirement')
    
    return results


if __name__ == "__main__":
    print("EV Station Optimization Module")
    print("=============================")
    print("This module provides linear programming optimization for EV charging station placement.")
    print("Use the EVStationOptimizer class to optimize station locations based on your requirements.")
