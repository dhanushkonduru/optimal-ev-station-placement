# City Selection Integration Guide

This guide shows you how to add city selection and optimization features to your existing EV station dashboard.

## 🚀 Quick Integration

### Option 1: Use the Enhanced Dashboard (Recommended)

Replace your current `app.py` with `enhanced_app.py`:

```bash
cd Dashboard
cp app.py app_original.py  # Backup your original
cp enhanced_app.py app.py  # Use the enhanced version
streamlit run app.py
```

### Option 2: Add City Selection to Your Existing Dashboard

Add these lines to your existing `app.py`:

```python
# Add at the top with other imports
from city_selector_component import create_city_selector, load_city_data, create_ml_predictions, run_optimization

# Add in your sidebar (around line 40)
selected_city = create_city_selector()

# Replace your data loading section with:
city_gdf = load_city_data(selected_city)
if city_gdf is not None:
    ml_predictions, features_df = create_ml_predictions(city_gdf)
    
    # Add optimization controls
    budget = st.sidebar.slider("Budget (€)", 100000, 2000000, 500000, 50000)
    station_cost = st.sidebar.slider("Station Cost (€)", 10000, 100000, 25000, 5000)
    min_distance = st.sidebar.slider("Min Distance (m)", 100, 1000, 300, 50)
    
    # Run optimization
    optimized_locations, opt_results, demand_scores = run_optimization(
        city_gdf, ml_predictions, features_df, budget, min_distance, station_cost
    )
```

## 🎯 Features Added

### 1. City Selection
- Dropdown to choose from 9 German cities
- Automatic data loading for selected city
- Error handling for missing data

### 2. Optimization Controls
- Budget slider (€100k - €2M)
- Station cost slider (€10k - €100k)
- Minimum distance slider (100m - 1km)
- Real-time calculation of max stations

### 3. Enhanced Map Display
- **Red circles**: Existing EV stations
- **Blue circles**: ML-predicted locations (top 20%)
- **Green circles**: Optimized locations from linear programming
- **Gray background**: All grid cells
- **Orange markers**: Infrastructure overlays

### 4. Optimization Results
- Number of stations selected
- Total cost and budget utilization
- Demand coverage achieved
- Downloadable CSV results

## 🔧 Customization Options

### Change Available Cities
Edit the `available_cities` list in `city_selector_component.py`:

```python
available_cities = [
    'Saarbrucken', 'Berlin', 'Munich', 'Stuttgart', 
    'Frankfurt', 'Karlsruhe', 'Trier', 'Mainz', 'Kaiserslautern',
    'YourCity'  # Add your city here
]
```

### Modify Optimization Parameters
Adjust the slider ranges in `create_optimization_controls()`:

```python
budget = st.sidebar.slider(
    "Budget (€)",
    min_value=50000,    # Change minimum
    max_value=5000000,  # Change maximum
    value=500000,       # Change default
    step=25000          # Change step size
)
```

### Add New Map Layers
Extend the `create_city_map()` function to add new visualization layers.

## 📊 Data Requirements

The optimization requires your data to have these columns:
- `geometry`: Polygon geometries for grid cells
- `city`: City names
- `EV_stations`: Existing EV station counts
- `population`: Population density
- Infrastructure columns: `restaurant`, `commercial`, `parking_space`, etc.

## 🚨 Troubleshooting

### Common Issues:

1. **"No data found for city"**
   - Check city name spelling in your data
   - Ensure the city exists in your dataset

2. **"Optimization failed"**
   - Check if CVXPY is installed: `pip install cvxpy`
   - Verify data has required columns

3. **Map not displaying**
   - Check if folium is installed: `pip install folium streamlit-folium`
   - Verify geometry column format

### Performance Tips:

1. **Large datasets**: Consider filtering data by city before processing
2. **Slow optimization**: Reduce the number of candidate locations
3. **Memory issues**: Process cities one at a time

## 🎨 Styling

The enhanced dashboard includes:
- Clean, modern interface
- Color-coded map markers
- Responsive layout
- Professional metrics display

## 📈 Next Steps

1. **Test the integration** with your data
2. **Customize the parameters** for your use case
3. **Add more cities** to the selection
4. **Extend the optimization** with additional constraints
5. **Integrate with your existing ML models**

## 🔗 Files Created

- `enhanced_app.py`: Complete enhanced dashboard
- `city_selector_component.py`: Reusable components
- `optimization_module.py`: Linear programming optimization
- `integration_guide.md`: This guide

## 💡 Pro Tips

1. **Start with the enhanced dashboard** to see all features
2. **Gradually integrate** components into your existing code
3. **Test with different cities** to ensure compatibility
4. **Save your original app.py** as a backup
5. **Customize the styling** to match your brand

Happy optimizing! 🚗⚡
