# Pipeline Design 🛢️

This application solves a pipeline design and flow optimization problem. 🔧

[Try the live demo](https://pipeline-design.fly.dev/)

The application:
- 🗺️ Downloads Simple Maps data for city locations 
- 📏 Computes a distance matrix using the haversine formula
- 🌳 Computes the Minimum Spanning Tree (MST) to connect cities using Kruskal's algorithm 
- ⚡ Optimizes flows across cities to minimize unmet demand
- 📊 Displays metrics and a Plotly map in a Streamlit app 
- Can be deployed to fly.io 🚀
## Data Sources

City location data is provided by [Simple Maps World Cities Database](https://simplemaps.com/data/world-cities) (Basic v1.77) 🌍. 