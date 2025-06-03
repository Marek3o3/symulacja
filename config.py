# config.py
"""
Configuration file for satellite constellation simulation.
Edit this file to change satellites, stations, map, and simulation settings.
"""

from datetime import datetime, timedelta
from skyfield.api import utc

# --- Simulation mode ---
# Choose 'MODIS', 'WorldView-3', or 'Both'
simulation_mode = 'WorldView-3'

# --- Earth parameters ---
R_earth_km = 6371
mu_km3_s2 = 398600.4418

# --- Satellite definitions (add more as needed) ---
satellites = {
    "MODIS_Terra": {
        "altitude_km": 617,
        "inclination_deg": 98,
        "fov_deg": 110,  # Effective FOV for 2330km swath from 705km altitude
        "color": "blue",
        "tle_line1": "1 25994U 99068A   25141.50000000  .00000714  00000-0  39998-4 0  9991",
        "tle_line2": "2 25994  98.2041 100.6060 0001200  80.5935 279.5400 14.57107008270001"
    },
    "WorldView-3": {
        "altitude_km": 617,
        "inclination_deg": 98,
        "fov_deg": 1.2,  # For ~13.1 km swath at 617 km
        "color": "red",
        "tle_line1": "1 40115U 14045A   25141.50000000  .00000714  00000-0  39998-4 0  9992",
        "tle_line2": "2 40115  97.8743  47.2925 0001529  71.0323  289.1058 14.83390144560002"
    }
}

# --- Time configuration ---
start_datetime = datetime(2025, 5, 21, 0, 0, 0, tzinfo=utc)
simulation_duration_hours = 4  # 24h analysis
time_step_seconds = 60

# --- Reception Stations ---
reception_stations = {
    "Poznan_Station": {"lat": 52.4064, "lon": 16.9252, "marker": "o", "color": "green"},
    "Berlin_Station": {"lat": 52.5200, "lon": 13.4050, "marker": "s", "color": "purple"},
}

# --- Map configuration ---
use_custom_map = True  # Set to True to use your custom map
custom_map_path = 'custom_map.png'  # Place your PNG/GeoTIFF in the same folder
custom_map_extent = [-180, 180, -90, 90]  # [lon_min, lon_max, lat_min, lat_max]

# --- Animation parameters ---
animation_fps = 10
animation_bitrate = 1800
fov_history_length = 5  # Number of previous FOVs to show with transparency
output_video_filename = "dual_satellite_animation.mp4"

# --- Coverage analysis ---
latitude_limit = 66.73  # Exclude coverage above/below +/- this latitude (e.g., 80 deg) 