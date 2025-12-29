# FEATURE_NAMES = [
#     "1d_position_x", # 1d node static
#     "1d_position_y", # 1d node static
#     "1d_length",  # 1d edge static
#     "2d_position_x",  # 2d node static
#     "2d_position_y",  # 2d node static
#     "area", # 2d node static
#     "roughness", # 2d node static
#     "elevation", # 2d node static
#     "aspect", # 2d node static
#     "curvature", # 2d node static
#     "flow_accumulation", # 2d node static
#     "relative_position_x",  # 2d edge static
#     "relative_position_y",  # 2d edge static
#     "face_length",  # 2d edge static
#     "2d_length",  # 2d edge static
#     "slope",  # 2d node static
#     "inlet_flow", # 1d node dynamic
#     "1d_flow", # 1d edge dynamic
#     "1d_velocity", # 1d edge dynamic
#     "rainfall", # 2d node dynamic
#     # "water_level", # Excluded to prevent data leakage
#     "water_volume", # 2d node dynamic
#     "2d_flow", # 2d edge dynamic
#     "2d_velocity", # 2d edge dynamic
# ]

# STATIC_FEATURES = [
#     "1d_position_x", "1d_position_y", "1d_length",
#     "2d_position_x", "2d_position_y", "area", "roughness",
#     "elevation", "aspect", "curvature", "flow_accumulation", "slope",
#     "relative_position_x", "relative_position_y", "face_length", "2d_length"
# ]

# DYNAMIC_FEATURES = [
#     "inlet_flow", "1d_flow", "1d_velocity",
#     "2d_flow", "2d_velocity", "rainfall", "water_volume"
# ]

FEATURE_NAMES = [
    # 1D Node Features
    "1d_position_x",      # 1d node static
    "1d_position_y",      # 1d node static
    "inlet_flow",         # 1d node dynamic
    
    # 2D Node Features
    "2d_position_x",      # 2d node static
    "2d_position_y",      # 2d node static
    "area",               # 2d node static
    "roughness",          # 2d node static
    "elevation",          # 2d node static
    "aspect",             # 2d node static
    "curvature",          # 2d node static
    "flow_accumulation",  # 2d node static
    "slope",              # 2d node static
    "rainfall",           # 2d node dynamic
    "water_volume",       # 2d node dynamic
    
    # Note: "water_level" excluded to prevent data leakage
]

STATIC_FEATURES = [
    "1d_position_x", "1d_position_y",
    "2d_position_x", "2d_position_y", "area", "roughness",
    "elevation", "aspect", "curvature", "flow_accumulation", "slope"
]

DYNAMIC_FEATURES = [
    "inlet_flow", "rainfall", "water_volume"
]