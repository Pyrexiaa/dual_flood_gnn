FEATURE_NAMES = [
    # 1D Node Features
    "1d_position_x",      # 1d node static
    "1d_position_y",      # 1d node static
    # "depth",              # 1d node static
    # "invert_elevation",   # 1d node static
    # "surface_elevation",  # 1d node static
    # "base_area",          # 1d node static
    
    # 2D Node Features
    "2d_position_x",      # 2d node static
    "2d_position_y",      # 2d node static
    "area",               # 2d node static
    "roughness",          # 2d node static
    # "min_elevation",
    "elevation",          # 2d node static
    "aspect",             # 2d node static
    "curvature",          # 2d node static
    "flow_accumulation",  # 2d node static
    "slope",              # 2d node static
    "rainfall",           # 2d node dynamic
]

# Static Node features
STATIC_FEATURES = [
    "1d_position_x", "1d_position_y", "depth", "invert_elevation",
    "surface_elevation", "base_area",
    "2d_position_x", "2d_position_y", "area", "roughness",
    "elevation", "aspect", "curvature", "flow_accumulation", "slope"
]

DYNAMIC_FEATURES = ["rainfall"]