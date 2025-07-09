import numpy as np
import random
import math

def circular_obstacle_terrain(terrain, radius=1.0, height=-20.0, platform_size=1.0):
    ########################################
    ############ IMPROVED ENV ##############
    ########################################

    length_px, width_px = terrain.height_field_raw.shape
    radius_px = int(radius / terrain.horizontal_scale)
    
    # Initialize height map
    hmap = np.zeros_like(terrain.height_field_raw, dtype=np.float32)

    # Fixed positions for the two circular obstacles
    # Place them strategically to create a meaningful navigation challenge
    center1 = (int(length_px * 0.35), int(width_px * 0.3))
    center2 = (int(length_px * 0.65), int(width_px * 0.7))
    centers = [center1, center2]

    # Create circular pits at fixed positions
    for (cx, cy) in centers:
        for i in range(cx - radius_px, cx + radius_px + 1):
            for j in range(cy - radius_px, cy + radius_px + 1):
                if 0 <= i < length_px and 0 <= j < width_px:
                    if (i - cx)**2 + (j - cy)**2 <= radius_px**2:
                        hmap[i, j] = height

    # Set the height field
    terrain.height_field_raw[:, :] = hmap.astype(terrain.height_field_raw.dtype)

    # Define start and end points that make sense for navigation
    start_px = np.array([int(length_px * 0.1), int(width_px * 0.1)])
    end_px = np.array([int(length_px * 0.9), int(width_px * 0.9)])
    
    # Create waypoints that navigate around the obstacles
    num_goals = 8
    goals = np.zeros((num_goals, 2))
    
    # Create a path that goes around the obstacles
    waypoints_px = [
        start_px,
        np.array([int(length_px * 0.25), int(width_px * 0.15)]),  # Before first obstacle
        np.array([int(length_px * 0.45), int(width_px * 0.05)]),  # Around first obstacle (top)
        np.array([int(length_px * 0.55), int(width_px * 0.45)]),  # Between obstacles
        np.array([int(length_px * 0.75), int(width_px * 0.5)]),   # Around second obstacle
        np.array([int(length_px * 0.85), int(width_px * 0.8)]),   # After second obstacle
        end_px
    ]
    
    # Interpolate between waypoints to create smooth path
    if len(waypoints_px) >= 2:
        total_segments = len(waypoints_px) - 1
        points_per_segment = num_goals // total_segments
        remaining_points = num_goals % total_segments
        
        goal_idx = 0
        for seg_idx in range(total_segments):
            start_point = waypoints_px[seg_idx]
            end_point = waypoints_px[seg_idx + 1]
            
            # Add extra point to last segment if there are remaining points
            segment_points = points_per_segment + (1 if seg_idx == total_segments - 1 and remaining_points > 0 else 0)
            
            for i in range(segment_points):
                if goal_idx < num_goals:
                    alpha = (i + 1) / (segment_points + 1) if segment_points > 0 else 0
                    interpolated_point = (1 - alpha) * start_point + alpha * end_point
                    goals[goal_idx] = interpolated_point
                    goal_idx += 1
        
        # Ensure we have exactly num_goals by filling any remaining with the end point
        while goal_idx < num_goals:
            goals[goal_idx] = end_px
            goal_idx += 1

    # Convert pixel coordinates to world coordinates
    terrain.goals = goals * terrain.horizontal_scale
    terrain.idx = 21
    
    return terrain

def visualize_terrain_plan(terrain, radius=1.0):
    """
    Helper function to visualize the terrain layout for debugging
    """
    length_px, width_px = terrain.height_field_raw.shape
    radius_px = int(radius / terrain.horizontal_scale)
    
    print(f"Terrain size: {length_px} x {width_px} pixels")
    print(f"Obstacle radius: {radius_px} pixels")
    
    # Print obstacle centers
    center1 = (int(length_px * 0.35), int(width_px * 0.3))
    center2 = (int(length_px * 0.65), int(width_px * 0.7))
    print(f"Obstacle 1 center: {center1}")
    print(f"Obstacle 2 center: {center2}")
    
    # Print goals in world coordinates
    print("Goals (world coordinates):")
    for i, goal in enumerate(terrain.goals):
        print(f"  Goal {i+1}: ({goal[0]:.2f}, {goal[1]:.2f})")