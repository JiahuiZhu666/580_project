import numpy as np
import heapq
from collections import deque

def circular_obstacle_terrain_advanced(terrain, radius=1.0, height=-20.0, platform_size=1.0):
    """
    Advanced circular obstacle terrain with A* path planning
    """
    length_px, width_px = terrain.height_field_raw.shape
    radius_px = int(radius / terrain.horizontal_scale)
    
    # Initialize height map
    hmap = np.zeros_like(terrain.height_field_raw, dtype=np.float32)

    # Fixed positions for the two circular obstacles
    center1 = (int(length_px * 0.35), int(width_px * 0.3))
    center2 = (int(length_px * 0.65), int(width_px * 0.7))
    centers = [center1, center2]

    # Create obstacle map for path planning
    obstacle_map = np.zeros((length_px, width_px), dtype=bool)
    
    # Create circular pits and mark obstacles
    for (cx, cy) in centers:
        for i in range(cx - radius_px, cx + radius_px + 1):
            for j in range(cy - radius_px, cy + radius_px + 1):
                if 0 <= i < length_px and 0 <= j < width_px:
                    if (i - cx)**2 + (j - cy)**2 <= radius_px**2:
                        hmap[i, j] = height
                        obstacle_map[i, j] = True
                    # Add safety margin around obstacles
                    elif (i - cx)**2 + (j - cy)**2 <= (radius_px + 3)**2:
                        obstacle_map[i, j] = True

    # Set the height field
    terrain.height_field_raw[:, :] = hmap.astype(terrain.height_field_raw.dtype)

    # Define start and end points
    start_px = (int(length_px * 0.05), int(width_px * 0.05))
    end_px = (int(length_px * 0.95), int(width_px * 0.95))
    
    # Use A* algorithm to find optimal path
    path = astar_path_planning(obstacle_map, start_px, end_px)
    
    # Create goals along the path
    num_goals = 8
    if len(path) > 1:
        goals = create_goals_from_path(path, num_goals)
    else:
        # Fallback to simple interpolation if path planning fails
        goals = simple_path_goals(start_px, end_px, num_goals)
    
    # Convert pixel coordinates to world coordinates
    terrain.goals = goals * terrain.horizontal_scale
    terrain.idx = 21
    
    return terrain

def astar_path_planning(obstacle_map, start, goal):
    """
    A* pathfinding algorithm to navigate around obstacles
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    
    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        # 8-connected grid (including diagonal movement)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < obstacle_map.shape[0] and 
                    0 <= ny < obstacle_map.shape[1] and 
                    not obstacle_map[nx, ny]):
                    neighbors.append((nx, ny))
        return neighbors
    
    # A* algorithm
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
        
        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found, return empty list
    return []

def create_goals_from_path(path, num_goals):
    """
    Create evenly spaced goals along the found path
    """
    if len(path) <= num_goals:
        return np.array(path, dtype=np.float32)
    
    goals = np.zeros((num_goals, 2), dtype=np.float32)
    path_length = len(path)
    
    for i in range(num_goals):
        # Calculate the index along the path
        path_progress = i / (num_goals - 1)
        path_idx = int(path_progress * (path_length - 1))
        goals[i] = np.array(path[path_idx], dtype=np.float32)
    
    return goals

def simple_path_goals(start, end, num_goals):
    """
    Fallback: simple interpolation between start and end
    """
    goals = np.zeros((num_goals, 2), dtype=np.float32)
    for i in range(num_goals):
        alpha = i / (num_goals - 1)
        goals[i] = (1 - alpha) * np.array(start) + alpha * np.array(end)
    return goals

def circular_obstacle_terrain_with_multiple_paths(terrain, radius=1.0, height=-20.0, platform_size=1.0):
    """
    Version with multiple possible paths around obstacles
    """
    length_px, width_px = terrain.height_field_raw.shape
    radius_px = int(radius / terrain.horizontal_scale)
    
    # Initialize height map
    hmap = np.zeros_like(terrain.height_field_raw, dtype=np.float32)

    # Fixed positions for obstacles creating a corridor
    center1 = (int(length_px * 0.3), int(width_px * 0.4))
    center2 = (int(length_px * 0.7), int(width_px * 0.6))
    centers = [center1, center2]

    # Create circular pits
    for (cx, cy) in centers:
        for i in range(cx - radius_px, cx + radius_px + 1):
            for j in range(cy - radius_px, cy + radius_px + 1):
                if 0 <= i < length_px and 0 <= j < width_px:
                    if (i - cx)**2 + (j - cy)**2 <= radius_px**2:
                        hmap[i, j] = height

    terrain.height_field_raw[:, :] = hmap.astype(terrain.height_field_raw.dtype)

    # Create goals that follow an S-curve around obstacles
    num_goals = 8
    goals = np.zeros((num_goals, 2))
    
    # S-curve waypoints
    waypoints = [
        (int(length_px * 0.05), int(width_px * 0.2)),   # Start
        (int(length_px * 0.2), int(width_px * 0.15)),    # Approach first obstacle
        (int(length_px * 0.35), int(width_px * 0.1)),    # Around first obstacle (bottom)
        (int(length_px * 0.5), int(width_px * 0.3)),     # Between obstacles
        (int(length_px * 0.65), int(width_px * 0.9)),    # Around second obstacle (top)
        (int(length_px * 0.8), int(width_px * 0.85)),    # After second obstacle
        (int(length_px * 0.95), int(width_px * 0.8)),    # End
    ]
    
    # Smooth interpolation through waypoints
    total_segments = len(waypoints) - 1
    segment_length = num_goals / total_segments
    
    for i in range(num_goals):
        progress = i / (num_goals - 1)
        segment_idx = min(int(progress * total_segments), total_segments - 1)
        local_progress = (progress * total_segments) - segment_idx
        
        start_wp = np.array(waypoints[segment_idx])
        end_wp = np.array(waypoints[min(segment_idx + 1, len(waypoints) - 1)])
        
        goals[i] = (1 - local_progress) * start_wp + local_progress * end_wp
    
    terrain.goals = goals * terrain.horizontal_scale
    terrain.idx = 21
    
    return terrain

def visualize_path_planning(terrain, obstacle_map, path, radius=1.0):
    """
    Visualization helper for debugging path planning
    """
    length_px, width_px = terrain.height_field_raw.shape
    
    print(f"Terrain size: {length_px} x {width_px}")
    print(f"Obstacles detected: {np.sum(obstacle_map)} pixels")
    print(f"Path length: {len(path)} waypoints")
    
    if len(path) > 0:
        print(f"Path start: {path[0]}")
        print(f"Path end: {path[-1]}")
        
        # Calculate path efficiency
        direct_distance = np.sqrt((path[-1][0] - path[0][0])**2 + (path[-1][1] - path[0][1])**2)
        path_distance = sum(np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2) 
                           for i in range(len(path)-1))
        efficiency = direct_distance / path_distance if path_distance > 0 else 0
        print(f"Path efficiency: {efficiency:.2f} (1.0 = straight line)")
    
    print("Goals (world coordinates):")
    for i, goal in enumerate(terrain.goals):
        print(f"  Goal {i+1}: ({goal[0]:.2f}, {goal[1]:.2f})")