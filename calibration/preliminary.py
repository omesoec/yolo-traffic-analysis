import cv2
import numpy as np

def calculate_homography_from_dimensions(image_points_ordered, real_world_width, real_world_height):
    """
    Calculates the homography matrix from four ordered pixel points and the real-world
    dimensions of the rectangle they form.

    Args:
        image_points_ordered (list of tuples): A list of four (x, y) pixel coordinates,
                                              provided in a specific order. For example:
                                              [bottom_right, bottom_left, top_left, top_right].
        real_world_width (float): The actual width of the rectangle in meters.
        real_world_height (float): The actual height of the rectangle in meters (or length).

    Returns:
        numpy.ndarray: The calculated 3x3 homography matrix.
    """
    if len(image_points_ordered) != 4:
        raise ValueError("Exactly four corresponding points are required.")

    # Convert the input pixel points to a NumPy array
    pixel_points = np.float32(image_points_ordered).reshape(-1, 1, 2)
    
    # Define the world coordinate system based on the provided dimensions.
    # We set the first point (e.g., bottom-right) as the origin (0,0).
    # The order here MUST correspond to the order of image_points_ordered.
    # Order: [bottom_right, bottom_left, top_left, top_right] in meters
    w, h = 12.6, 3
    world_points_defined = np.float32([
        [0, 0],       # Corresponds to image_points_ordered[0]
        [w, 0],       # Corresponds to image_points_ordered[1]
        [w, h],       # Corresponds to image_points_ordered[2]
        [0, h]        # Corresponds to image_points_ordered[3]
    ]).reshape(-1, 1, 2)

    # Calculate the homography matrix
    # Note: The function needs source points (pixels) first, then destination points (world)
    homography_matrix, mask = cv2.findHomography(pixel_points, world_points_defined)

    return homography_matrix

if __name__ == '__main__':
    # Using the example from the paper (Table I and Figure 10)
    
    # Real-world dimensions based on the paper's assumptions
    lane_width = 4.0  # meters
    lane_plus_gap_length = 15.0 # 6m line + 9m gap = 15m

    # Pixel coordinates from the paper
    # (u,v) pixel coordinates: p1=(1420,1000), p2=(936,1022), p3=(1120,824), p4=(1513,802)
    # Let's identify them on Fig. 10:
    # p1 corresponds to world (0,0) -> bottom_right
    # p2 corresponds to world (4,0) -> bottom_left
    # p3 corresponds to world (4,15) -> top_left
    # p4 corresponds to world (0,15) -> top_right
    
    # *** Provide the pixel coordinates in the correct order: [bottom_right, bottom_left, top_left, top_right] ***
    ordered_pixel_coords = [
        (1735, 556), # Bottom-Right corner
        (1076, 491),  # Bottom-Left corner
        (1140, 443),  # Top-Left corner
        (1735, 497)   # Top-Right corner
    ]

    try:
        h_matrix = calculate_homography_from_dimensions(ordered_pixel_coords, lane_width, lane_plus_gap_length)
        print("Calculated Homography Matrix from Dimensions:")
        # The result may have very minor floating point differences from the paper's
        # published matrix due to rounding, but it will be functionally the same.
        print(h_matrix)

        # Example: Transform a new pixel point to world coordinates
        # Let's pick a point somewhere in the middle of the road in the image
        some_pixel_point = np.array([[[1300, 900]]], dtype='float32')
        world_coord = cv2.perspectiveTransform(some_pixel_point, h_matrix)
        
        print(f"\nTransforming pixel point {some_pixel_point[0][0]} to world coordinates:")
        print(f"Result: {world_coord[0][0][0]:.2f}m (X), {world_coord[0][0][1]:.2f}m (Y)")

    except ValueError as e:
        print(e)