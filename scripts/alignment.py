import numpy as np
import open3d as o3d
import copy
from sklearn.decomposition import PCA

def preprocess_point_cloud(pcd, voxel_size):
    """ Downsamples and computes normals and FPFH features for a point cloud. """
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def align_point_cloud_to_standard(pcd):
    """ Aligns a point cloud using PCA to ensure X=right, Y=up, Z=forward. """
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # Compute PCA to find the principal axes
    pca = PCA(n_components=3)
    pca.fit(points_centered)
    
    # PCA components are the new axes directions
    rotation_matrix = pca.components_.T  # Transpose to get correct rotation

    print("PCA Alignment Matrix (Columns = New Basis Vectors):")
    print(rotation_matrix)

    # Create a transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = -rotation_matrix @ centroid  # Move to origin

    # Apply transformation
    pcd_aligned = copy.deepcopy(pcd)
    pcd_aligned.transform(transform_matrix)
    
    return pcd_aligned, transform_matrix

def detect_height_axis(target):
    """ Detects which axis (X, Y, or Z) represents the height of the target point cloud. """
    bbox = target.get_axis_aligned_bounding_box()
    dims = bbox.max_bound - bbox.min_bound  # Compute dimensions along X, Y, Z

    print(f"Bounding Box Dimensions (X, Y, Z): {dims}")
    
    # Find the axis with the largest dimension (assuming it's height)
    height_axis = np.argmax(dims)
    axis_names = ["X", "Y", "Z"]
    print(f"Detected height is along {axis_names[height_axis]}-axis")

    return height_axis, dims[height_axis]

def scale_target_to_correct_height(target, desired_height):
    """ Aligns and scales the target point cloud along the detected height axis. """
    # Step 1: Align the point cloud to Open3D coordinate system
    target_aligned, alignment_transform = align_point_cloud_to_standard(target)
    
    # Step 2: Detect the correct height axis
    height_axis, current_height = detect_height_axis(target_aligned)
    
    # Step 3: Compute scaling factor
    scale_factor = desired_height / current_height
    print(f"Scaling target along {['X', 'Y', 'Z'][height_axis]} axis by factor: {scale_factor:.3f}")

    # Step 4: Apply the scaling transformation
    scaling_matrix = np.eye(4)
    scaling_matrix[height_axis, height_axis] = scale_factor
    
    target_scaled = copy.deepcopy(target_aligned)
    target_scaled.transform(scaling_matrix)
    
    return target_scaled

def prepare_dataset(voxel_size, desired_height):
    """ Reads, aligns, scales, and preprocesses source and target point clouds. """
    source = o3d.io.read_point_cloud("/home/kolla/mannequin_detection/extracted_clouds/fullbody/converted_point_cloud.ply")
    target = o3d.io.read_point_cloud("/home/kolla/mannequin_detection/reconstructed_data/segmented.ply")
    
    print(f"Source points: {len(source.points)}")
    print(f"Target points: {len(target.points)}")

    # Align and scale target before registration
    target_scaled = scale_target_to_correct_height(target, desired_height)

    # Visualize source and target before registration
    print(":: Visualizing Source and Scaled Target Before Registration...")
    source.paint_uniform_color([1, 0, 0])              # Red for source
    target_scaled.paint_uniform_color([0, 1, 0])       # Green for target

    o3d.visualization.draw([
        {"name": "source", "geometry": source},
        {"name": "target_scaled", "geometry": target_scaled},
        {"name": "Coordinate Frame", "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)}
    ], show_ui=True, show_skybox=False)

    # Preprocess both clouds
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_scaled, voxel_size)

    return source, target_scaled, source_down, target_down, source_fpfh, target_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """ Runs Fast Global Registration (FGR) to align target with source. """
    distance_threshold = voxel_size * 0.5
    print(f":: Running FGR with distance threshold {distance_threshold}")
    
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        target_down, source_down, target_fpfh, source_fpfh,  # Swapped order
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    print(f":: FGR Registration completed")
    print(f"Fitness: {result.fitness}")
    print(f"Inlier RMSE: {result.inlier_rmse}")
    
    return result

# Set voxel size and mannequin height
voxel_size = 0.1
desired_height = 1.8  # Mannequin height in meters

# Prepare dataset with scaled target
source, target_scaled, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, desired_height)

# Execute registration with the scaled target
result_fast = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

# Apply transformation to the scaled target
target_transformed = copy.deepcopy(target_scaled)
target_transformed.transform(result_fast.transformation)

# Paint point clouds for final visualization
source.paint_uniform_color([1, 0, 0])              # Red for source
target_transformed.paint_uniform_color([0, 1, 0])  # Green for transformed target

# Visualize the final registered result
print(":: Visualizing Final Registration Result...")
o3d.visualization.draw([
    {"name": "source", "geometry": source},
    {"name": "target_transformed", "geometry": target_transformed},
    {"name": "Coordinate Frame", "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)}
], show_ui=True, show_skybox=False)
