import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
import time
import argparse
from pathlib import Path
import os
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


def segment_multiple_planes(pcd, max_planes=2):
    """
    RANSAC拟合平面
    参数:
        pcd:点云对象
        max_planes:拟合次数
    返回：
        segments:返回平面列表
        remaining_pcd:剩余点云
    """
    segments = []
    remaining_pcd = pcd
    for _ in range(max_planes):
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=0.5,
            ransac_n=10,
            num_iterations=1000
        )
        if len(inliers) == 0:
            break
        segments.append((plane_model, remaining_pcd.select_by_index(inliers)))
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    return segments, remaining_pcd

def find_closest_to_z_axis(normals, normal=[0,0,1]):
    """
    找到最靠近z轴的平面法向量
    参数:
    normals:平面法向量列表
    返回:
    normals[closest_idx]:最靠近z轴的平面法向量
    """
    normals=np.array(normals)[:,0:3]
    z_axis = np.array(normal)
    # 归一化法向量（若未归一化）
    normals_normalized = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # 计算点积
    dot_products = np.dot(normals_normalized, z_axis)
    # 找到点积最大的索引
    closest_idx = np.argmax(dot_products)
    return normals[closest_idx]

def chunk_point_feature(rotated_points, all_features, max_points=2000):
    """
    随机采样划分点云和特征
    参数:
    rotated_points:点云
    all_features:特征
    max_points:最大划分容量
    返回:
    grouped_points:划分后的点云子集列表
    grouped_features:划分后的特征子集列表

    """
    N = rotated_points.shape[0]
    if N <= max_points:
        return [rotated_points],[all_features]

    remaining_indices = np.arange(N)
    np.random.shuffle(remaining_indices)  # 打乱索引
    grouped_points = []
    grouped_features = []
    
    while len(remaining_indices) > 0:
        chunk_size = min(max_points, len(remaining_indices))
        chunk_indices = remaining_indices[:chunk_size]
        chunk_p = rotated_points[chunk_indices]
        chunk_f = all_features[chunk_indices]
        # 如果最后一组数据 < max_points，并且已经有至少一组数据
        if len(chunk_p) < max_points and len(grouped_points) > 0:
            # 合并到前一组
            grouped_points[-1] = np.vstack([grouped_points[-1], chunk_p])
            grouped_features[-1] = np.vstack([grouped_features[-1], chunk_f])
        else:
            # 否则，正常添加新组
            grouped_points.append(chunk_p)
            grouped_features.append(chunk_f)
        remaining_indices = remaining_indices[chunk_size:]  # 更新剩余索引
    
    return grouped_points, grouped_features

def angle_between_vectors(v1, v2):
    """计算两个向量之间的夹角(弧度)"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cluster_and_save(points, features, eps=0.8, min_samples=10, angle_threshold=10):
    """
    DBSCAN聚类分割底板
    参数:
    points:点云
    features:特征
    eps:DBSCAN样本邻域距离阈值
    min_samples:DBSCAN与样本距离为eps的邻域中样本个数的阈值
    返回：
    base:底板点云和特征
    other:非底板点云和特征
    """
    all_data = np.hstack((points, features))
    features_to_cluster = np.hstack(
        ((points[:, 2]*2).reshape(-1, 1), features)
    )  # z 和 feature
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features_to_cluster)

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    # 按点数降序排序簇
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]
    
    for i, label in enumerate(sorted_labels):
        cluster_mask = (labels == label)
        cluster_points = points[:, :3][cluster_mask]
        
        # 计算簇的平面法向
        a, b, c = OLS_get_plane(cluster_points)
        normal = np.array([a, b, c])
       
        # 计算法向与Y轴(0,1,0)的夹角
        angle = angle_between_vectors(normal, np.array([0, 1, 0]))
        
        # 如果夹角小于阈值，接受为底板
        if angle < angle_threshold or i == len(sorted_labels)-1:
            base = all_data[cluster_mask]
            other = all_data[~cluster_mask]
            return base, other

def fit_planes(points, features, n_clusters=4):
    plane_points = []
    plane_features = []
    base_pts = points[:, :3]
    kmeans = KMeans(n_clusters=n_clusters).fit(base_pts)
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        plane_points.append(points[mask])
        plane_features.append(features[mask])
    
    return plane_points, plane_features

def OLS_get_plane(points):
    # 拟合 base_points 的平面，使用最小二乘法
    xyz = points  # 提取 x, y, z
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]  # [x, y, 1]
    b = xyz[:, 2]  # z

    # 求解平面系数 z = ax + by + c
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, c = coeff

    return a, b, c


def expand_base_with_plane_fit(base, other, upper_threshold=7.0, lower_threshold=-25.0):
    """
    DBSCAN点云后处理,平面拟合与点云吸收
    参数:
    base_points:底板点云和特征
    other_points:非底板点云和特征
    up_distance_threshold:平面上表面距离阈值
    down_distance_threshold:平面下表面距离阈值
    返回：
    new_base_points:处理后的底板点云和特征
    remaining_points:处理后的底板点云和特征
    """ 
    a, b, c = OLS_get_plane(base[:, :3])
    # 计算 other_points 到该平面的距离
    other_xyz = other[:, 0:3]
    x, y, z = other_xyz[:, 0], other_xyz[:, 1], other_xyz[:, 2]
    z_plane = a * x + b * y + c
    
    dists = z - z_plane

    # 筛选距离小于 threshold 的点
    mask_near = (lower_threshold < dists) & \
                    (dists < upper_threshold)
    absorbed = other[mask_near]
    remaining = other[~mask_near]

    # 合并
    new_base = np.vstack((base, absorbed))

    return new_base, remaining

def select_base(base_points, other_points):
    """
    根据z轴均值选择底板
    参数:
    base_points:底板点云和特征
    other_points:其他点云和特征
    返回：
    base_points:底板点云和特征
    other_points:其他点云和特征
    """
    if base_points.shape[0]==0 or other_points.shape[0]==0:
        return base_points, other_points

    mean_z_base = sum(base_points[:, 2])/base_points.shape[0]
    mean_z_other = sum(other_points[:, 2])/other_points.shape[0]

    if mean_z_base <= mean_z_other:
        return base_points, other_points
    else:
        return other_points, base_points

def read_file(input_points, input_features):
    """
    读取pcd文件
    参数：
        input_points:输入点云txt文件
        input_features:输入特征txt文件
        input_cam_pos:输入相机轨迹txt文件
    返回：
        points:点云[N, 4] 数组，每一行为 (x, y, z, 线号)
        features:特征[N, 4] 数组，每一行为 (x, y, z, 曲率)
        cam_pos:相机轨迹[M, 4] 数组，每一行为 (x, y, z, 线号)
    """
    points = np.loadtxt(input_points,delimiter=' ')
    features = np.loadtxt(input_features,delimiter=' ')

    return points, features


def rotated_points_cam_pos(all_points):
    """
    拟合点云底板并旋转点云
    参数:
        all_points:点云
        cam_pos:相机轨迹
    返回:
        rotated_points:旋转后点云
        rotated_cam_pos:旋转后相机轨迹
        rotation_matrix:旋转矩阵
    """
    points_coords = all_points[:, 0:3]
    points_line_idxs = all_points[:, 3].astype(int)
    
    # 构造点云对象并拟合底板
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_coords)
    segments, non_planes = segment_multiple_planes(pcd)
    clouds=[]
    models=[]
    for i, (model, cloud) in enumerate(segments):
        clouds.append(cloud)
        models.append(model)
    normal = find_closest_to_z_axis(models)

    # 旋转点云使底板法向量对齐 z 轴
    normal = normal / np.linalg.norm(normal)
    target = np.array([0, 0, 1])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)
    if s == 0:
        rotation_matrix = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # 如果拟合出的底板法向与z轴夹角超过一定阈值，则不旋转点云
    if np.degrees(np.arccos(np.dot(normal, (0,0,1))))>15:
        rotation_matrix = np.eye(3)
        return all_points, rotation_matrix

    rotated_points_coords = (rotation_matrix @ points_coords.T).T  # shape: (N, 3)
    rotated_points = np.hstack((rotated_points_coords, points_line_idxs.reshape(-1, 1)))

    return rotated_points, rotation_matrix

def sub_split_base_other(plane_points, plane_features):

    grouped_points, grouped_features = chunk_point_feature(plane_points, plane_features)

    # 底板分割
    all_base = []
    all_other = []
    for i in range(len(grouped_points)):
        base, other = cluster_and_save(grouped_points[i], grouped_features[i])
        base, other = expand_base_with_plane_fit(base, other)
        all_base.append(base)
        all_other.append(other)

    all_base = np.vstack(all_base) if all_base else np.empty((0, 8))
    all_other = np.vstack(all_other) if all_other else np.empty((0, 8))

    return all_base, all_other

def split_base_other(rotated_points, all_features):
    """
    分割底板和点云
    参数:
        rotated_points:旋转后点云
        all_features:特征
    返回：
        rotated_base_points:旋转后底板点云
        base_features:底板点云特征
        rotated_other_points:旋转后非底板点云
        other_features:非底板点云特征
    """
    all_base = []
    all_other = []
    plane_points, plane_features = fit_planes(rotated_points, all_features)

    for i in range(len(plane_points)):
        base, other = sub_split_base_other(plane_points[i], plane_features[i])

        all_base.append(base)
        all_other.append(other)
    all_base = np.vstack(all_base)
    all_other = np.vstack(all_other)
  

    rotated_base_points = all_base[:, 0:4]
    base_features = all_base[:, 4:8]
    rotated_other_points = all_other[:, 0:4]
    other_features = all_other[:, 4:8]

    return rotated_base_points, base_features, rotated_other_points, other_features

def process_split_base(input_points, input_features):
    """
    主程序-底板分割
    参数:
        input_points:点云(x, y, z, 线号)
        input_features:特征(n_x, n_y, n_z, c)
        cam_pos:相机轨迹(x, y, z, 线号)
    返回:
        rotated_base_points:旋转后的底板点云
        base_features:底板点云特征
        rotated_other_points:旋转后的非底板点云
        other_features:非底板点云特征
        rotated_cam_pos:旋转后的相机轨迹
        rotated_matrix:旋转矩阵
    """
    # start_time = time.time()
    # 读取txt文件
    all_points, all_features = read_file(input_points, input_features)

    # 旋转点云和相机轨迹
    rotated_points, rotation_matrix = rotated_points_cam_pos(all_points)

    # 分割底板和其他点云
    rotated_base_points, base_features, rotated_other_points, other_features = split_base_other(rotated_points, all_features)

    # end_time = time.time()
    # print(f"底板分割总耗时 {end_time - start_time:.4f} seconds")

    return rotated_base_points, base_features, rotated_other_points, other_features, rotation_matrix

def process_directory(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    for subdir in os.listdir(input_dir):
        subdir_path = Path(input_dir) / subdir
        out_subdir_path = Path(output_dir) / subdir
        
        out_subdir_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        if subdir_path.is_dir():
            pre_downsample_file = subdir_path / 'point_cloud.txt'
            feature_file = subdir_path / 'feature.txt'

            if pre_downsample_file.exists() and feature_file.exists():
                start_time = time.time()
                rotated_base_points, base_features, rotated_other_points, other_features, rotation_matrix = process_split_base(pre_downsample_file,feature_file)
                end_time = time.time()
                print(f"底板分割总耗时 {end_time - start_time:.4f} seconds")
                
                # 保存点云
                print(rotated_base_points.shape, base_features.shape, rotated_other_points.shape, other_features.shape)
                np.savetxt(out_subdir_path / 'base.txt', rotated_base_points, fmt='%.6f %.6f %.6f %.6f')
                np.savetxt(out_subdir_path / 'base_features.txt', base_features, fmt='%.6f %.6f %.6f %.6f')
                np.savetxt(out_subdir_path / 'other.txt', rotated_other_points, fmt='%.6f %.6f %.6f %.6f')
                np.savetxt(out_subdir_path / 'other_features.txt', other_features, fmt='%.6f %.6f %.6f %.6f')
                np.savetxt(out_subdir_path / 'rotation_matrix.txt', rotation_matrix)
                print(f"Saved base.txt ({rotated_base_points.shape[0]} pts) and other.txt ({rotated_other_points.shape[0]} pts) for {subdir}")

def main():
    # input_directory = r'D:\1-4data\baffle'  # 输入文件夹路径
    # output_directory = r'D:\1-4data\baffle'  # 输出文件夹路径
    # input_directory = r"C:\Users\chenjiayi\Documents\jiayi\split_base\point_cloud_test_data"
    # output_directory = r"C:\Users\chenjiayi\Documents\jiayi\split_base\point_cloud_test_data_output"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    process_directory(args)
    # input_dir = input_directory
    # output_dir = output_directory
    # subdir = '14'
    # subdir_path = Path(input_dir) / subdir
    # out_subdir_path = Path(output_dir) / subdir
    # pre_downsample_file = subdir_path / 'point_cloud.txt'
    # feature_file = subdir_path / 'feature.txt'
    # eps = 0.8
    # max_points = 2000
    # start_time = time.time()
    # rotated_base_points, base_features, rotated_other_points, other_features, rotation_matrix = process_split_base(pre_downsample_file,feature_file, eps, max_points)
    # end_time = time.time()
    # print(f"底板分割总耗时 {end_time - start_time:.4f} seconds")
    
    # # 保存点云
    # print(rotated_base_points.shape, base_features.shape, rotated_other_points.shape, other_features.shape)
    # np.savetxt(out_subdir_path / f'base_{eps}_{max_points}.txt', rotated_base_points, fmt='%.6f %.6f %.6f %.6f')
    # # np.savetxt(out_subdir_path / 'base_features.txt', base_features, fmt='%.6f %.6f %.6f %.6f')
    # np.savetxt(out_subdir_path / f'other_{eps}_{max_points}.txt', rotated_other_points, fmt='%.6f %.6f %.6f %.6f')
    # # np.savetxt(out_subdir_path / 'other_features.txt', other_features, fmt='%.6f %.6f %.6f %.6f')
    # # np.savetxt(out_subdir_path / 'rotation_matrix.txt', rotation_matrix)
    # print(f"Saved base.txt ({rotated_base_points.shape[0]} pts) and other.txt ({rotated_other_points.shape[0]} pts) for {subdir}")

if __name__ == "__main__":
    main()
