# Pointcloud_segmentation
Divide the bottom plate of the workpiece point cloud.

Visualization of segmentation results

![示例图片](results/1.jpg)

![示例图片](results/2.jpg)

![示例图片](results/3.jpg)

![示例图片](results/4.jpg)  

Split_base_2.py adds the k-means method to first divide the point cloud into n blocks, and then perform dbscan clustering to solve the problem of bent or uneven bottom plates
