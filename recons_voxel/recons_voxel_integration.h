#pragma once
#include <opencv2\opencv.hpp>

#include "recons_voxel.h"
#include "recons_cylinder.h"

//integrate volume with the entire voxel set
//depth multiplier should be -1 if depth is negative
void integrate_volume(
	const cv::Mat& bodypart_transform,
	const Grid3D<char> voxel_assignment,
	const cv::Mat& depth,
	const cv::Mat& camera_pose,
	const cv::Mat& camera_pose_inv,
	const cv::Mat& camera_matrix,
	const cv::Mat& camera_matrix_inv,
	VoxelMatrix volume,
	cv::Mat& TSDF,
	cv::Mat& TSDF_weight,
	float voxel_size,
	float TSDF_MU,
	int depth_multiplier,
	const cv::Mat& optional_bg_mask = cv::Mat());

//if voxelAssignments[i][x][y][z] is 1, the voxel in body part volume i coord x,y,z belongs to i;
//if 2, it belongs to another part;
//if 0, it is empty space;
std::vector<Grid3D<char>> assign_voxels_to_body_parts(
	const BodyPartDefinitionVector& bpdv,
	const std::vector<cv::Mat>& bodypart_transforms,
	const std::vector<Cylinder>& cylinderVector,
	const cv::Mat& depth,
	const cv::Mat& camera_pose,
	const cv::Mat& camera_matrix,
	const std::vector<VoxelMatrix>& volumeSet,
	float voxel_size);



void save_voxels(const std::string& filename, const std::vector<Cylinder>& cylinders, const std::vector<VoxelMatrix>& volumes, const std::vector<cv::Mat>& TSDF_array, const std::vector<cv::Mat>& weight_array, const float voxel_size);


void load_voxels(const std::string& filename, std::vector<Cylinder>& cylinders, std::vector<VoxelMatrix>& volumes, std::vector<cv::Mat>& TSDF_array, std::vector<cv::Mat>& weight_array, float& voxel_size);