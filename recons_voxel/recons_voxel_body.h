#pragma once

#include <opencv2\opencv.hpp>
#include "recons_voxel.h"
#include "recons_cylinder.h"

#define VOXEL_VOLUME_X_RATIO 2.0f
#define VOXEL_VOLUME_Z_RATIO 3.0f

//transformation from voxel coordinates (integer values) to body part local values (i.e. given voxel size, etc)
cv::Mat get_voxel_transform(float width, float height, float depth, float voxel_size);

//initializes a voxelset. make sure to call delete_voxel_set later!
void init_voxel_set(
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeHardMap& snhMap,
	const std::vector<Cylinder>& fittedCylinders,
	const cv::Mat& external_parameters,
	std::vector<VoxelMatrix>& voxelSetVector,
	const std::vector<VolumeDimensions>& volume_sizes,
	VoxelSetMap& map,
	float voxel_size);


//initializes a voxelset. make sure to call delete_voxel_set later! ABSOLUTE
void init_voxel_set(
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeAbsoluteVector& snav,
	const std::vector<Cylinder>& fittedCylinders,
	const cv::Mat& external_parameters,
	std::vector<VoxelMatrix>& voxelSetVector,
	const std::vector<VolumeDimensions>& volume_sizes,
	VoxelSetMap& map,
	float voxel_size);

void delete_voxel_set(std::vector<VoxelSet *>& voxelSetVector);

//draws the voxels in each voxelset
void voxel_draw_volume(cv::Mat& image,
	const cv::Vec3b& color_v,
	const cv::Mat& volume_transform,
	const cv::Mat& camera_matrix,
	const VoxelSet * voxel_volume,
	float voxel_size);
