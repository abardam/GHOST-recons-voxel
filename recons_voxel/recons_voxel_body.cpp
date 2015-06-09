#include "recons_voxel_body.h"
#include <cv_draw_common.h>
#include <cv_pointmat_common.h>

cv::Vec4f vertex_(const cv::Mat& transform, const cv::Vec4f& local_vertex){
	//first make a translation matrix out of the vertex
	cv::Mat translation = cv::Mat::eye(4, 4, CV_32F);
	translation.ptr<float>(0)[3] = local_vertex(0);
	translation.ptr<float>(1)[3] = local_vertex(1);
	translation.ptr<float>(2)[3] = local_vertex(2);

	//next transform
	return get_origin(transform * translation);
}

cv::Mat get_voxel_transform(float width, float height, float depth, float voxel_size){
	cv::Mat voxel_transform = cv::Mat::eye(4, 4, CV_32F);
	voxel_transform.ptr<float>(0)[3] = -width / 2 - voxel_size/2;
	voxel_transform.ptr<float>(1)[3] = 0 - voxel_size/2;
	voxel_transform.ptr<float>(2)[3] = -depth/2 - voxel_size/2;
	voxel_transform = cv::Mat::diag(cv::Mat(cv::Vec4f(voxel_size, voxel_size, voxel_size, 1))) * voxel_transform;
	return voxel_transform;
}

void init_voxel_set(
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeHardMap& snhMap,
	const std::vector<Cylinder>& fittedCylinders,
	const cv::Mat& external_parameters,
	std::vector<VoxelMatrix>& voxelSetVector,
	const std::vector<VolumeDimensions>& volume_sizes,
	VoxelSetMap& map,
	float voxel_size){

	for (int i = 0; i < bpdv.size(); ++i){
		
		get_bodypart_transform(bpdv[i], snhMap, external_parameters);


		if (volume_sizes[i].width <= 0 || volume_sizes[i].depth <= 0 || volume_sizes[i].height <= 0){
			std::cout << "error! init_voxel_set: volume_sizes has a 0 width/height/depth!\n";
			//int w = fittedCylinders[i].width * VOXEL_VOLUME_X_RATIO / voxel_size;
			//int h = length / voxel_size;
			//int d = fittedCylinders[i].height* VOXEL_VOLUME_Z_RATIO / voxel_size;
			//
			//if (w <= 0) w = 1;
			//if (h <= 0) h = 1;
			//if (d <= 0) d = 1;
			//
			//VoxelMatrix voxelSet_m(w, h, d, true);
			//
			//voxelSetVector.push_back(voxelSet_m);
			//map.insert(VoxelSetEntry(bpdv[i].mBodyPartName, voxelSetVector.size() - 1));
		}
		{
			int w = volume_sizes[i].width / voxel_size;
			int h = volume_sizes[i].height / voxel_size;
			int d = volume_sizes[i].depth / voxel_size;

			if (w <= 0) w = 1;
			if (h <= 0) h = 1;
			if (d <= 0) d = 1;

			VoxelMatrix voxelSet_m(w, h, d, true);

			voxelSetVector.push_back(voxelSet_m);
			VoxelSetEntry vse(bpdv[i].mBodyPartName, voxelSetVector.size() - 1);
			map.insert(vse);

		}
	}
}



void init_voxel_set(
	const BodyPartDefinitionVector& bpdv,
	const SkeletonNodeAbsoluteVector& snav,
	const std::vector<Cylinder>& fittedCylinders,
	const cv::Mat& external_parameters,
	std::vector<VoxelMatrix>& voxelSetVector,
	const std::vector<VolumeDimensions>& volume_sizes,
	VoxelSetMap& map,
	float voxel_size){

	for (int i = 0; i < bpdv.size(); ++i){

		get_bodypart_transform(bpdv[i], snav, external_parameters);


		if (volume_sizes[i].width <= 0 || volume_sizes[i].depth <= 0 || volume_sizes[i].height <= 0){
			std::cout << "error! init_voxel_set: volume_sizes has a 0 width/height/depth!\n";
			//int w = fittedCylinders[i].width * VOXEL_VOLUME_X_RATIO / voxel_size;
			//int h = length / voxel_size;
			//int d = fittedCylinders[i].height* VOXEL_VOLUME_Z_RATIO / voxel_size;
			//
			//if (w <= 0) w = 1;
			//if (h <= 0) h = 1;
			//if (d <= 0) d = 1;
			//
			//VoxelMatrix voxelSet_m(w, h, d, true);
			//
			//voxelSetVector.push_back(voxelSet_m);
			//map.insert(VoxelSetEntry(bpdv[i].mBodyPartName, voxelSetVector.size() - 1));
		}
		{
			int w = volume_sizes[i].width / voxel_size;
			int h = volume_sizes[i].height / voxel_size;
			int d = volume_sizes[i].depth / voxel_size;

			if (w <= 0) w = 1;
			if (h <= 0) h = 1;
			if (d <= 0) d = 1;

			VoxelMatrix voxelSet_m(w, h, d, true);

			voxelSetVector.push_back(voxelSet_m);
			VoxelSetEntry vse(bpdv[i].mBodyPartName, voxelSetVector.size() - 1);
			map.insert(vse);

		}
	}
}

void delete_voxel_set(std::vector<VoxelSet *>& voxelSetVector){
	for (int i = 0; i < voxelSetVector.size(); ++i){
		delete voxelSetVector[i];
	}
	voxelSetVector.clear();
}

void voxel_draw_volume(cv::Mat& image, 
	const cv::Vec3b& color_v,
	const cv::Mat& volume_transform,
	const cv::Mat& camera_matrix, 
	const VoxelSet * voxel_volume, 
	float voxel_size){

	//volume

	const VoxelSet * vs = voxel_volume;

	if (vs->type == VoxelSet::Type::Array)
	{
		cv::Scalar color(color_v(0), color_v(1), color_v(2));
		const VoxelArray * vs_a = dynamic_cast<const VoxelArray*>(vs);

		for (int x = 0; x < vs->width; ++x){
			for (int y = 0; y < vs->height; ++y){
				for (int z = 0; z < vs->depth; ++z){
					//TODO: find a way to matrixify this

					if (vs_a->voxels[x][y][z].exists){
						cv::circle(image,
							project2D(vertex_(volume_transform,
							cv::Vec4f(
							(x - vs->width / 2)*voxel_size,
							(y - vs->height / 2)*voxel_size,
							(z - vs->depth / 2)*voxel_size, 1)),
							camera_matrix), 1, color, -1);
					}
				}
			}
		}
	}
	else if (vs->type == VoxelSet::Type::Matrix){
		const VoxelMatrix* vs_m = dynamic_cast<const VoxelMatrix*>(vs);
		if (vs_m->voxel_coords.empty()) return;

		const cv::Mat voxel_transform = get_voxel_transform(vs->width, vs->height, vs->depth, voxel_size);
		const cv::Mat transform = volume_transform * voxel_transform;

		cv::Mat transformed = camera_matrix * transform * vs_m->voxel_coords;

		divide_pointmat_by_z(transformed);

		for (int i = 0; i < transformed.cols; ++i){

			if (vs_m->voxel_data.ptr<cv::Vec4b>()[i](3) == 0xff){

				int x = transformed.ptr<float>(0)[i];
				int y = transformed.ptr<float>(1)[i];

				if (CLAMP(x, y, image.cols, image.rows)){
					image.ptr<cv::Vec3b>(y)[x] = color_v;
				}
			}

		}
	}
}
