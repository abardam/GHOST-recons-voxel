#include "recons_voxel.h"

unsigned int get_voxel_index(int x, int y, int z, int width, int height, int depth){
	return x*(height*depth) + y*(depth)+z;
}

VoxelMatrix::VoxelMatrix(const VoxelArray& voxelA) :
VoxelSet(voxelA.width, voxelA.height, voxelA.depth){
	type = Type::Matrix;
	std::vector<cv::Vec4f> points;
	std::vector<cv::Vec4b> voxel_color_existence;

	for (int x = 0; x < voxelA.width; ++x){
		for (int y = 0; y < voxelA.height; ++y){
			for (int z = 0; z < voxelA.depth; ++z){
				points.push_back(cv::Vec4f(x, y, z, 1));
				voxel_color_existence.push_back(cv::Vec4b(
					voxelA.voxels[x][y][z].color(0),
					voxelA.voxels[x][y][z].color(1),
					voxelA.voxels[x][y][z].color(2),
					voxelA.voxels[x][y][z].exists ? 0xff : 0
					));
			}
		}
	}

	cv::Mat voxels_r(1, points.size(), CV_32FC4, points.data());
	voxel_coords = voxels_r.reshape(1, voxels_r.cols).t();

	cv::Mat voxel_data_r(1, voxel_color_existence.size(), CV_8UC4, voxel_color_existence.data());
	voxel_data = voxel_data_r;
}

VoxelMatrix::VoxelMatrix(int x, int y, int z, bool init_exists):
VoxelSet(x,y,z){
	type = Type::Matrix;
	std::vector<cv::Vec4f> points;
	std::vector<cv::Vec4b> voxel_color_existence;

	for (int i = 0; i < x; ++i){
		for (int j = 0; j < y; ++j){
			for (int k = 0; k < z; ++k){
				points.push_back(cv::Vec4f(i, j, k, 1));
				voxel_color_existence.push_back(cv::Vec4b(
					0,
					0,
					0,
					init_exists?0xff:0
					));
			}
		}
	}

	cv::Mat voxels_r(1, points.size(), CV_32FC4, points.data());
	voxel_coords = voxels_r.reshape(1, voxels_r.cols).t();

	cv::Mat voxel_data_r(1, voxel_color_existence.size(), CV_8UC4, voxel_color_existence.data());
	voxel_data = voxel_data_r.clone();
}


VoxelArray::VoxelArray(const VoxelMatrix& voxelM) :
VoxelSet(voxelM.width, voxelM.height, voxelM.depth),
voxels(voxelM.width, std::vector<std::vector<Voxel>>(voxelM.height, std::vector<Voxel>(voxelM.depth)))
{
	type = Type::Array;

	if (!voxelM.voxel_coords.empty()){
		for (int i = 0; i < voxelM.voxel_coords.cols; ++i){

			int x = voxelM.voxel_coords.ptr<float>(0)[i];
			int y = voxelM.voxel_coords.ptr<float>(1)[i];
			int z = voxelM.voxel_coords.ptr<float>(2)[i];

			cv::Vec4b voxel_data = voxelM.voxel_data.ptr<cv::Vec4b>()[i];

			voxels[x][y][z].exists = voxel_data(3)==0xff;
			voxels[x][y][z].color = cv::Vec3b(
				voxel_data(0),
				voxel_data(1),
				voxel_data(2)
				);
		}
	}
};


void voxelset_array_to_matrix(std::vector<VoxelSet*>& voxelSet){
	for (int i = 0; i < voxelSet.size(); ++i){
		VoxelSet * vs = voxelSet[i];

		if (vs->type == VoxelSet::Array){
			VoxelArray * vs_a = dynamic_cast<VoxelArray*>(vs);
			VoxelMatrix * vs_m = new VoxelMatrix(*vs_a);
			delete vs;
			voxelSet[i] = vs_m;
		}
	}
}


void voxelset_matrix_to_array(std::vector<VoxelSet*>& voxelSet){
	for (int i = 0; i < voxelSet.size(); ++i){
		VoxelSet * vs = voxelSet[i];

		if (vs->type == VoxelSet::Matrix){
			VoxelMatrix * vs_m = dynamic_cast<VoxelMatrix*>(vs);
			VoxelArray * vs_a = new VoxelArray(*vs_m);
			delete vs;
			voxelSet[i] = vs_a;
		}
	}
}