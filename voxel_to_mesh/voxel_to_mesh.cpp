#include <recons_marchingcubes.h>
#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include <recons_voxel_integration.h>

std::vector<Cylinder> cylinders;
std::vector<VoxelMatrix> voxels;

int main(int argc, char ** argv){

	if (argc < 3){
		std::cout << "specify directory and voxel reconstruction path!\n";
		return 0;
	}
	std::string video_directory(argv[1]);
	std::string voxel_recons_path(argv[2]);

	cv::FileStorage fs;
	std::stringstream filename_ss;
	filename_ss << video_directory << "/bodypartdefinitions.xml.gz";

	BodyPartDefinitionVector bpdv;
	fs.open(filename_ss.str(), cv::FileStorage::READ);
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}
	fs.release();

	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;
	float voxel_size;

	load_voxels(voxel_recons_path, cylinders, voxels, TSDF_array, weight_array, voxel_size);

	std::vector<std::vector<float>> triangle_vertices;
	std::vector<std::vector<int>> triangle_indices;
	std::vector<std::vector<unsigned char>> triangle_colors;

	triangle_vertices.resize(bpdv.size());
	triangle_indices.resize(bpdv.size());
	triangle_colors.resize(bpdv.size());

	double num_vertices = 0;

	for (int i = 0; i < bpdv.size(); ++i){
		std::vector<TRIANGLE> tri_add;

		if (TSDF_array[i].empty()){
			tri_add = marchingcubes_bodypart(voxels[i], voxel_size);
		}
		else{
			tri_add = marchingcubes_bodypart(voxels[i], TSDF_array[i], voxel_size);
		}
		std::vector<cv::Vec4f> vertices;
		std::vector<int> vertex_indices;
		for (int j = 0; j < tri_add.size(); ++j){
			for (int k = 0; k < 3; ++k){
				cv::Vec4f candidate_vertex = tri_add[j].p[k];

				bool vertices_contains_vertex = false;
				int vertices_index;
				for (int l = 0; l < vertices.size(); ++l){
					if (vertices[l] == candidate_vertex){
						vertices_contains_vertex = true;
						vertices_index = l;
						break;
					}
				}
				if (!vertices_contains_vertex){
					vertices.push_back(candidate_vertex);
					vertices_index = vertices.size() - 1;
				}
				vertex_indices.push_back(vertices_index);
			}
		}
		triangle_vertices[i].reserve(vertices.size() * 3);
		triangle_colors[i].reserve(vertices.size() * 3);
		triangle_indices[i].reserve(vertex_indices.size());
		for (int j = 0; j < vertices.size(); ++j){
			triangle_vertices[i].push_back(vertices[j](0));
			triangle_vertices[i].push_back(vertices[j](1));
			triangle_vertices[i].push_back(vertices[j](2));
			triangle_colors[i].push_back(bpdv[i].mColor[0] * 255);
			triangle_colors[i].push_back(bpdv[i].mColor[1] * 255);
			triangle_colors[i].push_back(bpdv[i].mColor[2] * 255);
		}
		num_vertices += vertices.size();
		for (int j = 0; j < vertex_indices.size(); ++j){
			triangle_indices[i].push_back(vertex_indices[j]);
		}
	}

	fs.open(video_directory + "/meshmodel.xml.gz", cv::FileStorage::WRITE);

	fs << "meshmodel" << "[";

	for (int i = 0; i < bpdv.size(); ++i){
		fs << "{";
		fs << "triangle_vertices" << "[";

		for (int j = 0; j < triangle_vertices[i].size(); ++j){
			fs << triangle_vertices[i][j];
		}
		fs << "]";

		fs << "triangle_indices" << "[";
		for (int j = 0; j < triangle_indices[i].size(); ++j){
			fs << triangle_indices[i][j];
		}
		fs << "]";
		fs << "triangle_colors" << "[";
		for (int j = 0; j < triangle_colors[i].size(); ++j){
			fs << triangle_colors[i][j];
		}
		fs << "]";
		fs << "}";
	}

	fs << "]";

	fs.release();
}