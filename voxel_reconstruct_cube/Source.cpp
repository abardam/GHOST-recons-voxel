#include <opencv2\opencv.hpp>
#include "recons_optimization.h"
#include "recons_voxel_integration.h"
#include "recons_voxel_body.h"
#include "recons_voxel.h"
#include "AssimpOpenGL.h"

int main(int argc, char** argv){

	float voxel_size = 0.01;
	float volume_width = 0.2;
	float volume_height = 0.2;
	float volume_depth = 0.2;

	if (argc <= 1){
		std::cout << "Please enter directory\n";
		return 0;
	}

	bool point_to_point = true;
	bool point_to_plane = true;

	std::string video_directory(argv[1]);
	std::stringstream filenameSS;
	int startframe = 90;
	int numframes = 90;
	cv::FileStorage fs;

	std::vector<std::string> filenames;


	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << ".xml.gz";

		filenames.push_back(filenameSS.str());

	}

	std::vector<PointMap> point_maps;
	std::vector<FrameData> frame_datas;

	load_frames(filenames, point_maps, frame_datas);

	

	while (true){

		cv::Mat current;

		VoxelMatrix cube_volume(volume_width / voxel_size, volume_height / voxel_size, volume_depth / voxel_size);
		Grid3D<char> voxel_assignment = make_Grid3D<char>(volume_width / voxel_size, volume_height / voxel_size, volume_depth / voxel_size, 2);

		cv::Mat TSDF(1, volume_width / voxel_size * volume_height / voxel_size * volume_depth / voxel_size, CV_32F, cv::Scalar(0));
		cv::Mat TSDF_weight(1, TSDF.cols, CV_32F, cv::Scalar(0));

		for (int frame = 0; frame < numframes - 1; ++frame){
			cv::Mat currentTransform = frame_datas[frame].mmCameraPose;
			cv::Mat bp_transform = cv::Mat::eye(4, 4, CV_32F);
			bp_transform.ptr<float>(1)[3] = -volume_height/2;
			bp_transform = currentTransform * bp_transform;

			integrate_volume(bp_transform, voxel_assignment, frame_datas[frame].mmDepth, currentTransform, currentTransform.inv(), frame_datas[frame].mmCameraMatrix, frame_datas[frame].mmCameraMatrix.inv(), cube_volume, TSDF, TSDF_weight, voxel_size, voxel_size, -1);


			cv::Mat im = frame_datas[frame].mmColor.clone();

			cv::imshow("color", frame_datas[frame].mmColor);

			cv_draw_volume(cv::Scalar(0x00, 0x55, 0xff), bp_transform, cube_volume.width * voxel_size, cube_volume.height * voxel_size, cube_volume.depth * voxel_size, im, currentTransform, frame_datas[frame].mmCameraMatrix);
			voxel_draw_volume(im, cv::Vec3b(0xff, 0x55, 0x11), bp_transform, frame_datas[frame].mmCameraMatrix, &cube_volume, voxel_size);

			cv::imshow("pts", im);

			cv::waitKey(5);

			//print out left side TSDF
			int s = volume_width / voxel_size;

			std::cout << "---------\n";
			for (int y = 0; y < s; ++y){
				for (int z = 0; z < s; ++z){
					unsigned int v_index = get_voxel_index(4, y, z, s, s, s);

					printf("%.1f ", TSDF.ptr<float>()[v_index]);
				}
				std::cout << std::endl;
			}
		}

	}
}