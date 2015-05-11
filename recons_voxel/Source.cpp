#include <opencv2\opencv.hpp>
#include <recons_optimization.h>
#include <AssimpCV.h>
#include <cv_pointmat_common.h>
#include <limits>


#include "recons_voxel.h"
#include "recons_voxel_body.h"
#include "recons_cylinder.h"
#include "recons_voxel_integration.h"

SkeletonNodeHard generateFromReference(const SkeletonNodeHard * const ref, const SkeletonNodeHard * const prev){
	SkeletonNodeHard snh;
	cv::Mat generatedTransformation = skeleton_constraints_optimize(prev->mTransformation, ref->mTransformation, 1, 1);

	//cv::Mat rotationMatrix = generatedTransformation(cv::Range(0, 3), cv::Range(0, 3)) * prev->mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	//cv::Mat translationMatrix = generatedTransformation(cv::Range(0, 3), cv::Range(3, 4)) + prev->mTransformation(cv::Range(0, 3), cv::Range(3, 4));
	//
	//rotationMatrix.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));
	//translationMatrix.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(3, 4)));


	for (int i = 0; i < prev->mChildren.size(); ++i){
		snh.mChildren.push_back(generateFromReference(&ref->mChildren[i], &prev->mChildren[i]));
	}

	snh.mTransformation = generatedTransformation * prev->mTransformation;

	//SVD correct scale error?
	//cv::SVD genSVD(snh.mTransformation);
	//cv::SVD refSVD(ref->mTransformation);
	//snh.mTransformation = genSVD.u * cv::Mat::diag(refSVD.w) * (genSVD.u * cv::Mat::diag(genSVD.w)).inv() * snh.mTransformation;

	//extract scale and correct it?
	cv::Vec3f scale(
		1 / cv::norm(snh.mTransformation(cv::Range(0, 1), cv::Range(0, 3))),
		1 / cv::norm(snh.mTransformation(cv::Range(1, 2), cv::Range(0, 3))),
		1 / cv::norm(snh.mTransformation(cv::Range(2, 3), cv::Range(0, 3)))
		);

	cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	rot.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));

	snh.mName = prev->mName;
	snh.mParentName = prev->mParentName;

	return snh;
}

void display_points_and_volumes(
	const BodyPartDefinitionVector& bpdv, 
	const SkeletonNodeHardMap& snhMap, 
	const std::vector<Cylinder>& cylinderVector,
	const cv::Mat& pointCloud_, 
	const std::vector<VoxelMatrix>& voxels, 
	const cv::Mat& camera_pose_, 
	const cv::Mat& camera_matrix,
	float voxel_size){
	cv::Mat image_(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));

	int i = 0;

	cv::Mat temp_trans = cv::Mat::eye(4, 4, CV_32F);

	cv::reduce(pointCloud_, temp_trans(cv::Range(0,4), cv::Range(3,4)), 1, CV_REDUCE_AVG);

	while (true){

		cv::Mat temp_rot;
		{
			cv::Vec3f rot_v(0, i*0.1, 0);
			cv::Mat temp_rot_;
			cv::Rodrigues(rot_v, temp_rot_);

			temp_rot = cv::Mat::eye(4, 4, CV_32F);
			temp_rot_.copyTo(temp_rot(cv::Range(0, 3), cv::Range(0, 3)));

			temp_rot = temp_trans * temp_rot * temp_trans.inv();
		}

		cv::Mat image = image_.clone();
		cv::Mat pointCloud = temp_rot * pointCloud_;
		cv::Mat camera_pose = temp_rot * camera_pose_;


		draw_points_on_image(pointCloud, camera_matrix, image, cv::Vec3b(0x00, 0x55, 0xff));

		for (int i = 0; i < bpdv.size(); ++i){
			//cv_draw_volume(bpdv[i], cylinderVector[i].width, cylinderVector[i].height, image, camera_pose, camera_matrix, snhMap);
			//voxel_draw_volume(image, bpdv[i], snhMap, camera_pose, camera_matrix, voxels[i]);
			const VoxelMatrix * vs_m = &(voxels[i]);

			cv::Mat voxelPC = temp_rot * get_bodypart_transform(bpdv[i], snhMap, camera_pose) * get_voxel_transform(vs_m->width, vs_m->height, vs_m->depth, voxel_size) * vs_m->voxel_coords;
			cv::Vec3b color(bpdv[i].mColor[0]*255,
				bpdv[i].mColor[1]*255,
				bpdv[i].mColor[2]*255);
			draw_points_on_image(voxelPC, camera_matrix, image, color);
		}

		cv::imshow("im", image);
		cv::waitKey(50);
		++i;
	}
}

int main(int argc, char * argv[]){
	if (argc <= 1){
		std::cout << "Please enter directory\n";
		return 0;
	}

	float cylinder_fitting_threshold = 0.4;
	float cylinder_fitting_max = 2;
	float cylinder_fitting_increment = 0.1;

	std::string video_directory(argv[1]);
	int i = 0;
	std::stringstream filenameSS;
	cv::FileStorage fs;

	filenameSS << video_directory << "/bodypartdefinitions.xml.gz";

	fs.open(filenameSS.str(), cv::FileStorage::READ);
	BodyPartDefinitionVector bpdv;
	std::vector<Cylinder> cylinderVector; //1 to 1 with the body part definitions
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
		cylinderVector.push_back(Cylinder(1, 1));
	}
	fs.release();

	float voxel_size = 0.1;

	cv::Mat depthMat, colorMat, camera_extrinsic, camera_intrinsic;
	SkeletonNodeHardMap snhMap;
	VoxelSetMap vsMap;

	SkeletonNodeHard root;
	SkeletonNodeHard prevRoot;

	std::vector<VoxelMatrix> voxelSet;
	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;
	std::vector<std::vector<Voxel>> surface;

	bool firstframe = true;

	cv::Mat camera_1;

	//experimental
	//std::vector<std::pair<float, float>> volume_sizes;
	//volume_sizes.push_back(std::pair<float, float>(3, 3));
	//volume_sizes.push_back(std::pair<float, float>(6, 6));
	//volume_sizes.push_back(std::pair<float, float>(5, 5));
	//volume_sizes.push_back(std::pair<float, float>(2.5, 2.5));
	//volume_sizes.push_back(std::pair<float, float>(2.5, 2.5));
	//volume_sizes.push_back(std::pair<float, float>(2, 2));
	//volume_sizes.push_back(std::pair<float, float>(2, 2));
	//volume_sizes.push_back(std::pair<float, float>(4, 2));
	//volume_sizes.push_back(std::pair<float, float>(4, 2));
	//volume_sizes.push_back(std::pair<float, float>(3, 3));
	//volume_sizes.push_back(std::pair<float, float>(3, 3));
	//volume_sizes.push_back(std::pair<float, float>(2, 2));
	//volume_sizes.push_back(std::pair<float, float>(2, 2));

	while (true){
		filenameSS.str("");
		filenameSS << video_directory << "/" << i << ".xml.gz";
		fs.open(filenameSS.str(), cv::FileStorage::READ);

		if (!fs.isOpened()) {
			break;
		}

		fs["color"] >> colorMat;
		fs["depth"] >> depthMat;

		//todo: put some background behind our dude
		//for now: just set depth values to some wall number

		for (int i = 0; i < depthMat.rows*depthMat.cols; ++i){
			if (depthMat.ptr<float>()[i] == 0){
				depthMat.ptr<float>()[i] = -4;
			}
		}

		fs["camera_extrinsic"] >> camera_extrinsic;
		//fs["camera_intrinsic"] >> camera_intrinsic;

		float win_width, win_height;
		float fovy;

		fs["camera_intrinsic"]["width"] >> win_width;
		fs["camera_intrinsic"]["height"] >> win_height;
		fs["camera_intrinsic"]["fovy"] >> fovy;

		camera_intrinsic = cv::Mat::eye(4, 4, CV_32F);
		camera_intrinsic.ptr<float>(0)[0] = -win_width / (2 * tan(AI_DEG_TO_RAD((fovy * (win_width / win_height) / 2.)))); //for some strange reason this is inaccurate for non-square aspect ratios
		camera_intrinsic.ptr<float>(1)[1] = win_height / (2 * tan(AI_DEG_TO_RAD(fovy / 2.)));
		camera_intrinsic.ptr<float>(0)[2] = win_width / 2 + 0.5;
		camera_intrinsic.ptr<float>(1)[2] = win_height / 2 + 0.5;
		//camera_intrinsic.ptr<float>(2)[2] = -1;

		fs["skeleton"] >> root;

		SkeletonNodeHard gen_root;
		if (!firstframe){
			gen_root = generateFromReference(&root, &prevRoot);
		}
		else{
			firstframe = false;

			camera_1 = camera_extrinsic.clone();

			gen_root = root;
			cv_draw_and_build_skeleton(&gen_root, cv::Mat::eye(4,4,CV_32F), camera_intrinsic, camera_extrinsic, &snhMap);
			voxelSet.clear();
			vsMap.clear();


			PointMap pointMap(win_width, win_height);
			read_depth_image(depthMat, camera_intrinsic, pointMap);
			cv::Mat pointCloud(4, pointMap.mvPointLocations.size(), CV_32F);
			read_points_pointcloud(pointMap, pointCloud);

			cylinder_fitting(bpdv, snhMap, pointCloud, camera_extrinsic, cylinderVector, cylinder_fitting_increment, cylinder_fitting_max, cylinder_fitting_threshold);//, &camera_intrinsic, &win_width, &win_height);

			init_voxel_set(bpdv, snhMap, cylinderVector, camera_extrinsic, voxelSet, volume_sizes, vsMap, voxel_size);


			TSDF_array.reserve(voxelSet.size());
			weight_array.reserve(voxelSet.size());
			for (int i = 0; i < voxelSet.size(); ++i){
				int size = voxelSet[i].width * voxelSet[i].height * voxelSet[i].depth;
				TSDF_array.push_back(cv::Mat(1, size, CV_32F, cv::Scalar(0)));
				weight_array.push_back(cv::Mat(1, size, CV_32F, cv::Scalar(0)));
			}
			
			//integrate_volume(bpdv, snhMap, cylinderVector, depthMat, camera_intrinsic, voxelSet);


			//display_points_and_volumes(bpdv, snhMap, cylinderVector, pointCloud, voxelSet, camera_extrinsic, camera_intrinsic);
			snhMap.clear();
		}

		//PointMap pointMap(win_width, win_height);
		//read_depth_image(depthMat, camera_intrinsic, pointMap);
		//cv::Mat pointCloud(4, pointMap.mvPointLocations.size(), CV_32F);
		//read_points(pointMap, pointCloud);

		cv_draw_and_build_skeleton(&gen_root, camera_extrinsic, camera_intrinsic, &snhMap);

		//integrate_volume(bpdv, snhMap, cylinderVector, depthMat, camera_extrinsic, camera_intrinsic, voxelSet, TSDF_array, weight_array, voxel_size);

		std::vector<cv::Mat> bodypart_transforms(bpdv.size());
		for (int i = 0; i < bpdv.size(); ++i){
			bodypart_transforms[i] = get_bodypart_transform(bpdv[i], snhMap);
		}

		std::vector<Grid3D<char>> voxel_assignments = assign_voxels_to_body_parts(bpdv, bodypart_transforms, cylinderVector, depthMat, camera_extrinsic, camera_intrinsic, voxelSet, voxel_size);

		cv::Mat camera_intrinsic_inv = camera_intrinsic.inv();
		cv::Mat camera_extrinsic_inv = camera_extrinsic.inv();

		for (int i = 0; i < bpdv.size(); ++i){
			integrate_volume(bodypart_transforms[i], voxel_assignments[i], depthMat, camera_extrinsic, camera_extrinsic_inv, camera_intrinsic, camera_intrinsic_inv, voxelSet[i], TSDF_array[i], weight_array[i], voxel_size, voxel_size);
		}

		cv::Mat volume_im(win_height, win_width, CV_8UC3, cv::Scalar(0, 0, 0));
		
		for (int i = 0; i < bpdv.size();++i){
			//cv_draw_volume(bpdv[i], cylinderVector[i].width, cylinderVector[i].height, colorMat, camera_extrinsic, camera_intrinsic, snhMap);
			cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
			voxel_draw_volume(colorMat, color, bodypart_transforms[i], camera_intrinsic, &voxelSet[i], voxel_size);
			voxel_draw_volume(volume_im, color, bodypart_transforms[i], camera_intrinsic, &voxelSet[i], voxel_size);
			//cv_draw_bodypart_cylinder(*it, colorMat, camera_extrinsic, camera_intrinsic, snhMap);
		}

		//voxelset_matrix_to_array(voxelSet);

		

		snhMap.clear();

		cv::imshow("color", colorMat);
		cv::imshow("volumes", volume_im);

		//cv::Mat depthMat;
		//fs["depth"] >> depthMat;
		//
		//cv::Mat depthHSV = depth_to_HSV(depthMat);
		//
		//cv::imshow("depth", depthHSV);

		cv::waitKey(1);
		++i;

		prevRoot = gen_root;

		std::cout << i << "\n";
	}

	std::stringstream voxel_recons_SS;

	voxel_recons_SS << video_directory << "/voxels.xml.gz";

	save_voxels(voxel_recons_SS.str(), cylinderVector, voxelSet, TSDF_array, weight_array, voxel_size);

}