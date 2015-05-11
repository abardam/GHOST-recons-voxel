#include <recons_common.h>
#include <recons_optimization.h>
#include <ReconsVoxel.h>
#include <AssimpCV.h>

SkeletonNodeHard generateFromReference(const SkeletonNodeHard * const ref, const SkeletonNodeHard * const prev){
	SkeletonNodeHard snh;
	cv::Mat generatedTransformation = skeleton_constraints_optimize(prev->mTransformation, ref->mTransformation, 1, 1);


	for (int i = 0; i < prev->mChildren.size(); ++i){
		snh.mChildren.push_back(generateFromReference(&ref->mChildren[i], &prev->mChildren[i]));
	}

	snh.mTransformation = generatedTransformation * prev->mTransformation;

	//extract scale and correct it?
	cv::Vec3f scale(
		cv::norm(prev->mTransformation(cv::Range(0, 1), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(0, 1), cv::Range(0, 3))),
		cv::norm(prev->mTransformation(cv::Range(1, 2), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(1, 2), cv::Range(0, 3))),
		cv::norm(prev->mTransformation(cv::Range(2, 3), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(2, 3), cv::Range(0, 3)))
		);

	cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	rot.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));

	snh.mName = prev->mName;
	snh.mParentName = prev->mParentName;

	return snh;
}




int main(int argc, char * argv[]){
	if (argc <= 2){
		std::cout << "Please enter directory and voxel reconstruct file\n";
		return 0;
	}

	std::string voxel_recons_path(argv[2]);

	std::vector<Cylinder> cylinders;
	std::vector<VoxelMatrix> voxels;
	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;
	float voxel_size;
	load_voxels(voxel_recons_path, cylinders, voxels, TSDF_array, weight_array, voxel_size);


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


	cv::Mat depthMat, colorMat, camera_extrinsic, camera_intrinsic;
	SkeletonNodeHardMap snhMap;
	VoxelSetMap vsMap;

	SkeletonNodeHard root;
	SkeletonNodeHard prevRoot;



	bool firstframe = true;


	while (true){
		filenameSS.str("");
		filenameSS << video_directory << "/" << i << ".xml.gz";
		fs.open(filenameSS.str(), cv::FileStorage::READ);

		double time;

		if (!load_input_frame(filenameSS.str(), time, camera_extrinsic, camera_intrinsic, root, colorMat, depthMat)) {
			break;
		}

		SkeletonNodeHard gen_root;
		if (!firstframe){
			gen_root = generateFromReference(&root, &prevRoot);
		}
		else{

			firstframe = false;

			gen_root = root;

		}


		cv_draw_and_build_skeleton(&gen_root, cv::Mat::eye(4,4,CV_32F), camera_intrinsic, camera_extrinsic, &snhMap, colorMat);

		for (int i = 0; i < bpdv.size(); ++i){

			cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
			voxel_draw_volume(colorMat, color, get_bodypart_transform(bpdv[i], snhMap, camera_extrinsic), camera_intrinsic, &voxels[i], voxel_size);
		}
		snhMap.clear();
		cv::imshow("color", colorMat);
		cv::waitKey(1);
		++i;

		prevRoot = gen_root;
	}
}