#include "recons_voxel_integration.h"
#include "recons_voxel_body.h"
#include <recons_common.h>
#include <cv_draw_common.h>
#include <cv_pointmat_common.h>

#define SIGNUM(x)((x > 0) - (x < 0))
#define TSDF_MAX 100000000
#define TSDF_MIN -100000000


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
	float voxel_size){

	std::vector<Grid3D<char>> voxelAssignments(bpdv.size());

	//i guess find the transform for each body part individually
	//then mark each pt if it falls within the range of a body part
	//then if a pt falls under multiple body parts, compare the range
	// use two vectors that are numpoint-sized: which body part, and the distance

	PointMap pointMap(depth.cols, depth.rows);
	read_depth_image(depth, camera_matrix, pointMap);
	cv::Mat pointCloud(4, pointMap.mvPointLocations.size(), CV_32F);
	read_points_pointcloud(pointMap, pointCloud);

	std::vector<unsigned int> bodyPartAssignments(pointCloud.cols, unsigned int(bpdv.size()));
	std::vector<float> bodyPartDistances(pointCloud.cols);
	std::vector<cv::Vec3s> voxelCoordinates(pointCloud.cols, cv::Vec3s(0, 0, 0));

	for (int i = 0; i < bpdv.size(); ++i){
		//copied and modified from recons_cylinder.cpp (cylinder_fitting())

		voxelAssignments[i] = make_Grid3D<char>(volumeSet[i].width, volumeSet[i].height, volumeSet[i].depth, 0);

		//first get the inverse of the body part transform
		cv::Mat bp_transform_inv = bodypart_transforms[i].inv();

		RadiusSettings radii(cylinderVector[i].width, cylinderVector[i].height, 0);

		const cv::Mat voxel_transform_inv = get_voxel_transform(volumeSet[i].width, volumeSet[i].height, volumeSet[i].depth, voxel_size).inv();

		//in order to transform the point cloud into the body part's/voxel's coordinate system
		cv::Mat pc_bpcs = bp_transform_inv * pointCloud;
		cv::Mat pc_vcs = voxel_transform_inv * pc_bpcs;

		{
			cv::Mat pc_vcs_t = pc_vcs.t();
			cv::Mat pc_vcs_r = pc_vcs_t.reshape(4, 1);

			cv::Mat pc_bpcs_t = pc_bpcs.t();
			cv::Mat pc_bpcs_r = pc_bpcs_t.reshape(4, 1);

			for (int j = 0; j < pc_vcs_r.cols; ++j){
				cv::Vec4f& pt_v4 = pc_vcs_r.ptr<cv::Vec4f>()[j];
				cv::Vec4f& pt_bp4 = pc_bpcs_r.ptr<cv::Vec4f>()[j];

				int x = pt_v4(0);
				int y = pt_v4(1);
				int z = pt_v4(2);

				if (0 <= x && x < volumeSet[i].width &&
					0 <= y && y < volumeSet[i].height &&
					0 <= z && z < volumeSet[i].depth){

					//get the distance from the cylinder using pc_bpcs

					float dist = dist_radius(pt_bp4, radii);
					if (bodyPartAssignments[j] == bpdv.size() ||
						bodyPartDistances[j] > dist){

						if (bodyPartAssignments[j] != bpdv.size()){ //aka it has been assigned to before
							unsigned int prev = bodyPartAssignments[j];
							unsigned int prev_x = voxelCoordinates[j](0);
							unsigned int prev_y = voxelCoordinates[j](1);
							unsigned int prev_z = voxelCoordinates[j](2);

							voxelAssignments[prev][prev_x][prev_y][prev_z] = 2;
						}

						bodyPartAssignments[j] = i;
						bodyPartDistances[j] = dist;

						voxelCoordinates[j] = cv::Vec3s(x, y, z);

						voxelAssignments[i][x][y][z] = 1;
					}
				}
			}
		}
	}

	return voxelAssignments;
}

//integrate volume with the entire voxel set
void integrate_volume_all
(const BodyPartDefinitionVector& bpdv,
const std::vector<cv::Mat>& bodypart_transforms,
const std::vector<Cylinder>& cylinderVector,
const std::vector<Grid3D<char>> voxel_assignments,
const cv::Mat& depth,
const cv::Mat& camera_pose,
const cv::Mat& camera_matrix,
std::vector<VoxelMatrix>& volumeSet,
std::vector<cv::Mat>& TSDF_array,
std::vector<cv::Mat>& weight_array,
float voxel_size,
float TSDF_MU){
	

	cv::Mat camera_pose_inv = camera_pose.inv();
	cv::Mat translation = camera_pose(cv::Range(0, 4), cv::Range(3, 4));

	cv::Mat camera_matrix_inv = camera_matrix.inv();

#if DEBUG_DRAW_PTS
	//super debug
	cv::Mat debug_image(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat temp_rot;
	{


		{
			cv::Mat temp_trans = cv::Mat::eye(4, 4, CV_32F);
			cv::reduce(pointCloud, temp_trans(cv::Range(0, 4), cv::Range(3, 4)), 1, CV_REDUCE_AVG);
			cv::Vec3f rot_v(0, CV_PI / 2, 0);
			cv::Mat temp_rot_;
			cv::Rodrigues(rot_v, temp_rot_);

			temp_rot = cv::Mat::eye(4, 4, CV_32F);
			temp_rot_.copyTo(temp_rot(cv::Range(0, 3), cv::Range(0, 3)));

			temp_rot = temp_trans * temp_rot * temp_trans.inv();
		}

		cv::Mat rotated_pointcloud = temp_rot * pointCloud;

		draw_points_on_image(rotated_pointcloud, camera_matrix, debug_image, cv::Vec3b(0x00, 0x55, 0xff));

		//for (int i = 0; i < bpdv.size(); ++i){
		//	//cv_draw_volume(bpdv[i], cylinderVector[i].width, cylinderVector[i].height, image, camera_pose, camera_matrix, snhMap);
		//	//voxel_draw_volume(image, bpdv[i], snhMap, camera_pose, camera_matrix, voxels[i]);
		//	VoxelMatrix * vs_m = dynamic_cast<VoxelMatrix*>(volumeSet[i]);
		//	cv::Mat voxelPC = temp_rot * get_bodypart_transform(bpdv[i], snhMap) * get_voxel_transform(vs_m->width, vs_m->height, vs_m->depth) * vs_m->voxels;
		//	cv::Vec3b color(bpdv[i].mColor[0] * 255,
		//		bpdv[i].mColor[1] * 255,
		//		bpdv[i].mColor[2] * 255);
		//	draw_points_on_image(voxelPC, camera_matrix, debug_image, color);
		//}
	}
#endif

	//do some TSDF shit
	for (int i = 0; i < bpdv.size(); ++i){

#if DEBUG_DRAW_PTS
		//debug
		cv::Mat colorDepth = depth_to_HSV(depth);
		cv_draw_volume(bpdv[i], cylinderVector[i].width, cylinderVector[i].height, colorDepth, camera_pose, camera_matrix, snhMap);
#endif

		const cv::Mat bp_transform = bodypart_transforms[i];
		const cv::Mat voxel_transform = get_voxel_transform(volumeSet[i].width, volumeSet[i].height, volumeSet[i].depth, voxel_size);
		const cv::Mat global_transform = bp_transform * voxel_transform;


		if (volumeSet[i].type == VoxelSet::Type::Matrix){
			VoxelMatrix * vs_m = &(volumeSet[i]);

			//debug
			std::cout << "w: " << vs_m->width << " h: " << vs_m->height << " d: " << vs_m->depth << " size: " << vs_m->voxel_coords.cols << std::endl;
		
			cv::Mat pts_screen = global_transform * vs_m->voxel_coords;

			cv::Mat global_pts = camera_pose_inv * pts_screen;

			cv::Mat pts_2D = camera_matrix * pts_screen;

			divide_pointmat_by_z(pts_2D);

			for (int j = 0; j < global_pts.cols; ++j){

				int v_x = vs_m->voxel_coords.ptr<float>(0)[j];
				int v_y = vs_m->voxel_coords.ptr<float>(1)[j];
				int v_z = vs_m->voxel_coords.ptr<float>(2)[j];

				//if (voxelAssignments[i][v_x][v_y][v_z] == 2){
				//	//std::cout << "continue\n";
				//	continue; //i.e. only proceed to the next part if the voxel has been assigned to the body part, or if it hasn't been touched
				//}

				bool pixel_assigned_to_body_part = voxel_assignments[i][v_x][v_y][v_z] == 2;

				cv::Mat pt = global_pts(cv::Range(0, 4), cv::Range(j, j + 1));
				cv::Mat pt_screen = pts_screen(cv::Range(0, 4), cv::Range(j, j + 1));
				cv::Mat pt_2d = pts_2D(cv::Range(0, 4), cv::Range(j, j + 1));

				int x = pt_2d.ptr<float>(0)[0];
				int y = pt_2d.ptr<float>(1)[0];

				float lambda = cv::norm(pt_screen);
				float pt_based_depth = cv::norm(translation - pt) / lambda;

				float depth_val = depth.ptr<float>(y)[x];



				cv::Mat depth_pt(cv::Vec4f(x, y, 1, 1));
				cv::Mat repro_depth_pt = cv::Mat::diag(cv::Mat(cv::Vec4f(depth_val, depth_val, depth_val, 1))) * camera_matrix_inv * depth_pt;


				float eta = lambda - cv::norm(repro_depth_pt);

				//std::cout << "z: " << pt_screen.ptr<float>(2)[0] << "\tdepth: " << depth.ptr<float>(y)[x] << "\teta: " << eta << std::endl;
#if DEBUG_DRAW_PTS
				colorDepth.ptr<cv::Vec3b>(y)[x] = eta > 0 ? cv::Vec3b(0xff, 0, 0) : cv::Vec3b(0, 0, 0xff);

				//super debug
				{
					cv::Mat debug_image_clone = debug_image.clone();
					cv::Mat transformed_pt = camera_matrix * temp_rot * pt_screen;
					int t_x = transformed_pt.ptr<float>(0)[0] / transformed_pt.ptr<float>(2)[0];
					int t_y = transformed_pt.ptr<float>(1)[0] / transformed_pt.ptr<float>(2)[0];

					debug_image_clone.ptr<cv::Vec3b>(t_y)[t_x] = cv::Vec3b(0xff, 0xff, 0xff);

					cv::Mat depth_pt(cv::Vec4f(x, y, 1, 1));
					depth_pt = camera_matrix.inv() * depth_pt;
					depth_pt.ptr<float>(0)[0] *= depth_val;
					depth_pt.ptr<float>(1)[0] *= depth_val;
					depth_pt.ptr<float>(2)[0] *= depth_val;

					std::cout << "ptnorm: " << lambda << " depthnorm: " << cv::norm(depth_pt) << std::endl;

					depth_pt = camera_matrix * temp_rot * depth_pt;
					int t_dx = depth_pt.ptr<float>(0)[0] / depth_pt.ptr<float>(2)[0];
					int t_dy = depth_pt.ptr<float>(1)[0] / depth_pt.ptr<float>(2)[0];

					debug_image_clone.ptr<cv::Vec3b>(t_dy)[t_dx] = cv::Vec3b(0xff, 0xff, 0xff);

					cv::imshow("debug", debug_image_clone);
				}

				cv::imshow("pts", colorDepth);
				cv::waitKey(1);
#endif

				//if (-eta >= -TSDF_MU  /*&& depth_val < 0*/){
				float abs_eta = abs(eta);
				if (abs_eta < TSDF_MU && pixel_assigned_to_body_part || abs_eta > TSDF_MU){
					float tsdf = std::min(1.f, eta / TSDF_MU);
					float weight = 1;

					if (TSDF_array[i].ptr<float>()[j] == TSDF_MAX ||
						TSDF_array[i].ptr<float>()[j] == TSDF_MIN){
						TSDF_array[i].ptr<float>()[j] = 0;
					}

					TSDF_array[i].ptr<float>()[j] =
						(weight_array[i].ptr<float>()[j] * TSDF_array[i].ptr<float>()[j] +
						weight * tsdf) /
						(weight_array[i].ptr<float>()[j] + weight);

					weight_array[i].ptr<float>()[j] += weight;


				}
				else{

				}
			}

			//std::cout << "------------\n";

			for (int j = 0; j < volumeSet[i].voxel_data.cols; ++j){
				volumeSet[i].voxel_data.ptr<cv::Vec4b>()[j](3) = 0;
			}

			//TO DO: find actual zero crossings in a 3D grid
			//probably we'll only go in the z direction

			for (int k = 0; k < TSDF_array[i].cols; k += volumeSet[i].depth)
			{

				//step 1: find non-zero
				std::vector<unsigned int> nonzero_entries;
				std::vector<float> tsdf_weight_array;

				for (int j = 0; j < volumeSet[i].depth; ++j){
					float tsdf_weight = weight_array[i].ptr<float>()[j + k] * TSDF_array[i].ptr<float>()[j + k];
					if (tsdf_weight != 0){
						nonzero_entries.push_back(j + k);
						tsdf_weight_array.push_back(tsdf_weight);
					}
				}

				if (tsdf_weight_array.empty()) continue;

				//step 2: signum

				for (int j = 0; j < tsdf_weight_array.size(); ++j){
					tsdf_weight_array[j] = SIGNUM(tsdf_weight_array[j]);
				}

				//step 3: compare adjacent entries

				std::vector<int> adjacent_difference(tsdf_weight_array.size() - 1);

				for (int j = 0; j < tsdf_weight_array.size() - 1; ++j){
					adjacent_difference[j] = tsdf_weight_array[j] - tsdf_weight_array[j + 1];
				}

				//step 3: find non-zero
				std::vector<unsigned int> zero_crossings;

				for (int j = 0; j < adjacent_difference.size(); ++j){
					if (adjacent_difference[j] != 0) zero_crossings.push_back(nonzero_entries[j]);
				}


				for (int j = 0; j < zero_crossings.size(); ++j){
					volumeSet[i].voxel_data.ptr<cv::Vec4b>()[zero_crossings[j]](3) = 0xff;
				}
			}

		}
	}
}

void integrate_volume
	(const cv::Mat& bodypart_transform,
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
	int depth_multiplier){


	cv::Mat translation = camera_pose(cv::Range(0, 4), cv::Range(3, 4));

	const cv::Mat voxel_transform = get_voxel_transform(volume.width, volume.height, volume.depth, voxel_size);
	const cv::Mat global_transform = bodypart_transform * voxel_transform;


	if (volume.type == VoxelSet::Type::Matrix){
		VoxelMatrix * vs_m = &(volume);

		//debug
		std::cout << "w: " << vs_m->width << " h: " << vs_m->height << " d: " << vs_m->depth << " size: " << vs_m->voxel_coords.cols << std::endl;

		cv::Mat pts_screen = global_transform * vs_m->voxel_coords;

		cv::Mat global_pts = camera_pose_inv * pts_screen;

		cv::Mat pts_2D = camera_matrix * pts_screen;

		divide_pointmat_by_z(pts_2D);

		//debug
		//cv::Mat debug_im(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));
		//draw_pointmat_on_image(debug_im, pts_2D, cv::Vec3b(255, 0, 0));

		for (int j = 0; j < global_pts.cols; ++j){

			int v_x = vs_m->voxel_coords.ptr<float>(0)[j];
			int v_y = vs_m->voxel_coords.ptr<float>(1)[j];
			int v_z = vs_m->voxel_coords.ptr<float>(2)[j];

			bool pixel_assigned_to_body_part = voxel_assignment[v_x][v_y][v_z] == 2;

			cv::Mat pt = global_pts(cv::Range(0, 4), cv::Range(j, j + 1));
			cv::Mat pt_screen = pts_screen(cv::Range(0, 4), cv::Range(j, j + 1));
			cv::Mat pt_2d = pts_2D(cv::Range(0, 4), cv::Range(j, j + 1));

			int x = pt_2d.ptr<float>(0)[0];
			int y = pt_2d.ptr<float>(1)[0];


			if (!CLAMP(x, y, depth.cols, depth.rows)){
				continue;
			}

			//cv::Mat _debug_im = debug_im.clone();
			//_debug_im.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0, 255, 0);
			//cv::imshow("debug", _debug_im);
			//cv::waitKey(1);

			//float lambda = cv::norm(pt_screen);
			//float pt_based_depth = cv::norm(translation - pt) / lambda;

			float depth_val = depth.ptr<float>(y)[x];

			cv::Mat depth_pt(cv::Vec4f(x, y, 1, 1));
			cv::Mat repro_depth_pt = cv::Mat::diag(cv::Mat(cv::Vec4f(depth_val, depth_val, depth_val, 1))) * camera_matrix_inv * depth_pt;


			float eta = (-1) * (depth_multiplier) * (cv::norm(repro_depth_pt) - cv::norm(pt_screen)); //depth multiplier is negative if depth is negative

			if (depth_val == 0) eta = TSDF_MU+1; //handles "infinite" depth (i.e. in synthetic data)


			float abs_eta = abs(eta);
			if (eta > -TSDF_MU && !pixel_assigned_to_body_part || eta < -TSDF_MU && pixel_assigned_to_body_part){
				float tsdf = SIGNUM(eta) * std::min(1.f, abs_eta / TSDF_MU); //try signum
				//float weight = eta < -TSDF_MU? 0.05:1;
				float weight = 1;

				if (TSDF.ptr<float>()[j] == TSDF_MAX ||
					TSDF.ptr<float>()[j] == TSDF_MIN){
					TSDF.ptr<float>()[j] = 0;
				}

				TSDF.ptr<float>()[j] =
					(TSDF_weight.ptr<float>()[j] * TSDF.ptr<float>()[j] +
					weight * tsdf) /
					(TSDF_weight.ptr<float>()[j] + weight);

				TSDF_weight.ptr<float>()[j] += weight;


			}
			else{
			}
		}

		//std::cout << "------------\n";

		for (int j = 0; j < volume.voxel_data.cols; ++j){
			volume.voxel_data.ptr<cv::Vec4b>()[j](3) = 0;
		}

		//TO DO: find actual zero crossings in a 3D grid
		//probably we'll only go in the z direction

		for (int k = 0; k < TSDF.cols; k += volume.depth)
		{

			//step 1: find non-zero
			std::vector<unsigned int> nonzero_entries;
			std::vector<float> tsdf_weight_array;

			for (int j = 0; j < volume.depth; ++j){
				float tsdf_weight = TSDF_weight.ptr<float>()[j + k] * TSDF.ptr<float>()[j + k];
				if (tsdf_weight != 0){
					nonzero_entries.push_back(j + k);
					tsdf_weight_array.push_back(tsdf_weight);
				}
			}

			if (tsdf_weight_array.empty()) continue;

			//step 2: signum

			for (int j = 0; j < tsdf_weight_array.size(); ++j){
				tsdf_weight_array[j] = SIGNUM(tsdf_weight_array[j]);
			}

			//step 3: compare adjacent entries

			std::vector<int> adjacent_difference(tsdf_weight_array.size() - 1);

			for (int j = 0; j < tsdf_weight_array.size() - 1; ++j){
				adjacent_difference[j] = tsdf_weight_array[j] - tsdf_weight_array[j + 1];
			}

			//step 3: find non-zero
			std::vector<unsigned int> zero_crossings;

			for (int j = 0; j < adjacent_difference.size(); ++j){
				if (adjacent_difference[j] != 0) zero_crossings.push_back(nonzero_entries[j]);
			}


			for (int j = 0; j < zero_crossings.size(); ++j){
				volume.voxel_data.ptr<cv::Vec4b>()[zero_crossings[j]](3) = 0xff;
			}
		}

	}
	
}


void save_voxels(const std::string& filename, const std::vector<Cylinder>& cylinders, const std::vector<VoxelMatrix>& volumes, const std::vector<cv::Mat>& TSDF_array, const std::vector<cv::Mat>& weight_array, const float voxel_size){
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::WRITE);

	fs << "num_bodyparts" << (int)cylinders.size()
		<< "voxelsize" << voxel_size
		<< "bodyparts" << "[";

	for (int i = 0; i < cylinders.size(); ++i){
		fs << "{";

		fs << "cylinder_width" << cylinders[i].width
			<< "cylinder_height" << cylinders[i].height
			<< "volume_width" << volumes[i].width
			<< "volume_height" << volumes[i].height
			<< "volume_depth" << volumes[i].depth;

		fs << "surface" << volumes[i].voxel_data;
		fs << "TSDF" << TSDF_array[i];
		fs << "weight" << weight_array[i];

		fs << "}";
	}

	fs << "]";

	fs.release();
}


void load_voxels(const std::string& filename, std::vector<Cylinder>& cylinders, std::vector<VoxelMatrix>& volumes, std::vector<cv::Mat>& TSDF_array, std::vector<cv::Mat>& weight_array, float& voxel_size){
	cv::FileStorage fs;
	fs.open(filename, cv::FileStorage::READ);

	int num_bodyparts;
	fs["num_bodyparts"] >> num_bodyparts;
	fs["voxelsize"] >> voxel_size;

	cv::FileNode bodyparts = fs["bodyparts"];
	for (auto it = bodyparts.begin(); it != bodyparts.end(); ++it){
		Cylinder c;
		(*it)["cylinder_width"] >> c.width;
		(*it)["cylinder_height"] >> c.height;
		cylinders.push_back(c);

		int x, y, z;
		(*it)["volume_width"] >> x;
		(*it)["volume_height"] >> y;
		(*it)["volume_depth"] >> z;
		
		VoxelMatrix vs_m(x, y, z, true);
		volumes.push_back(vs_m);

		cv::Mat voxel_data;
		(*it)["surface"] >> voxel_data;
		volumes.back().voxel_data = voxel_data.clone();

		cv::Mat TSDF;
		(*it)["TSDF"] >> TSDF;
		cv::Mat weight;
		(*it)["weight"] >> weight;

		TSDF_array.push_back(TSDF);
		weight_array.push_back(weight);
	}

	fs.release();
}