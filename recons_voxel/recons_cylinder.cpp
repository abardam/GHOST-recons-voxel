#include "recons_cylinder.h"
#include <cv_draw_common.h>s
#include <cv_pointmat_common.h>

float dist_radius(const cv::Vec4f& point, const RadiusSettings& radius){
	float angle = atan2(point(2), point(0));
	float x_t = radius.x * cos(angle);
	float z_t = radius.z * sin(angle);
	float dist_to_ellipse_sq = x_t * x_t + z_t * z_t;
	float dist_to_pt_sq = point(0)*point(0) + point(2)*point(2);

	return abs(dist_to_ellipse_sq - dist_to_pt_sq);
}

bool filter_height_criteria(const cv::Vec4f& point, void* float_length){
	float * length = (float*)float_length;

	float l = point(1);
	return 0 <= l && l < *length;
}

bool filter_radius_criteria(const cv::Vec4f& point, void* radius_settings_float_radius){
	RadiusSettings * radius = (RadiusSettings*)radius_settings_float_radius;
	
	return dist_radius(point, *radius) < radius->threshold_squared;
}

void cylinder_fitting(const BodyPartDefinitionVector& bpdv, const SkeletonNodeHardMap& snhMap, const cv::Mat& pointCloud, const cv::Mat& camera_pose, std::vector<Cylinder>& cylinderVector,
	float radius_search_increment, float radius_search_max, float radius_threshold,
	const std::vector<VolumeDimensions> * volume_dimensions,
	const cv::Mat * DEBUG_camera_matrix, float * DEBUG_width, float * DEBUG_height){
	bool debug = DEBUG_width != 0 && DEBUG_height != 0 && DEBUG_camera_matrix != 0;
	float radius_threshold_squared = radius_threshold * radius_threshold;

	{

		cylinderVector.resize(bpdv.size());

		cv::Mat test_;

			
		//debug
		if (debug){
			test_ = cv::Mat(*DEBUG_height, *DEBUG_width, CV_8UC3, cv::Scalar(0, 0, 0));
			cv::Mat projected = *DEBUG_camera_matrix * pointCloud;
			divide_pointmat_by_z(projected);
			draw_pointmat_on_image(test_, projected, cv::Vec3b(0, 0x55, 0xff));
		}


		for (int i = 0; i < bpdv.size(); ++i){
			float length;
			//first get the inverse of the body part transform
			cv::Mat bp_transform;
			
			if (volume_dimensions){
				bp_transform = get_bodypart_transform(bpdv[i], snhMap, camera_pose);
				length = (*volume_dimensions)[i].height;
			}
			else{
				bp_transform = get_bodypart_transform(bpdv[i], snhMap, camera_pose, &length);
			}

			cv::Mat bp_transform_inv = bp_transform.inv();

			//in order to transform the point cloud into the body part's coordinate system
			cv::Mat pc_bpcs = bp_transform_inv * pointCloud;
			cv::Mat pc_bpcs_filter;

			//now lets filter out all the pts that are not along the length of the axis
			pc_bpcs_filter = filter_pointmat(pc_bpcs, filter_height_criteria, &length);

			if (pc_bpcs_filter.empty()) continue;

			//expand dong
			{
				float bestrad_x = radius_search_increment;
				float bestrad_z = radius_search_increment;
				int bestradpts = 0;
				for (float radz = radius_search_increment; radz <= radius_search_max; radz += radius_search_increment)
				{
					for (float radx = radz; radx <= radius_search_max; radx += radius_search_increment)
					{
						RadiusSettings rads(radx, radz, radius_threshold_squared);
						cv::Mat pc_bpcs_fit = filter_pointmat(pc_bpcs_filter, filter_radius_criteria, &rads);


						if (pc_bpcs_fit.cols > bestradpts) {
							bestradpts = pc_bpcs_fit.cols;
							bestrad_x = radx;
							bestrad_z = radz;

							if (debug){

								cv::Mat assigned_pts_im = test_.clone();


								cv::Mat projected = *DEBUG_camera_matrix * get_bodypart_transform(bpdv[i], snhMap, camera_pose) * pc_bpcs_fit;
								divide_pointmat_by_z(projected);
								draw_pointmat_on_image(assigned_pts_im, projected, cv::Vec3b(0xff, 0x55, 0));
								cv::imshow(bpdv[i].mBodyPartName, assigned_pts_im);
							}
						}

#ifdef DEBUG_STUFF
						//debug
						if (!pc_bpcs_fit.empty() && debug){
							cv::Mat test = test_.clone();
							cv::Mat projected = *DEBUG_camera_matrix * get_bodypart_transform(bpdv[i], snhMap, camera_pose) * pc_bpcs_fit;
							divide_pointmat_by_z(projected);
							draw_pointmat_on_image(test, projected, cv::Vec3b(0xff, 0x55, 0));


							cv::Scalar color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);

							cv_draw_volume(color, bp_transform, length, radx*2, radz*2, test, camera_pose, *DEBUG_camera_matrix);

							cv::imshow("test", test);
							cv::waitKey(5);
						}
#endif

						//keep limbs circular
						if (
							bpdv[i].mBodyPartName != "HEAD" &&
							bpdv[i].mBodyPartName != "CHEST" &&
							bpdv[i].mBodyPartName != "ABS" &&
							bpdv[i].mBodyPartName != "HAND LEFT" &&
							bpdv[i].mBodyPartName != "HAND RIGHT" &&
							bpdv[i].mBodyPartName != "FOOT LEFT" &&
							bpdv[i].mBodyPartName != "FOOT RIGHT"
							)
							break;
					}

				}
#ifdef DEBUG_STUFF
				if (debug){
					cv::Scalar color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
					cv_draw_volume(color, bp_transform, length, bestrad_x * 2, bestrad_z * 2, test_, camera_pose, *DEBUG_camera_matrix);
				}
#endif
				cylinderVector[i].width = bestrad_x;
				cylinderVector[i].height = bestrad_z;
			}

		}
	}
}



void cylinder_fitting(const BodyPartDefinitionVector& bpdv, const SkeletonNodeAbsoluteVector& snav, const cv::Mat& pointCloud, const cv::Mat& camera_pose, std::vector<Cylinder>& cylinderVector,
	float radius_search_increment, float radius_search_max, float radius_threshold,
	const std::vector<VolumeDimensions> * volume_dimensions,
	const cv::Mat * DEBUG_camera_matrix, float * DEBUG_width, float * DEBUG_height){
	bool debug = DEBUG_width != 0 && DEBUG_height != 0 && DEBUG_camera_matrix != 0;
	float radius_threshold_squared = radius_threshold * radius_threshold;

	{

		cylinderVector.resize(bpdv.size());

		cv::Mat test_;


		//debug
		if (debug){
			test_ = cv::Mat(*DEBUG_height, *DEBUG_width, CV_8UC3, cv::Scalar(0, 0, 0));
			cv::Mat projected = *DEBUG_camera_matrix * pointCloud;
			divide_pointmat_by_z(projected);
			draw_pointmat_on_image(test_, projected, cv::Vec3b(0, 0x55, 0xff));
		}


		for (int i = 0; i < bpdv.size(); ++i){
			float length;
			//first get the inverse of the body part transform
			cv::Mat bp_transform;

			if (volume_dimensions){
				bp_transform = get_bodypart_transform(bpdv[i], snav, camera_pose);
				length = (*volume_dimensions)[i].height;
			}
			else{
				bp_transform = get_bodypart_transform(bpdv[i], snav, camera_pose, &length);
			}

			cv::Mat bp_transform_inv = bp_transform.inv();

			//in order to transform the point cloud into the body part's coordinate system
			cv::Mat pc_bpcs = bp_transform_inv * pointCloud;
			cv::Mat pc_bpcs_filter;

			//now lets filter out all the pts that are not along the length of the axis
			pc_bpcs_filter = filter_pointmat(pc_bpcs, filter_height_criteria, &length);

			if (pc_bpcs_filter.empty()) continue;

			//expand dong
			{
				float bestrad_x = radius_search_increment;
				float bestrad_z = radius_search_increment;
				int bestradpts = 0;
				for (float radz = radius_search_increment; radz <= radius_search_max; radz += radius_search_increment)
				{
					for (float radx = radz; radx <= radius_search_max; radx += radius_search_increment)
					{
						RadiusSettings rads(radx, radz, radius_threshold_squared);
						cv::Mat pc_bpcs_fit = filter_pointmat(pc_bpcs_filter, filter_radius_criteria, &rads);


						if (pc_bpcs_fit.cols > bestradpts) {
							bestradpts = pc_bpcs_fit.cols;
							bestrad_x = radx;
							bestrad_z = radz;

							if (debug){

								cv::Mat assigned_pts_im = test_.clone();


								cv::Mat projected = *DEBUG_camera_matrix * get_bodypart_transform(bpdv[i], snav, camera_pose) * pc_bpcs_fit;
								divide_pointmat_by_z(projected);
								draw_pointmat_on_image(assigned_pts_im, projected, cv::Vec3b(0xff, 0x55, 0));
								cv::imshow(bpdv[i].mBodyPartName, assigned_pts_im);
							}
						}

#ifdef DEBUG_STUFF
						//debug
						if (!pc_bpcs_fit.empty() && debug){
							cv::Mat test = test_.clone();
							cv::Mat projected = *DEBUG_camera_matrix * get_bodypart_transform(bpdv[i], snav, camera_pose) * pc_bpcs_fit;
							divide_pointmat_by_z(projected);
							draw_pointmat_on_image(test, projected, cv::Vec3b(0xff, 0x55, 0));


							cv::Scalar color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);

							cv_draw_volume(color, bp_transform, length, radx * 2, radz * 2, test, camera_pose, *DEBUG_camera_matrix);

							cv::imshow("test", test);
							cv::waitKey(5);
						}
#endif
						//keep limbs circular
						if (
							bpdv[i].mBodyPartName != "HEAD" &&
							bpdv[i].mBodyPartName != "CHEST" &&
							bpdv[i].mBodyPartName != "ABS" &&
							bpdv[i].mBodyPartName != "HAND LEFT" &&
							bpdv[i].mBodyPartName != "HAND RIGHT" &&
							bpdv[i].mBodyPartName != "FOOT LEFT" &&
							bpdv[i].mBodyPartName != "FOOT RIGHT"
							)
							break;
					}

				}

#ifdef DEBUG_STUFF
				if (debug){
					cv::Scalar color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
					cv_draw_volume(color, bp_transform, length, bestrad_x * 2, bestrad_z * 2, test_, camera_pose, *DEBUG_camera_matrix);

				}
#endif
				cylinderVector[i].width = bestrad_x;
				cylinderVector[i].height = bestrad_z;
			}

		}
	}
}