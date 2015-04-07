#pragma once

#include <AssimpOpenGL.h>

#define CYLINDER_FITTING_THRESHOLD_SQ 0.16
#define CYLINDER_FITTING_THRESHOLD 0.4
#define CYLINDER_FITTING_THRESHOLD_ELLIPTIC 0.3

#define CYLINDER_FITTING_RADIUS_SEARCH_MAX 2
#define CYLINDER_FITTING_RADIUS_SEARCH_INCREMENT 0.1

struct Cylinder{
	//this is the width and height of the ellipse, not the cylinder. the cylinder's height depends on the length of the body part it is associated with.
	float width, height;

	Cylinder(float w, float h):
		width(w), height(h){}

	Cylinder() :
		width(0), height(0){}
};

void cylinder_fitting(const BodyPartDefinitionVector& bpdv, const SkeletonNodeHardMap& snhMap, const cv::Mat& pointCloud, const cv::Mat& camera_pose, std::vector<Cylinder>& cylinderVector,
	const cv::Mat * DEBUG_camera_matrix = 0, float * DEBUG_width = 0, float * DEBUG_height = 0);

float dist_radius(const cv::Vec4f& point, const std::pair<float, float>& radius);

bool filter_height_criteria(const cv::Vec4f& point, void* float_length);
bool filter_radius_criteria(const cv::Vec4f& point, void* pair_float_radius);