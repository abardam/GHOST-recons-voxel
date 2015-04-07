#pragma once

#include <opencv2\opencv.hpp>

#include "recons_voxel_body.h"

typedef cv::Vec4f XYZ;

typedef struct {
	XYZ p[3];
} TRIANGLE;

typedef struct {
	XYZ p[8];
	double val[8];
} GRIDCELL;

/*
Given a grid cell and an isolevel, calculate the triangular
facets required to represent the isosurface through the cell.
Return the number of triangular facets, the array "triangles"
will be loaded up with the vertices at most 5 triangular facets.
0 will be returned if the grid cell is either totally above
of totally below the isolevel.
*/
int Polygonise(GRIDCELL grid, double isolevel, TRIANGLE *triangles);

/*
Linearly interpolate the position where an isosurface cuts
an edge between two vertices, each with their own scalar value
*/
XYZ VertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2);

std::vector<TRIANGLE> marchingcubes_bodypart(const VoxelMatrix& volume, float voxel_size);
std::vector<TRIANGLE> marchingcubes_bodypart(const VoxelMatrix& volume, const cv::Mat& TSDF, float voxel_size);