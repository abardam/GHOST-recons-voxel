#pragma once

#include <opencv2\opencv.hpp>

//#define VOXEL_SIZE 0.1
#define VOXEL_VOLUME_WIDTH 3.5
#define VOXEL_VOLUME_HEIGHT 1
#define VOXEL_VOLUME_DEPTH 3.5

unsigned int get_voxel_index(int x, int y, int z, int width, int height, int depth);

template <typename T>
using Grid3D = std::vector < std::vector <std::vector<T>> > ;

struct VolumeDimensions{
	float width, height, depth;
	VolumeDimensions(float w, float h, float d) :
		width(w), height(h), depth(d){};
};

struct Voxel{
	cv::Vec3b color;
	bool exists;

	Voxel() :exists(false){}
	Voxel(bool init_exists) : exists(init_exists){}
};

template <typename T>
Grid3D<T> make_Grid3D(int x, int y, int z, T value = T()){
	return Grid3D<T>(x, std::vector<std::vector<T>>(y, std::vector<T>(z, T(value))));
};

struct VoxelSet{
	const float width, height, depth;
	enum Type
	{
		Array,
		Matrix
	} type;

	virtual ~VoxelSet(){}

protected:
	VoxelSet(int x, int y, int z) :
		width(x), height(y), depth(z){}
};

struct VoxelMatrix;

//represents voxels as a 3D grid, allowing for easy reshaping.
struct VoxelArray :
	public VoxelSet{
	Grid3D<Voxel> voxels;

	//initializes empty voxel array
	VoxelArray(int x, int y, int z, bool init_exists = false) :
		VoxelSet(x, y, z),
		voxels(x, std::vector<std::vector<Voxel>>(y, std::vector<Voxel>(z, Voxel(init_exists))) )
	{
		type = Type::Array;
	};

	//converts a VoxelMatrix into a VoxelArray
	VoxelArray::VoxelArray(const VoxelMatrix& voxelM);

};

//represents voxels as a matrix, allowing for easy matrix ops (e.g. for rendering).
struct VoxelMatrix :
	public VoxelSet{
	cv::Mat voxel_coords;
	cv::Mat voxel_data;

	//converts a VoxelArray into a VoxelMatrix
	VoxelMatrix(const VoxelArray& voxelA);
	VoxelMatrix(int x, int y, int z, bool init_exists = false);
};

typedef std::map<std::string, unsigned int> VoxelSetMap;
typedef std::pair<std::string, unsigned int> VoxelSetEntry;


void voxelset_array_to_matrix(std::vector<VoxelSet*>& voxelSet);
void voxelset_matrix_to_array(std::vector<VoxelSet*>& voxelSet);

