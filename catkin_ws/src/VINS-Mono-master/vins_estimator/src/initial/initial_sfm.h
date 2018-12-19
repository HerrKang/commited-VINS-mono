#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
    bool state;//是否被三角化
    int id;
    vector<pair<int,Vector2d>> observation;//2d坐标
    double position[3];//3d坐标
    double depth;
};

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
    /*\brief 外部调用接口,主要处理函数. 输入第l帧和最后一帧的相对R,t, 根据特征点的观测估计所有帧的位姿和特征点的3D坐标
    \param[in] frame_num: pose的个数, elements in q,T
    \param[out] q: SFM结果,每帧在l帧参考系下的quaternion
    \param[out] T: SFM结果,每帧在l帧参考系下的position
    \param[in] l: 以第l帧为参考系,即l帧的pose为坐标原点
    \param[in] relative_R: 第l帧到最后一帧的相对旋转
    \param[in] relative_T: 第l帧到最后一帧的相对平移
    \param[in] sfm_f: feature list,每个SFMFeature中包含多个观测
    \param[out] sfm_tracked_point: 优化后的3D特征点在l帧参考系的position
    */
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
    /**\brief 根据当前已经三角化的特征点估计某一帧的R,t */
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);
    /**\brief 输入两个pose和2D观测点, 三角化3D特征点 */
	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
    /**\brief 输入两帧的pose, 三角化它们共同观测的特征点, 之前已经被三角化的特征点不再处理 */
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};