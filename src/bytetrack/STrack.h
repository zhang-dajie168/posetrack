#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

enum TrackState
{
	New = 0,
	Tracked,
	Lost,
	Removed
};

class STrack
{
public:
	STrack(std::vector<float> tlwh_, float score);
	~STrack();

	std::vector<float> static tlbr_to_tlwh(std::vector<float> &tlbr);
	void static multi_predict(std::vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	std::vector<float> tlwh_to_xyah(std::vector<float> tlwh_tmp);
	std::vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();

	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

	// 添加设置关键点的方法
	void set_keypoints(const std::vector<cv::Point2f> &kpts, const std::vector<float> &kpts_scores)
	{
		keypoints = kpts;
		keypoints_score = kpts_scores;
	}

public:
	bool is_activated;
	int track_id;
	int state;

	std::vector<float> _tlwh;
	std::vector<float> tlwh;
	std::vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;

	// 添加关键点相关成员
	std::vector<cv::Point2f> keypoints;
	std::vector<float> keypoints_score;

private:
	byte_kalman::KalmanFilter kalman_filter;
};