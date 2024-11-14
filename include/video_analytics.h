#pragma once
/**
*	@file config.h
*	@brief 视频分析实例类
*	Datails.
*
*	@author johan
*	@email 2806762759@qq.com
*	@data 2024.11.14
*/

#include"config.h"
#include <opencv2/cudabgsegm.hpp>  // 背景法
#include <opencv2/cudaoptflow.hpp> // 光流法
#include <opencv2/cudaarithm.hpp>	// cuda::split cuda::cartToPolar()



class VideoAnalytics {
public:
	VideoAnalytics() = default;
	explicit VideoAnalytics(const VideoCapture& video) : video_(video) {
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
	}

	const VideoCapture VideoGet(const VideoCapture& video) { return video; }
	void VideoPlay();
	virtual void VideoaNalyse() {};
	virtual ~VideoAnalytics() = default;

	string video_name_;
	VideoCapture video_;

};

class BackGraundAnalytics : public VideoAnalytics{
public:
	BackGraundAnalytics() = default;
	explicit BackGraundAnalytics(const VideoCapture video) : VideoAnalytics(video){
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
	}
	~BackGraundAnalytics() = default;
	void VideoaNalyse();
};

class OpticalFlowAnalytics : public VideoAnalytics {
public:
	OpticalFlowAnalytics() = default;
	explicit OpticalFlowAnalytics(const VideoCapture video) : VideoAnalytics(video) {
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
	}
	~OpticalFlowAnalytics() = default;
	void VideoaNalyse();
};