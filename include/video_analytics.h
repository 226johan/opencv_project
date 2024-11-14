#pragma once

#ifndef VIDEO_ANALYTICS_H
#define VIDEO_ANALYTICS_H
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
#include <opencv2/cudaarithm.hpp>	// cuda::split cuda::cartToPolar
#include "label_read.h"


class VideoAnalytics {
public:
	VideoAnalytics() = default;
	explicit VideoAnalytics(const VideoCapture& video) : video_(video) {
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
	}

	explicit VideoAnalytics(const string video_name) : video_name_(video_name) {
		if (video_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty video file");
		}
	}


	explicit VideoAnalytics(const VideoCapture& video,const string model_name,const string label_name) : 
		video_(video),model_name_(model_name),label_name_(label_name){
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
	}

	explicit VideoAnalytics(const string video_name, const string model_name, const string label_name) :
		video_name_(video_name), model_name_(model_name), label_name_(label_name) {
		if (video_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty video file");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
	}


	const VideoCapture VideoGet(const VideoCapture& video) { return video; }
	const string VideoGet(const string video_name) { video_name_ = video_name; return video_name_; }
	void VideoPlay();
	virtual void VideoaNalyse() {};
	virtual void VideoObjDetect() {};
	virtual ~VideoAnalytics() = default;

	string video_name_;
	string model_name_;
	string label_name_;
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

	explicit BackGraundAnalytics(const string video_name) : VideoAnalytics(video_name) {
		if (video_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty video file");
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

	explicit OpticalFlowAnalytics(const string video_name) : VideoAnalytics(video_name) {
		if (video_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty video file");
		}
	}

	~OpticalFlowAnalytics() = default;
	void VideoaNalyse();
};

class ObjDetect_yolov8 : public VideoAnalytics {
public:
	ObjDetect_yolov8() = default;
	explicit ObjDetect_yolov8(const VideoCapture video) : VideoAnalytics(video) {
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
	}

	explicit ObjDetect_yolov8(const string video_name) : VideoAnalytics(video_name) {
		if (video_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty video file");
		}
	}

	explicit ObjDetect_yolov8(const VideoCapture& video, const string model_name, const string label_name, bool use_cuda=true) :VideoAnalytics(video,model_name,label_name),use_cuda_(use_cuda) {
		if (!video_.isOpened()) {
			throw std::runtime_error("Error: Could not open video.");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
	}

	explicit ObjDetect_yolov8(const string video_name, const string model_name, const string label_name, bool use_cuda = true) :VideoAnalytics(video_name, model_name, label_name), use_cuda_(use_cuda) {
		if (video_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty video file");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
	}


	void VideoObjDetect();
	~ObjDetect_yolov8() = default;

private:
	LabelObj label_obj_;
	bool use_cuda_;
};

#endif