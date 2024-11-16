#pragma once
#ifndef IMAGE_ANALYTICS_H
#define IMAGE_ANALYTICS_H
/**
*	@file image_analytics.h
*	@brief 图片分析
*	Datails.
*
*	@author johan
*	@email 2806762759@qq.com
*	@data 2024.11.15
*/

// video analytics template
/*****************************************************************************************
	namedWindow("img");
	bool isopen = true;
	while (isopen)
	{
	int64 start = getTickCount();
// ********************************* your code start *************************************


// ********************************* your code end ***************************************
	int64 end = getTickCount();
	double runtime = 1000.0f/(end-start);
	putText(image_host_, format("runtime: %.2f", 1000), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("img", image_host_);
	char key = waitKey(1);
		switch (key)
		{
		case 27:
			isopen = false;
			break;
		case 'q':
			isopen = false;
			break;
		}
	}
******************************************************************************************/


// todolists. 添加多张图片读取处理功能

#include"config.h"
#include"label_read.h"
#include <opencv2/cudafeatures2d.hpp> // orb特征子
class ImageAnalytics {
public:
	ImageAnalytics() = default;

	explicit ImageAnalytics(const string image_name) : image_name_(image_name) {
		if (image_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty image file");
		}
		image_host_ = imread(image_name);
		image_.upload(image_host_);
	}

	explicit ImageAnalytics(const Mat& image_host) : image_host_(image_host) {
		if (image_host_.empty()) {
			throw std::runtime_error("Error: empty image_host");
		}
		image_.upload(image_host_);
	}

	explicit ImageAnalytics(const GpuMat& image) : image_(image) {
		if (image_.empty()) {
			throw std::runtime_error("Error: empty image");
		}
		image_.download(image_host_);
	}


	explicit ImageAnalytics(const string image_name, const string model_name, const string label_name) :
		image_name_(image_name), model_name_(model_name), label_name_(label_name) {
		if (image_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty image file");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_host_ = imread(image_name);
		image_.upload(image_host_);
		
	}

	explicit ImageAnalytics(const GpuMat& image, const string model_name, const string label_name) :
		image_(image), model_name_(model_name), label_name_(label_name) {
		if (image_.empty()) {
			throw std::runtime_error("Error: empty image");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_.download(image_host_);
		
	}

	explicit ImageAnalytics(const Mat& image_host, const string model_name, const string label_name) :
		image_host_(image_host), model_name_(model_name), label_name_(label_name) {
		if (image_host_.empty()) {
			throw std::runtime_error("Error: empty image_host");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_.upload(image_host_);
		
	}


	const string ImageGet(const string image_name);
	const Mat ImageGet(const Mat& image_host);
	const GpuMat ImageGet(const GpuMat& image);

	void ImagePlay();
	virtual void ImageNalyse() {};
	virtual void ImageObjDetect() {};
	virtual ~ImageAnalytics() = default;

	string image_name_;
	string model_name_;
	string label_name_;
	Mat image_host_;
	GpuMat image_;

};


// todolists.  ImageNalyse() 参数调节
class ImageFeatureMatch : public ImageAnalytics{
public:
	ImageFeatureMatch() = default;

	explicit ImageFeatureMatch(const string image_name) : ImageAnalytics(image_name) {
		if (image_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty image file");
		}
		image_host_ = imread(image_name);
		image_.upload(image_host_);
	}

	explicit ImageFeatureMatch(const Mat& image_host) : ImageAnalytics(image_host) {
		if (image_host_.empty()) {
			throw std::runtime_error("Error: empty image_host");
		}
		image_.upload(image_host_);
	}

	explicit ImageFeatureMatch(const GpuMat& image) : ImageAnalytics(image) {
		if (image_.empty()) {
			throw std::runtime_error("Error: empty image");
		}
		image_.download(image_host_);
	}


	explicit ImageFeatureMatch(const string image_name, const string model_name, const string label_name) :
		ImageAnalytics(image_name,model_name,label_name) {
		if (image_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty image file");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_host_ = imread(image_name);
		image_.upload(image_host_);
		label_obj_.FileRead(label_name_);
	}

	explicit ImageFeatureMatch(const GpuMat& image, const string model_name, const string label_name) :
		ImageAnalytics(image,model_name,label_name) {
		if (image_.empty()) {
			throw std::runtime_error("Error: empty image");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_.download(image_host_);
		label_obj_.FileRead(label_name_);
	}

	explicit ImageFeatureMatch(const Mat& image_host, const string model_name, const string label_name) :
		ImageAnalytics(image_host, model_name, label_name) {
		if (image_host_.empty()) {
			throw std::runtime_error("Error: empty image_host");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_.upload(image_host_);
		label_obj_.FileRead(label_name_);
	}

	LabelObj label_obj_;
	void ImageNalyse();

	~ImageFeatureMatch() = default;
};

class ImageClassification : public ImageAnalytics {
public:
	ImageClassification() = default;

	explicit ImageClassification(const string image_name) : ImageAnalytics(image_name) {
		if (image_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty image file");
		}
		image_host_ = imread(image_name);
		image_.upload(image_host_);
	}

	explicit ImageClassification(const Mat& image_host) : ImageAnalytics(image_host) {
		if (image_host_.empty()) {
			throw std::runtime_error("Error: empty image_host");
		}
		image_.upload(image_host_);
	}

	explicit ImageClassification(const GpuMat& image) : ImageAnalytics(image) {
		if (image_.empty()) {
			throw std::runtime_error("Error: empty image");
		}
		image_.download(image_host_);
	}


	explicit ImageClassification(const string image_name, const string model_name, const string label_name,bool use_cuda=true) :
		ImageAnalytics(image_name, model_name, label_name), use_cuda_(use_cuda) {
		if (image_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty image file");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_host_ = imread(image_name);
		image_.upload(image_host_);
		label_obj_.FileRead(label_name_);
	}

	explicit ImageClassification(const GpuMat& image, const string model_name, const string label_name, bool use_cuda = true) :
		ImageAnalytics(image, model_name, label_name), use_cuda_(use_cuda) {
		if (image_.empty()) {
			throw std::runtime_error("Error: empty image");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_.download(image_host_);
		label_obj_.FileRead(label_name_);
	}

	explicit ImageClassification(const Mat& image_host, const string model_name, const string label_name, bool use_cuda = true) :
		ImageAnalytics(image_host, model_name, label_name), use_cuda_(use_cuda) {
		if (image_host_.empty()) {
			throw std::runtime_error("Error: empty image_host");
		}
		if (label_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty label file");
		}
		if (model_name_.c_str() == nullptr) {
			throw std::runtime_error("Error: empty model file");
		}
		image_.upload(image_host_);
		label_obj_.FileRead(label_name_);
	}

	LabelObj label_obj_;
	bool use_cuda_;

	// model must use resnet18
	void ImageNalyse();

	~ImageClassification() = default;
};


#endif // !IMAGE_ANALYTICS_H

