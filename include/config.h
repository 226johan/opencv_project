#pragma once

/**
*	@file config.h
*	@brief �궨�壬��ͷģ��
*	Datails.
*
*	@author johan
*	@email 2806762759@qq.com
*	@data 2024.11.14
*/
#include <opencv2/opencv.hpp>
#include<opencv2/cudaimgproc.hpp>
#include<iostream>

using namespace cv;
using namespace cv::cuda;
using namespace std;


#define IMG_FILE "F:/Dev/opencv/data/images/"
#define VIDEO_FILE "F:/Dev/opencv/data/videos/"
#define MODEL_FILE "F:/Dev/opencv/data/models/"
#define LABEL_FILE "F:/Dev/opencv/data/labels/"

#define IMG_FILE_NAME "lena.jpg"
#define VIDEO_FILE_NAME "vtest.avi"
#define MODEL_FILE_NAME ""
#define LABEL_FILE_NAME ""