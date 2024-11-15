#pragma once
#ifndef TEST_H
#define TEST_H


/**
*	@file test.h 
*	@brief ´úÂë²âÊÔÄ£¿é
*	Datails.
*	
*	@author johan
*	@email 2806762759@qq.com
*	@data 2024.11.14
*/
#include"config.h"
#include"video_analytics.h"
#include"label_read.h"
#include"image_analytics.h"
/**
*	@brief opencv
*/
void opencv_test();

/**
*	@brief opencv_cuda
*/
void opencv_cuda_test();

/**
*	@brief video_analytics.h VideoAnalytics::VideoPlay
*/
void VideoAnalytics_VideoPlay_test();

/**
*	@brief video_analytics.h BackGraundAnalytics::VideoaNalyse
*/
void BackGraundAnalytics_VideoaNalyse_test();

/**
*	@brief video_analytics.h OpticalFlowAnalytics::VideoaNalyse
*/
void OpticalFlowAnalytics_VideoaNalyse_test();

/**
*	@brief video_analytics.h ObjDetect_yolov8::VideoObjDetect
*/
void ObjDetect_yolov8_VideoObjDetect_test();


/**
*	@brief label_read.h LabelObj::ClassseGet
*/

void LabelObj_ClassseGet_test();

/**
*	@brief image_analytics.h ImageAnalytics::ImagePlay
*/

void ImageAnalytics_ImagePlay_test();

/**
*	@brief image_analytics.h ImageFeatureMatch::ImagePlay
*/
void ImageFeatureMatch_ImagePlay_test();

/**
*	@brief image_analytics.h ImageFeatureMatch::ImageNalyse
*/
void ImageAnalytics_ImageNalyse_test();

#endif // !TEST_H_