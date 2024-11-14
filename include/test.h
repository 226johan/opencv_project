#pragma once
/**
*	@file test.h 
*	@brief opencv 运行环境测试模块
*	Datails.
*	
*	@author johan
*	@email 2806762759@qq.com
*	@data 2024.11.14
*/
#include"config.h"

/**
*	@brief opencv基础模块测试
*/
void opencv_test();

/**
*	@brief opencv_cuda模块测试
*/
void opencv_cuda_test();

/**
*	@brief video_analytics.h VideoAnalytics::VideoPlay 模块测试
*/
void VideoAnalytics_VideoPlay_test();

/**
*	@brief video_analytics.h BackGraundAnalytics::VideoaNalyse 模块测试
*/
void BackGraundAnalytics_VideoaNalyse_test();

/**
*	@brief video_analytics.h OpticalFlowAnalytics::VideoaNalyse 模块测试
*/
void OpticalFlowAnalytics_VideoaNalyse_test();