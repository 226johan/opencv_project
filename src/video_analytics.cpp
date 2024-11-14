#include"video_analytics.h"

void VideoAnalytics::VideoPlay()
{
	string windowName = "video";
	bool ifPuase = false;
	bool ifPlay = true;
	namedWindow(windowName);
	while (ifPlay) {
		if (!ifPuase) {

			Mat frame;
			bool ret = video_.read(frame);
			if (!ret) return;
			imshow(windowName, frame);

		}

		char key = waitKey(1);
		switch (key)
		{
		case ' ':
			ifPuase = !ifPuase;
			break;
		case 27 :
			ifPlay = false;
			break;
		case 'q':
			ifPlay = false;
			break;
		}
	}
	cv::destroyWindow(windowName);
}



void BackGraundAnalytics::VideoaNalyse() {
	auto mog = cuda::createBackgroundSubtractorMOG2();
	Mat frame;
	// 图像 前景 背景
	GpuMat d_frame, d_fgmask, d_bgimg;

	Mat fg_mask, bgimg, fgimg;
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("background", WINDOW_AUTOSIZE);
	namedWindow("mask", WINDOW_AUTOSIZE);
	bool ifPuase = false;
	bool ifPlay = true;

	if (!video_.isOpened()) { cout << "video not open" << endl; return; }

	
	Mat se = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
	while (ifPlay) {
		if (!ifPuase) {
			int64 start = getTickCount();
			bool ret = video_.read(frame);
			if (!ret) break;

			d_frame.upload(frame);

			// 分析前景
			mog->apply(d_frame, d_fgmask);
			// 提取背景
			mog->getBackgroundImage(d_bgimg);

			// download from GPU Mat
			d_bgimg.download(bgimg);
			d_fgmask.download(fg_mask);

			int64 end = getTickCount();
			double fps = getTickFrequency() / (end - start);
			putText(frame, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
			imshow("input", frame);
			imshow("background", bgimg);
			imshow("mask", fg_mask);
		}

		char key = waitKey(1);
		switch (key)
		{
		case ' ':
			ifPuase = !ifPuase;
			break;
		case 27:
			ifPlay = false;
			break;
		case 'q':
			ifPlay = false;
			break;
		}
	}
}


void OpticalFlowAnalytics::VideoaNalyse() {
	auto farn = cuda::FarnebackOpticalFlow::create();
	// 当前帧 上一帧
	Mat f,pf;
	video_.read(pf);


	GpuMat frame, gray, preframe, pregray;
	preframe.upload(pf);
	cuda::cvtColor(preframe, pregray, COLOR_BGR2GRAY);
	Mat hsv = Mat::zeros(preframe.size(), preframe.type());

	GpuMat flow;
	vector<Mat> mv;
	split(hsv, mv);

	GpuMat gMag, gAng;
	Mat mag = Mat::zeros(hsv.size(), CV_32FC1);
	Mat ang = Mat::zeros(hsv.size(), CV_32FC1);

	gMag.download(mag);
	gAng.download(ang);

	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("optical flow", WINDOW_AUTOSIZE);
	bool ifPuase = false;
	bool ifPlay = true;


	if (!video_.isOpened()) { cout << "video not open" << endl; return; }


	Mat se = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
	while (ifPlay) {
		if (!ifPuase) {
			int64 start = getTickCount();
			bool ret = video_.read(f);
			if (!ret) break;

			// 光流分析
			frame.upload(f);
			cuda::cvtColor(frame, gray, COLOR_BGR2GRAY);
			farn->calc(pregray, gray, flow);

			// 坐标变换
			vector<GpuMat> mm;
			cuda::split(flow, mm);
			cuda::cartToPolar(mm[0], mm[1], gMag, gAng);
			cuda::normalize(gMag, gMag, 0, 255, NORM_MINMAX, CV_32FC1);
			gMag.download(mag);
			gAng.download(ang);

			// display
			ang = ang * 180 / CV_PI / 2.0;
			convertScaleAbs(mag, mag);
			convertScaleAbs(ang, ang);
			mv[0] = ang;
			mv[1] = Scalar(255);
			mv[2] = mag;
			merge(mv, hsv);
			Mat bgr;
			cv::cvtColor(hsv, bgr, COLOR_HSV2BGR);


			int64 end = getTickCount();
			double fps = getTickFrequency() / (end - start);
			putText(f, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
			gray.copyTo(pregray);
			imshow("input", f);
			imshow("optical flow", bgr);
		}

		char key = waitKey(1);
		switch (key)
		{
		case ' ':
			ifPuase = !ifPuase;
			break;
		case 27:
			ifPlay = false;
			break;
		case 'q':
			ifPlay = false;
			break;
		}
	}
}