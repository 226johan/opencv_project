#include"video_analytics.h"

void VideoAnalytics::VideoPlay()
{
	string windowName = "video";
	bool ifPuase = false;
	bool ifPlay = true;
	namedWindow(windowName);
	if (!video_.isOpened()) video_.open(video_name_);
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

	if (!video_.isOpened()) { video_.open(video_name_); }

	
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

	if (!video_.isOpened()) { video_.open(video_name_); }
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


void ObjDetect_yolov8::VideoObjDetect()
{

	bool ifPuase = false;
	bool ifPlay = true;

	namedWindow("yolov8 detect");

	// 颜色表
	std::vector<cv::Scalar> colors;
	colors.push_back(cv::Scalar(0, 255, 0));
	colors.push_back(cv::Scalar(0, 255, 255));
	colors.push_back(cv::Scalar(255, 255, 0));
	colors.push_back(cv::Scalar(255, 0, 0));
	colors.push_back(cv::Scalar(0, 0, 255));

	if (!video_.isOpened()) { video_.open(video_name_); }
	label_obj_.FileRead(LABEL_FILE LABEL_FILE_NAME);


	auto net = cv::dnn::readNetFromONNX(model_name_);
	if (use_cuda_) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	cv::Mat frame;

	while (ifPlay) {
		if (!ifPuase) {
			int64 start = getTickCount();
			bool ret = video_.read(frame);
			if (!ret) break;

			// 图像预处理
			int w = frame.cols;
			int h = frame.rows;
			int max_ = std::max(h, w);
			cv::Mat image = cv::Mat::zeros(cv::Size(max_, max_), CV_8UC3);
			cv::Rect roi(0, 0, w, h);
			frame.copyTo(image(roi));

			float xy_factor = image.cols / 640.0f;

			// NCHW
			cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

			// interface
			net.setInput(blob);
			cv::Mat preds = net.forward("output0");

			cv::Mat detectProb(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());
			cv::Mat det_output = detectProb.t();
			std::vector<cv::Rect> boxes;
			std::vector<int> classIds;
			std::vector<float> confidences;

			for (int i = 0; i < det_output.rows; i++) {
				cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
				cv::Point classIdPoint;
				double score;
				cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

				// 置信度 0～1之间
				if (score > 0.25)
				{
					float cx = det_output.at<float>(i, 0);
					float cy = det_output.at<float>(i, 1);
					float ow = det_output.at<float>(i, 2);
					float oh = det_output.at<float>(i, 3);

					int x = static_cast<int>((cx - 0.5 * ow) * xy_factor);
					int y = static_cast<int>((cy - 0.5 * oh) * xy_factor);
					int width = static_cast<int>(ow * xy_factor);
					int height = static_cast<int>(oh * xy_factor);

					cv::Rect box;
					box.x = x;
					box.y = y;
					box.width = width;
					box.height = height;

					boxes.push_back(box);
					classIds.push_back(classIdPoint.x);
					confidences.push_back(score);
				}
			}

			// NMS
			std::vector<int> indexes;
			cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.50, indexes);
			for (size_t i = 0; i < indexes.size(); i++) {
				int index = indexes[i];
				int idx = classIds[index];
				cv::rectangle(frame, boxes[index], colors[idx % 5], 2, 8);
				cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
					cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(255, 255, 255), -1);
				cv::putText(frame, label_obj_.classes_names_[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
			}

			int64 end = getTickCount();
			double fps = getTickFrequency() / (end - start);
			putText(frame, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);

			imshow("yolov8 detect", frame);
		
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