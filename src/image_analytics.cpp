#include"image_analytics.h"



const string ImageAnalytics::ImageGet(const string image_name){
	image_name_ = image_name;
	image_host_ = imread(image_name);
	image_.upload(image_host_);
	return image_name_; 
}

const Mat ImageAnalytics::ImageGet(const Mat& image_host) {
	image_host_ = image_host.clone();
	image_.upload(image_host_);
	return image_host_;
}

const GpuMat ImageAnalytics::ImageGet(const GpuMat& image) {
	image_ = image.clone();
	image_.download(image_host_);
	return image_;
}

void ImageAnalytics::ImagePlay() {
	namedWindow("img");
	bool isopen = true;
	while (isopen)
	{
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

}

void ImageFeatureMatch::ImageNalyse() {
		namedWindow("Good Matches & Object detection");
		Mat object_image = imread(IMG_FILE "box_in_scene.png");
		Mat scene_image = image_host_.clone();
		Mat h_image_result;
		int64 start = getTickCount();

		// image upload
		GpuMat d_scene_image = image_.clone();
		GpuMat d_object_image(object_image);
		
		cuda::cvtColor(d_scene_image, d_scene_image, COLOR_BGR2GRAY);
		cuda::cvtColor(d_object_image, d_object_image, COLOR_BGR2GRAY);

		// cpu key points
		vector<KeyPoint> h_keypoints_scene, h_keypoints_object;
		// gpu descriptor
		GpuMat d_descriptors_scene, d_descriptors_object;

		// obj detect
		auto orb = cuda::ORB::create();

		// ��������� ��ȡ������
		orb->detectAndCompute(d_object_image, GpuMat(), h_keypoints_object, d_descriptors_object);
		orb->detectAndCompute(d_scene_image, GpuMat(), h_keypoints_scene, d_descriptors_scene);
		

		// ����ƥ��
		Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
		vector<vector<DMatch>> d_matches;
		matcher->knnMatch(d_descriptors_object, d_descriptors_scene, d_matches, 2);

		std::cout << "match size:" << d_matches.size() << endl;
		std::vector<DMatch> good_matches;
		for (int k = 0; k < min(h_keypoints_object.size() - 1, d_matches.size()); k++)
		{
			if ((d_matches[k][0].distance < 0.9*(d_matches[k][1].distance)) &&
				((int)d_matches[k].size() <= 2 && (int)d_matches[k].size() > 0))
			{
				good_matches.push_back(d_matches[k][0]);
			}
		}
		std::cout << "size:" << good_matches.size() << endl;

		// ����ƥ����
		
		drawMatches(object_image, h_keypoints_object, scene_image, h_keypoints_scene,
			good_matches, h_image_result, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::DEFAULT);

		// ������ƥ���Զ�Ӧ��ͼ������ 2D ����
		std::vector<Point2f> object;
		std::vector<Point2f> scene;
		for (int i = 0; i < good_matches.size(); i++)
		{
			object.push_back(h_keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(h_keypoints_scene[good_matches[i].trainIdx].pt);
		}

		// ���㵥Ӧ�Ծ���
		Mat Homo = findHomography(object, scene, RANSAC);
		std::vector<Point2f> corners(4); // four corners of the image
		std::vector<Point2f> scene_corners(4);

		// ͸�ӱ任
		corners[0] = Point(0, 0);
		corners[1] = Point(object_image.cols, 0);
		corners[2] = Point(object_image.cols, object_image.rows);
		corners[3] = Point(0, object_image.rows);
		perspectiveTransform(corners, scene_corners, Homo);


		int64 end = getTickCount();
		double runtime = getTickFrequency() / (end - start);

		// ���ƶ���
		line(h_image_result, scene_corners[0] + Point2f(object_image.cols, 0), scene_corners[1] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[1] + Point2f(object_image.cols, 0), scene_corners[2] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[2] + Point2f(object_image.cols, 0), scene_corners[3] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[3] + Point2f(object_image.cols, 0), scene_corners[0] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		putText(h_image_result, format("runtime: %.2f", 1000), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("Good Matches & Object detection", h_image_result);
		waitKey(0);
		destroyAllWindows();
}



void ImageClassification::ImageNalyse() {
	namedWindow("img");
	cv::dnn::Net net = cv::dnn::readNetFromONNX(model_name_);

	if (use_cuda_) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	// pre-precess for image
	cv::Mat rgb, blob;
	cv::cvtColor(image_host_, rgb, cv::COLOR_BGR2RGB);
	cv::resize(rgb, blob, cv::Size(224, 224));
	blob.convertTo(blob, CV_32F);
	blob = blob / 255.0;
	cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
	cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);
	auto input_blob = cv::dnn::blobFromImage(blob);

	net.setInput(input_blob);
	cv::Mat prob = net.forward();

	cv::Mat probMat = prob.reshape(1, 1);
	cv::Point maxLoc;
	double score;
	cv::minMaxLoc(probMat, NULL, &score, NULL, &maxLoc);

	std::cout << "score: " << score << " class text : " << label_obj_.classes_names_[maxLoc.x] << std::endl;
	imshow("img", image_host_);
	waitKey(0);
	destroyAllWindows();
}
