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
		// ********************************* your code start *************************************
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

		// 检测特征子 提取描述符
		orb->detectAndCompute(d_object_image, GpuMat(), h_keypoints_object, d_descriptors_object);
		orb->detectAndCompute(d_scene_image, GpuMat(), h_keypoints_scene, d_descriptors_scene);
		

		// 暴力匹配
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

		// 绘制匹配点对
		
		drawMatches(object_image, h_keypoints_object, scene_image, h_keypoints_scene,
			good_matches, h_image_result, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::DEFAULT);

		// 查找与匹配点对对应的图像像素 2D 坐标
		std::vector<Point2f> object;
		std::vector<Point2f> scene;
		for (int i = 0; i < good_matches.size(); i++)
		{
			object.push_back(h_keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(h_keypoints_scene[good_matches[i].trainIdx].pt);
		}

		// 计算单应性矩阵
		Mat Homo = findHomography(object, scene, RANSAC);
		std::vector<Point2f> corners(4); // four corners of the image
		std::vector<Point2f> scene_corners(4);

		// 透视变换
		corners[0] = Point(0, 0);
		corners[1] = Point(object_image.cols, 0);
		corners[2] = Point(object_image.cols, object_image.rows);
		corners[3] = Point(0, object_image.rows);
		perspectiveTransform(corners, scene_corners, Homo);

		// ********************************* your code end ***************************************
		int64 end = getTickCount();
		double runtime = getTickFrequency() / (end - start);

		// 绘制对象
		line(h_image_result, scene_corners[0] + Point2f(object_image.cols, 0), scene_corners[1] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[1] + Point2f(object_image.cols, 0), scene_corners[2] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[2] + Point2f(object_image.cols, 0), scene_corners[3] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		line(h_image_result, scene_corners[3] + Point2f(object_image.cols, 0), scene_corners[0] + Point2f(object_image.cols, 0), Scalar(255, 0, 0), 4);
		putText(h_image_result, format("runtime: %.2f", 1000), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("Good Matches & Object detection", h_image_result);
		waitKey(0);

}
