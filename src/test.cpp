#include"test.h"
#include"video_analytics.h"

void opencv_test() {
	namedWindow("frame", WINDOW_AUTOSIZE);
	cv::Mat image = cv::imread(IMG_FILE IMG_FILE_NAME);
	cv::imshow("frame", image);

	cv::waitKey(0);
	cv::destroyAllWindows();
	return;
	
}

void opencv_cuda_test() {
	namedWindow("dst", WINDOW_AUTOSIZE);
	Mat image_host = cv::imread(IMG_FILE IMG_FILE_NAME);

	GpuMat image;
	image.upload(image_host);
	
	Mat dst;
	image.download(dst);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();

	cuda::printCudaDeviceInfo(cuda::getDevice());
	int count = cuda::getCudaEnabledDeviceCount();
	cout << "number cudn: " << count << endl;

	return;
}

void VideoAnalytics_VideoPlay_test() {
	VideoAnalytics video(VideoCapture(VIDEO_FILE VIDEO_FILE_NAME));
	video.VideoPlay();
}

void BackGraundAnalytics_VideoaNalyse_test() {
	BackGraundAnalytics video(VideoCapture(VIDEO_FILE VIDEO_FILE_NAME));
	video.VideoaNalyse();
}

void OpticalFlowAnalytics_VideoaNalyse_test() {
	OpticalFlowAnalytics video(VideoCapture(VIDEO_FILE VIDEO_FILE_NAME));
	video.VideoaNalyse();
}