#include"test.h"


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
	//VideoAnalytics video(VIDEO_FILE VIDEO_FILE_NAME);
	video.VideoPlay();
}

void BackGraundAnalytics_VideoaNalyse_test() {
	BackGraundAnalytics video(VIDEO_FILE VIDEO_FILE_NAME);
	//BackGraundAnalytics video(VideoCapture(VIDEO_FILE VIDEO_FILE_NAME));
	video.VideoNalyse();
}

void OpticalFlowAnalytics_VideoaNalyse_test() {
	//OpticalFlowAnalytics video(VideoCapture(VIDEO_FILE VIDEO_FILE_NAME));
	OpticalFlowAnalytics video(VIDEO_FILE VIDEO_FILE_NAME);
	//video.VideoNalyse();
	video.VideoPlay();

}

void HogAnalytics_VideoaNalyse_test() {
	HogAnalytics video(VIDEO_FILE VIDEO_FILE_NAME);
	video.VideoNalyse();
	//video.VideoPlay();
}

void ObjDetect_yolov8_VideoObjDetect_test() {
	ObjDetect_yolov8 obj(VIDEO_FILE VIDEO_FILE_NAME, MODEL_FILE MODEL_FILE_NAME, LABEL_FILE LABEL_FILE_NAME);
	//obj.VideoPlay();
	obj.VideoObjDetect();
}

void ObjDetect_yolov5_VideoObjDetect_test() {
	ObjDetect_yolov5 obj(VIDEO_FILE "Pexels_Videos_2670.mp4", MODEL_FILE "yolov5s.onnx", LABEL_FILE LABEL_FILE_NAME);
	//ObjDetect_yolov5 obj(VIDEO_FILE "vtest.avi", MODEL_FILE "yolov5s.onnx", LABEL_FILE LABEL_FILE_NAME);
	//obj.VideoPlay();
	obj.VideoObjDetect();
}



void LabelObj_ClassseGet_test() {
	//LabelObj file(LABEL_FILE LABEL_FILE_NAME);
	LabelObj file;
	file.FileRead(LABEL_FILE LABEL_FILE_NAME);
	vector<string> classes = file.ClassseGet();
	cout << classes[3] << endl;
}


void ImageAnalytics_ImagePlay_test()
{
	//ImageAnalytics img_obj(IMG_FILE IMG_FILE_NAME);
	//ImageAnalytics img_obj(imread(IMG_FILE IMG_FILE_NAME));
	GpuMat img(imread(IMG_FILE IMG_FILE_NAME));
	ImageAnalytics img_obj(img);
	//ImageAnalytics img_obj;
	//img_obj.ImageGet(IMG_FILE IMG_FILE_NAME);
	//img_obj.ImageGet(imread(IMG_FILE IMG_FILE_NAME));
	//GpuMat img(imread(IMG_FILE IMG_FILE_NAME));
	//img_obj.ImageGet(img);
	img_obj.ImagePlay();
}

void ImageFeatureMatch_ImagePlay_test() {
	ImageFeatureMatch img_obj(IMG_FILE IMG_FILE_NAME);
	img_obj.ImagePlay();
}

void ImageAnalytics_ImageNalyse_test() {
	ImageFeatureMatch img_obj(IMG_FILE "box.png");
	img_obj.ImageNalyse();
}

void ImageClassification_ImageNalyse_test() {
	ImageClassification img_obj(IMG_FILE "messi.jpg", MODEL_FILE "resnet18.onnx", LABEL_FILE "imagenet_classes.txt");
	img_obj.ImageNalyse();
}