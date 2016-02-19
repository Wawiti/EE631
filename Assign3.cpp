// Assignment 3 Stereo Calibration
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

// Function to create an object vector for calibration
vector<vector<Point3f>> createObjectVector(Size cornerMat, double dim, int numIm) {
	vector<vector<Point3f>> objectPoints;
	for (int v = 0; v < numIm; v++) {
		vector<Point3f> object;
		for (int i = 0; i < cornerMat.height; i++) {
			for (int k = 0; k < cornerMat.width; k++) {
				object.push_back(Point3f(dim*k, dim*i, 0));
			}
		}
		objectPoints.push_back(object);
	}return(objectPoints);
}

// Function to load a single image
Mat loadImage(String Filename, int index, String extension) {
	Mat Image, ImGray;
	String str = "CalibrationImages\\" + Filename + to_string(index) + extension;
	Image = imread(str, CV_LOAD_IMAGE_COLOR);
	if (!Image.data) {
			cout << "couldn't find Left or Right image " << index << endl;
			system("pause");
			Mat empty;
			return empty;
	}else {
		cvtColor(Image, ImGray, CV_RGB2GRAY);
		return ImGray;
	}
}

// Function to find corners in a single MAT image
vector<Point2f> findCornersSingleImage(Mat Image) {
	vector<Point2f> corners;
	bool patternfound = findChessboardCorners(Image, Size(10, 7), corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
	if (patternfound) {
		cornerSubPix(Image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	else {
		cout << "     could not detect chessboard in image:                " << '\r' << flush;
		waitKey(500);
	}return(corners);
}

// Function to find corners from lots of images
vector<vector<Point2f>> findCorners(int numIm, String Filename, String extension) {
	vector<vector<Point2f>> corners;

	for (int i = 0; i < numIm; i++) {
		Mat imGray;
		vector<Point2f> InnerCorner;
		String st = "CalibrationImages\\" + Filename + to_string(i) + extension;
		imGray = loadImage(Filename, i, extension);
		bool patternfound = findChessboardCorners(imGray, Size(10, 7), InnerCorner, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
		if (patternfound) {
			cornerSubPix(imGray, InnerCorner, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			corners.push_back(InnerCorner);
		}
		else {
			cout << "     could not detect chessboard in image:         " << i << '\r' << flush;
			waitKey(500);
		}
	}
	return corners;
}


//===================== Task 1 Function ========================
// Function to calibrate individual cameras
void task1(int numIm, String filename, String Name) {
	Mat empty;
	vector<Mat> imGray(numIm, empty);
	vector<vector<Point2f>> corners;
	vector<vector<Point3f>> objectPoints;
	FileStorage fs(filename, FileStorage::WRITE);

	cout << "     Creating object vector for camera calibration     " << '\r' << flush;
	objectPoints = createObjectVector(Size(10,7), 1, numIm);

	cout << "     Finding chessboard in images                      " << '\r' << flush;
	corners = findCorners(numIm, Name, ".bmp");

	cout << "     Calibrating " << Name << " Camera... please wait                 " << '\r' << flush;
	Mat cameraMatrix, distCoeffs, rvecs, tvecs;
	calibrateCamera(objectPoints, corners, Size(640, 480), cameraMatrix, distCoeffs, rvecs, tvecs);

	cout << "     Saving camera calibration to " << filename << '\r' << flush;
	fs << "cameraMatrix" << cameraMatrix;
	fs << "distCoeffs" << distCoeffs;
	fs.release();
	cout << "     Camera Parameters saved, continuing to task 2     " << '\r' << flush;
	cout << "                                                                         " << '\r' << flush;
}


//===================== Task 2 Function ========================
// Function to calibrate stereo system
void task2(String NameL, String NameR, int numIm, String filename, String Lfilename, String Rfilename, double dim) {
	Mat RCamMatrix, RDistCoeffs, LCamMatrix, LDistCoeffs;
	vector<vector<Point2f>> LeftCorners;
	vector<vector<Point2f>> RightCorners;
	vector<vector<Point3f>> objectPoints;
	
	FileStorage fs1(Lfilename, FileStorage::READ);
	FileStorage fs2(Rfilename, FileStorage::READ);
	FileStorage fs3(filename, FileStorage::WRITE);

	cout << "     Importing left and right camera parameters                    " << '\r' << flush;
	fs1["cameraMatrix"] >> LCamMatrix;
	fs1["distCoeffs"] >> LDistCoeffs;
	fs2["cameraMatrix"] >> RCamMatrix;
	fs2["distCoeffs"] >> RDistCoeffs;

	cout << "     Creating object vector for camera calibration     " << '\r' << flush;
	objectPoints = createObjectVector(Size(10, 7), dim, numIm);

	cout << "     Finding chessboard in Stereo images                      " << '\r' << flush;
	LeftCorners = findCorners(numIm, NameL, ".bmp");
	RightCorners = findCorners(numIm, NameR, ".bmp");

	cout << "     Calibrating Stereo System... please wait                 " << '\r' << flush;
	Mat R, T, E, F;
	stereoCalibrate(objectPoints, LeftCorners, RightCorners, LCamMatrix, LDistCoeffs, RCamMatrix, RDistCoeffs, Size(640, 480), R, T, E, F);

	fs3 << "Rotation" << R;
	fs3 << "Translation" << T;
	fs3 << "EssentialMatrix" << E;
	fs3 << "FundamentalMatrix" << F;

	fs1.release();
	fs2.release();
	fs3.release();
}


//===================== Task 3 Function ========================
// Function to calculate Epipolar Lines
void task3(String Lfilename, String Rfilename, String Sfilename) {
	Mat ImLeft, ImLeftUndis, ImRight, ImRightUndis, OutImL, OutImR;
	Mat LCamMat, RCamMat, LDisCoef, RDisCoef;
	Mat FMat;

	vector<Point2f> Lcorners, Rcorners;

	FileStorage fs1(Lfilename, FileStorage::READ);
	FileStorage fs2(Rfilename, FileStorage::READ);
	FileStorage fs3(Sfilename, FileStorage::READ);

	fs1["cameraMatrix"] >> LCamMat;
	fs1["distCoeffs"] >> LDisCoef;
	fs2["cameraMatrix"] >> RCamMat;
	fs2["distCoeffs"] >> RDisCoef;
	fs3["FundamentalMatrix"] >> FMat;

	fs1.release();
	fs2.release();
	fs3.release();

	ImLeft = loadImage("StereoL", 0, ".bmp");
	ImRight = loadImage("StereoR", 0, ".bmp");

	undistort(ImLeft, ImLeftUndis, LCamMat, LDisCoef);
	undistort(ImRight, ImRightUndis, RCamMat, RDisCoef);
	
	Lcorners = findCornersSingleImage(ImLeft);
	Rcorners = findCornersSingleImage(ImRight);

	cvtColor(ImLeftUndis, OutImL, CV_GRAY2RGB);
	cvtColor(ImRightUndis, OutImR, CV_GRAY2RGB);

	vector<Point2f> LPoints, RPoints;
	Mat LLines, RLines;
	LPoints.push_back(Lcorners[6]);
	LPoints.push_back(Lcorners[12]);
	LPoints.push_back(Lcorners[40]);
	RPoints.push_back(Rcorners[8]);
	RPoints.push_back(Rcorners[19]);
	RPoints.push_back(Rcorners[60]);

	circle(OutImL, LPoints[0], 4, Scalar(255, 0, 0), 2);
	circle(OutImL, LPoints[1], 4, Scalar(255, 0, 0), 2);
	circle(OutImL, LPoints[2], 4, Scalar(255, 0, 0), 2);
	circle(OutImL, Lcorners[8], 4, Scalar(0, 0, 255), 1);
	circle(OutImL, Lcorners[19], 4, Scalar(0, 0, 255), 1);
	circle(OutImL, Lcorners[60], 4, Scalar(0, 0, 255), 1);
	circle(OutImR, RPoints[0], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, RPoints[1], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, RPoints[2], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, Rcorners[6], 4, Scalar(0, 0, 255), 1);
	circle(OutImR, Rcorners[12], 4, Scalar(0, 0, 255), 1);
	circle(OutImR, Rcorners[40], 4, Scalar(0, 0, 255), 1);

	computeCorrespondEpilines(LPoints, 1, FMat, LLines);
	computeCorrespondEpilines(RPoints, 2, FMat, RLines);

	for (int i = 0; i < 3; i++) {
		line(OutImR, Point2f(0, (-LLines.at<Vec3f>(i, 0)[2] / LLines.at<Vec3f>(i, 0)[1])), 
			Point2f(639, (LLines.at<Vec3f>(i, 0)[0] * 639 + LLines.at<Vec3f>(i, 0)[2]) / (-1*LLines.at<Vec3f>(i, 0)[1])), 
			Scalar(255, 0, 0));

		line(OutImL, Point2f(0, (-RLines.at<Vec3f>(i, 0)[2] / RLines.at<Vec3f>(i, 0)[1])), 
			Point2f(639, (RLines.at<Vec3f>(i, 0)[0] * 639 +	RLines.at<Vec3f>(i, 0)[2]) / (-1*RLines.at<Vec3f>(i, 0)[1])), 
			Scalar(255, 0, 0));
	}

	imwrite("Task3Left.bmp", OutImL);
	imwrite("Task3Right.bmp", OutImR);
}


//===================== Task 4 Function ========================
// Function to undistort and rectify images
void task4(String Lfilename, String Rfilename, String Sfilename) {
	Mat ImLeft, ImLeftUndis, ImRight, ImRightUndis, OutImL, OutImR;
	Mat LCamMat, RCamMat, LDisCoef, RDisCoef;
	Mat R, T, E, F;

	vector<Point2f> Lcorners, Rcorners;

	FileStorage fs1(Lfilename, FileStorage::READ);
	FileStorage fs2(Rfilename, FileStorage::READ);
	FileStorage fs3(Sfilename, FileStorage::READ);

	fs1["cameraMatrix"] >> LCamMat;
	fs1["distCoeffs"] >> LDisCoef;
	fs2["cameraMatrix"] >> RCamMat;
	fs2["distCoeffs"] >> RDisCoef;
	fs3["Rotation"] >> R;
	fs3["Translation"] >> T;
	fs3["EssentialMatrix"] >> E;
	fs3["FundamentalMatrix"] >> F;

	fs1.release();
	fs2.release();
	fs3.release();

	ImLeft = imread("CalibrationImages\\StereoL1.bmp", CV_LOAD_IMAGE_COLOR);
	ImRight = imread("CalibrationImages\\StereoR1.bmp", CV_LOAD_IMAGE_COLOR);

	Mat R1, R2, P1, P2, Q;
	stereoRectify(LCamMat, LDisCoef, RCamMat, RDisCoef, Size(640, 480), R, T, R1, R2, P1, P2, Q);

	Mat LMap1, LMap2, RMap1, RMap2, ImLeft2, ImRight2;
	initUndistortRectifyMap(LCamMat, LDisCoef, R1, P1, Size(640, 480), CV_32FC1, LMap1, LMap2);
	initUndistortRectifyMap(RCamMat, RDisCoef, R2, P2, Size(640, 480), CV_32FC1, RMap1, RMap2);

	remap(ImLeft, OutImL, LMap1, LMap2, INTER_LINEAR);
	remap(ImRight, OutImR, RMap1, RMap2, INTER_LINEAR);

	for (int i = 0; i < 480; i += 40) {
		line(OutImL, Point2f(0, i), Point2f(639, i), Scalar(255,0,0));
		line(OutImR, Point2f(0, i), Point2f(639, i), Scalar(255,0,0));
	}

	imwrite("Task4Left.bmp", OutImL);
	imwrite("Task4Right.bmp", OutImR);

	FileStorage fs("4- StereoRectification.yaml", FileStorage::WRITE);
	fs << "R1" << R1;
	fs << "R2" << R2;
	fs << "P1" << P1;
	fs << "P2" << P2;
	fs << "Q" << Q;

	Mat Difference;
	absdiff(OutImL, OutImR, Difference);

	imwrite("Task4Diff.bmp", Difference);
}


//===================== Task 5 Function ========================
// Function to undistort and rectify images
void task5(String Lfilename, String Rfilename, String Rectfilename) {
	Mat ImLeft, ImLeftUndis, ImRight, ImRightUndis, OutImL, OutImR;
	Mat LCamMat, RCamMat, LDisCoef, RDisCoef;
	Mat R1, R2, P1, P2, Q;

	vector<Point2f> Lcorners, Rcorners;

	FileStorage fs1(Lfilename, FileStorage::READ);
	FileStorage fs2(Rfilename, FileStorage::READ);
	FileStorage fs3(Rectfilename, FileStorage::READ);

	fs1["cameraMatrix"] >> LCamMat;
	fs1["distCoeffs"] >> LDisCoef;
	fs2["cameraMatrix"] >> RCamMat;
	fs2["distCoeffs"] >> RDisCoef;
	fs3["R1"] >> R1;
	fs3["R2"] >> R2;
	fs3["P1"] >> P1;
	fs3["P1"] >> P2;
	fs3["Q"] >> Q;

	fs1.release();
	fs2.release();
	fs3.release();

	ImLeft = imread("CalibrationImages\\StereoL2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	ImRight = imread("CalibrationImages\\StereoR2.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Lcorners = findCornersSingleImage(ImLeft);
	Rcorners = findCornersSingleImage(ImRight);

	cvtColor(ImLeft, OutImL, CV_GRAY2RGB);
	cvtColor(ImRight, OutImR, CV_GRAY2RGB);
	
	vector<Point2f> LPoints, RPoints, LPointsOut, RPointsOut;
	LPoints.push_back(Lcorners[6]);
	LPoints.push_back(Lcorners[12]);
	LPoints.push_back(Lcorners[21]);
	LPoints.push_back(Lcorners[40]);

	RPoints.push_back(Rcorners[6]);
	RPoints.push_back(Rcorners[12]);
	RPoints.push_back(Rcorners[21]);
	RPoints.push_back(Rcorners[40]);

	circle(OutImL, LPoints[0], 4, Scalar(255, 0, 0), 2);
	circle(OutImL, LPoints[1], 4, Scalar(255, 0, 0), 2);
	circle(OutImL, LPoints[2], 4, Scalar(255, 0, 0), 2);
	circle(OutImL, LPoints[3], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, RPoints[0], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, RPoints[1], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, RPoints[2], 4, Scalar(255, 0, 0), 2);
	circle(OutImR, RPoints[3], 4, Scalar(255, 0, 0), 2);

	undistortPoints(LPoints, LPointsOut, LCamMat, LDisCoef, R1, P1);
	undistortPoints(RPoints, RPointsOut, RCamMat, RDisCoef, R2, P2);

	vector<Point3f> Left3D, Right3D;
	for (int i = 0; i < 4; i++) {
		Left3D.push_back(Point3f(LPointsOut[i].x, LPointsOut[i].y, LPointsOut[i].x - RPointsOut[i].x));
		Right3D.push_back(Point3f(RPointsOut[i].x, RPointsOut[i].y, LPointsOut[i].x - RPointsOut[i].x));
	}

	Mat LeftCoordinate, RightCoordinate;
	perspectiveTransform(Left3D, LeftCoordinate, Q);
	perspectiveTransform(Right3D, RightCoordinate, Q);

	FileStorage fs("5- UndistortPointsOut.yaml", FileStorage::WRITE);

	fs << "LeftCoordinate" << LeftCoordinate;
	fs << "RightCoordinate" << RightCoordinate;

	cout << LeftCoordinate << endl << endl;
	cout << RightCoordinate << endl << endl;
}


//======================== Main Loop ===========================
int main() {
	// --------------- SET ALL VARIABLES ----------------------------
	int numImLR = 36, numImStereo = 30;
	double dim = 3.88;

	// --------------- PROCESS ALL IMAGES ---------------------------
	cout << "Starting Task 1: Calibrating Left and Right Cameras" << endl;
	task1(numImLR, "1- LeftCameraParameters.yaml", "Left");
	task1(numImLR, "1- RightCameraParameters.yaml", "Right");

	cout << "Starting Task 2: Calibrating Stereo System" << endl;
	task2("StereoL", "StereoR", numImStereo, "2- StereoCameraParameters.yaml", "1- LeftCameraParameters.yaml", "1- RightCameraParameters.yaml", dim);

	cout << "Starting Task 3: Drawing Epipolar Lines" << endl;
	task3("1- LeftCameraParameters.yaml", "1- RightCameraParameters.yaml", "2- StereoCameraParameters.yaml");

	cout << "Starting Task 4: Rectifying Images" << endl;
	task4("1- LeftCameraParameters.yaml", "1- RightCameraParameters.yaml", "2- StereoCameraParameters.yaml");

	cout << "Starting Task 5: Calculating 3D Information" << endl;
	task5("1- LeftCameraParameters.yaml", "1- RightCameraParameters.yaml", "4- StereoRectification.yaml");

	system("pause");
	return 0;
}
