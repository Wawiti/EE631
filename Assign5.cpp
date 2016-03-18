#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

//===================== Sub Functions ==========================

//===================== Rectify and undistort ===================
// Task 1 subfunction to rectify and undistort an uncalibrated system
// @Param1 number of images used to find difference
// @Param2 Path of file containing features
// @Param3 Filename of file containing features
Mat rectifyAndUndistortUncalibrated(int numIm, String Path, String Filename) {
	ifstream dataIn;
	vector<Point2f> featuresFirst;
	vector<Point2f> featuresLast;

	// Reading in points from previous assignment
	dataIn.open(Path + "\\" + Filename + "_points.txt");
	while (!dataIn.eof()) {
		float index, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
		dataIn >> index;
		dataIn >> x1; dataIn >> y1;
		dataIn >> x2; dataIn >> y2;
		dataIn >> x3; dataIn >> y3;
		dataIn >> x4; dataIn >> y4;
		dataIn >> x5; dataIn >> y5;
		dataIn >> x6; dataIn >> y6;
		featuresFirst.push_back(Point2f(x1, y1));
		featuresLast.push_back(Point2f(x6, y6));
	}
	// Calculate or guess parameters
	Mat F, H1, H2, R1, R2;
	Mat M, Dist;
	
	// Importing camera parameters
	FileStorage fs("CameraParam.yml", FileStorage::READ);
	fs["Camera Matrix"] >> M;
	fs["Distortion Coefficients"] >> Dist;
	vector<uchar> status;
	F = findFundamentalMat(featuresFirst, featuresLast, 8, 3.0, 0.989999999999911, status);

	// Removing bad points
	vector<Point2f> featuresFirstFiltered, featuresLastFiltered;
	for (int i = 0; i < status.size(); i++) {
		if (status[i]) {
			featuresFirstFiltered.push_back(featuresFirst[i]);
			featuresLastFiltered.push_back(featuresLast[i]);
		}
	}

	// Rectifying points
	stereoRectifyUncalibrated(featuresFirstFiltered, featuresLastFiltered, F, Size(640, 480), H1, H2);
	Mat temp = M.inv();
	R1 = M.inv()*H1*M;
	R2 = M.inv()*H2*M;

	// Loading images
	Mat Im1, Im2;
	Im1 = imread(Path + "\\" + "Task3\\" + Filename + "10.jpg", CV_LOAD_IMAGE_COLOR);
	Im2 = imread(Path + "\\" + "Task3\\" + Filename + "16.jpg", CV_LOAD_IMAGE_COLOR);

	// Undistort and Rectify
	Mat LM1, LM2, RM1, RM2;
	initUndistortRectifyMap(M, Dist, R1, M, Size(640, 480), CV_32FC1, LM1, LM2);
	initUndistortRectifyMap(M, Dist, R2, M, Size(640, 480), CV_32FC1, RM1, RM2);
	remap(Im1, Im1, LM1, LM2, INTER_LINEAR);
	remap(Im2, Im2, RM1, RM2, INTER_LINEAR);
	
	// Drawing lines on images to ensure horizontallity
	for (int i = 0; i < 480; i += 40) {
		line(Im1, Point2f(0, i), Point2f(639, i), Scalar(255, 0, 0));
		line(Im2, Point2f(0, i), Point2f(639, i), Scalar(255, 0, 0));
	}

	// Saving final rectified and undistorted images
	imwrite(Filename + "_First.jpg", Im1);
	imwrite(Filename + "_Last.jpg", Im2);

	return(F);
}


//===================== Camera Disparity R & T ==================
// Task 2 subfunction to calculate difference between cameras
// @Param1 number of images used to find difference
// @Param2 Path of file containing features
// @Param3 Filename of file containing features
void calculateRotAndTransUncalibrated(int numIm, String Path, String Filename) {
	ifstream dataIn;
	vector<Point2f> featuresFirst;
	vector<Point2f> featuresLast;

	// Reading in points from previous assignment
	dataIn.open(Path + "\\" + Filename + "_points.txt");
	while (!dataIn.eof()) {
		float index, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
		dataIn >> index;
		dataIn >> x1; dataIn >> y1;
		dataIn >> x2; dataIn >> y2;
		dataIn >> x3; dataIn >> y3;
		dataIn >> x4; dataIn >> y4;
		dataIn >> x5; dataIn >> y5;
		dataIn >> x6; dataIn >> y6;
		featuresFirst.push_back(Point2f(x1, y1));
		featuresLast.push_back(Point2f(x6, y6));
	}

	// Calculate or guess parameters
	Mat F;
	Mat M, Dist;

	// Importing camera parameters
	FileStorage fs("CameraParam.yml", FileStorage::READ);
	fs["Camera Matrix"] >> M;
	fs["Distortion Coefficients"] >> Dist;
	vector<uchar> status;
	F = findFundamentalMat(featuresFirst, featuresLast, 8, 3.0, 0.989999999999911, status);

	// Removing bad points
	vector<Point2f> featuresFirstFiltered, featuresLastFiltered;
	for (int i = 0; i < status.size(); i++) {
		if (status[i]) {
			featuresFirstFiltered.push_back(featuresFirst[i]);
			featuresLastFiltered.push_back(featuresLast[i]);
		}
	}

	undistortPoints(featuresFirstFiltered, featuresFirstFiltered, M, Dist);
	undistortPoints(featuresLastFiltered, featuresLastFiltered, M, Dist);

	Mat E, W, U, Vt, D, Enorm;
	E = M.t()*F*M;
	SVDecomp(E, W, U, Vt);
	double data[] = { 1, 0, 0, 0, 1, 0, 0, 0, 0 };
	D = Mat(3, 3, CV_64FC1, data);
	Enorm = U*D*Vt;

	Mat R, T;
	recoverPose(Enorm, featuresFirstFiltered, featuresLastFiltered, R, T);

	ofstream ofs;
	ofs.open(Filename + "Data.txt");
	ofs << "R = " << endl << R << endl << endl;
	ofs << "T = " << endl << T << endl << endl;
	ofs << "E = " << endl << E << endl << endl;
	ofs << "F = " << endl << F << endl << endl;
	ofs.close();
}


//===================== 3D points from 2D =======================
// Task 3 subfunction to find approximate 3D information from the system
// @Param1 number of images used to find difference
// @Param2 Path of file containing features
// @Param3 Filename of file containing features
void calculate3D(int numIm, String Path, String Filename) {
	ifstream dataIn;
	vector<Point2f> featuresFirst;
	vector<Point2f> featuresLast;

	// Reading in points from previous assignment
	dataIn.open(Path + "\\" + Filename + "_points.txt");
	while (!dataIn.eof()) {
		float index, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6;
		dataIn >> index;
		dataIn >> x1; dataIn >> y1;
		dataIn >> x2; dataIn >> y2;
		dataIn >> x3; dataIn >> y3;
		dataIn >> x4; dataIn >> y4;
		dataIn >> x5; dataIn >> y5;
		dataIn >> x6; dataIn >> y6;
		featuresFirst.push_back(Point2f(x1, y1));
		featuresLast.push_back(Point2f(x6, y6));
	}

	// Calculate or guess parameters
	Mat F;
	Mat M, Dist;

	// Importing camera parameters
	FileStorage fs("CameraParam.yml", FileStorage::READ);
	fs["Camera Matrix"] >> M;
	fs["Distortion Coefficients"] >> Dist;
	vector<uchar> status;
	F = findFundamentalMat(featuresFirst, featuresLast, 8, 3.0, 0.989999999999911, status);

	// Removing bad points
	vector<Point2f> featuresFirstFiltered, featuresLastFiltered;
	for (int i = 0; i < status.size(); i++) {
		if (status[i]) {
			featuresFirstFiltered.push_back(featuresFirst[i]);
			featuresLastFiltered.push_back(featuresLast[i]);
		}
	}

	undistortPoints(featuresFirstFiltered, featuresFirstFiltered, M, Dist);
	undistortPoints(featuresLastFiltered, featuresLastFiltered, M, Dist);

	Mat E, W, U, Vt, D, Enorm;
	E = M.t()*F*M;
	SVDecomp(E, W, U, Vt);
	double data[] = { 1, 0, 0, 0, 1, 0, 0, 0, 0 };
	D = Mat(3, 3, CV_64FC1, data);
	Enorm = U*D*Vt;

	Mat R, T;
	recoverPose(Enorm, featuresFirstFiltered, featuresLastFiltered, R, T);

	Mat R1, R2, P1, P2, Q;
	stereoRectify(M, Dist, M, Dist, Size(640, 480), R, 0.002*T, R1, R2, P1, P2, Q);

	// cast 2D to 3D
	vector<Point3f> FirstFeat, LastFeat;
	for (int i = 0; i < featuresFirstFiltered.size(); i++) {
		FirstFeat.push_back(Point3f(featuresFirstFiltered[i].x, featuresFirstFiltered[i].y, featuresFirstFiltered[i].x - featuresLastFiltered[i].x));
		LastFeat.push_back(Point3f(featuresLastFiltered[i].x, featuresLastFiltered[i].y, featuresFirstFiltered[i].x - featuresLastFiltered[i].x));
	}

	// Find 3D distance?
	vector<Point3f> F3D, L3D;
	perspectiveTransform(FirstFeat, F3D, Q);
	perspectiveTransform(LastFeat, L3D, Q);
}



//===================== Task 1 Function ========================
// Task 1 Rectify and Undistort images from an uncalibrated system
void task1() {
	String Path = "C:\\Users\\ecestudent\\Documents\\Visual Studio 2015\\Projects\\Assign4\\Assign4";
	cout << "     Finding features in ParallelCube images           " << '\r' << flush;
	rectifyAndUndistortUncalibrated(6, Path, "ParallelCube");
	
	cout << "     Finding features in ParallelReal images           " << '\r' << flush;
	rectifyAndUndistortUncalibrated(6, Path, "ParallelReal");
	
	cout << "     Finding features in TurnCube images               " << '\r' << flush;
	rectifyAndUndistortUncalibrated(6, Path, "TurnCube");
	
	cout << "     Finding features in TurnReal images               " << '\r' << flush;
	rectifyAndUndistortUncalibrated(6, Path, "TurnReal");
}


//===================== Task 2 Function ========================
// Task 2 Find Rotation and Translation from known intrinsic and 
// unknown extrinsic parameters
void task2() {
	String Path = "C:\\Users\\ecestudent\\Documents\\Visual Studio 2015\\Projects\\Assign4\\Assign4";
	cout << "     Finding R and T in ParallelCube images           " << '\r' << flush;
	calculateRotAndTransUncalibrated(6, Path, "ParallelCube");

	cout << "     Finding R and T in ParallelReal images           " << '\r' << flush;
	calculateRotAndTransUncalibrated(6, Path, "ParallelReal");
	
	cout << "     Finding R and T in TurnCube images               " << '\r' << flush;
	calculateRotAndTransUncalibrated(6, Path, "TurnCube");
	
	cout << "     Finding R and T in TurnReal images               " << '\r' << flush;
	calculateRotAndTransUncalibrated(6, Path, "TurnReal");
}


//===================== Task 3 Function ========================
// Task 3 Find approximate 3D location from known intrinsic and 
// extrinsic parameters
void task3() {
	String Path = "C:\\Users\\ecestudent\\Documents\\Visual Studio 2015\\Projects\\Assign4\\Assign4";
	cout << "     Finding R and T in ParallelCube images           " << '\r' << flush;
	calculate3D(6, Path, "ParallelCube");

	cout << "     Finding R and T in ParallelReal images           " << '\r' << flush;
	calculate3D(6, Path, "ParallelReal");

	cout << "     Finding R and T in TurnCube images               " << '\r' << flush;
	calculate3D(6, Path, "TurnCube");

	cout << "     Finding R and T in TurnReal images               " << '\r' << flush;
	calculate3D(6, Path, "TurnReal");
}



//======================== Main Loop ===========================
int main() {

	cout << "Starting Task 1: Unknown Intrinsic and Extrinsic Parameters" << endl;
	task1();

	cout << "Starting Task 2: Known Intrinsic and Unknown Extrinsic Parameter" << endl;
	task2();

	cout << "Starting Task 3: Known Intrinsic and Extrinsic Parameters" << endl;
	task3();


	//system("pause");
	return 0;
}
