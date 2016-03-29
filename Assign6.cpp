#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <sstream>
#include <string>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

//#define DEBUG


int findMatches(Mat img1, Mat img2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat descriptors1, Mat descriptors2, BFMatcher matcher, vector<Point2f>& finalPoint1, 
	vector<Point2f>& finalPoint2, double passRatio, vector<KeyPoint>& keypointsOut) {
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);
	vector<char> matchesMask(matches.size(), 0);

	// Find max distance
	double maxDistance = 0;
	for (int idx = 0; idx < matches.size(); idx++) {
		if (matches[idx].distance > maxDistance)
			maxDistance = matches[idx].distance;
	}

	// Cut out 1-passratio % or points
	for (int idx = 0; idx < matches.size(); idx++) {
		if (matches[idx].distance <= (maxDistance*passRatio))
			matchesMask[idx] = 1;
	}

#ifdef DEBUG
	namedWindow("Matches", CV_WINDOW_AUTOSIZE);
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), matchesMask, 2);

	while (1) {
		imshow("Matches", img_matches);
		int keypress = waitKey(30);
		if (keypress == 32) {
			break;
		}
	}
#endif

	// Output final points as well as a new vector of keypoints
	for (int idx = 0; idx < matches.size(); idx++) {
		if (matchesMask[idx]) {
			finalPoint1.push_back(keypoints1[matches[idx].queryIdx].pt);
			finalPoint2.push_back(keypoints2[matches[idx].trainIdx].pt);
			keypointsOut.push_back(keypoints2[matches[idx].trainIdx]);
		}
	}
	return 0;
}


int findExpansion(vector<Point2f>& finalPoint1, vector<Point2f>& finalPoint2) {
	double avg = 1+std::abs(1-(sqrt(pow(finalPoint2[0].x,2) + pow(finalPoint2[0].y,2)) / sqrt(pow(finalPoint1[0].x,2) + pow(finalPoint1[0].y,2))));
	for (int idx = 1; idx < finalPoint1.size(); idx++) {
		double a2 = 1+std::abs(1- sqrt(pow(finalPoint2[idx].x, 2) + pow(finalPoint2[idx].y, 2)) / sqrt(pow(finalPoint1[idx].x, 2) + pow(finalPoint1[idx].y, 2)));
		avg = (avg + a2) / 2;
	}
	//cout << avg << endl;
	return 0;
}


int findMinMaxX(vector<Point2f>& finalPoint1, vector<Point2f>& finalPoint2) {
	double minimum = 1000, maximum = 0;
	for (int idx = 0; idx < finalPoint1.size(); idx++) {
		if (finalPoint1[idx].x < minimum)
			minimum = finalPoint1[idx].x;
		else if (finalPoint1[idx].x > maximum)
			maximum = finalPoint1[idx].x;
	}
	double width = maximum - minimum;
	double distance = 825.09 * (59 / width);
	cout << distance << endl;
	return 0;
}


int drawExpansion(Mat img1, Mat img2, vector<Point2f>& finalPoint1, vector<Point2f>& finalPoint2) {
	Mat imOut;
	img1.copyTo(imOut);
	for (int idx = 0; idx < finalPoint1.size(); idx++) {
		arrowedLine(imOut, finalPoint1[idx], finalPoint2[idx], Scalar(0, 0, 255), 1);
	}
	namedWindow("flow", CV_WINDOW_AUTOSIZE);
	while (1) {
		imshow("flow", imOut);
		int keypress = waitKey(30);
		if (keypress == 32) {
			break;
		}
	}

	return 0;
}


int main() {
	int errorCode = 0;
	ofstream ofs;
	ofs.open("HW6Task1.txt");
	
	// CREATE A DETECTOR AND A MATCHER
	Ptr<SURF> detector = SURF::create(300);
	BFMatcher matcher = BFMatcher(NORM_L2, true);

	Mat img1, descriptors1;
	vector<KeyPoint> keypoints1;
	imread("Images//T" + to_string(1) + ".jpg", IMREAD_GRAYSCALE).copyTo(img1);
	vector<KeyPoint> keypointsTemp;
	detector->detect(img1, keypointsTemp, Mat());
	for (int idx = 0; idx < keypointsTemp.size(); idx++) {
		if (keypointsTemp[idx].pt.x >= 282 && keypointsTemp[idx].pt.x <= 370 && keypointsTemp[idx].pt.y >= 160 && keypointsTemp[idx].pt.y <= 360) {
			keypoints1.push_back(keypointsTemp[idx]);
		}
	}
	detector->compute(img1, keypoints1, descriptors1);

	for (int i = 2; i < 18; i++) {
		// Read in images for processing
		Mat img2;
		String path = "Images//T";
		imread(path + to_string(i) + ".jpg", IMREAD_GRAYSCALE).copyTo(img2);
		if (img2.empty()) {
			cout << "images didn't read correctly, please try again" << endl;
			cout << "     crashed while reading: " << path + to_string(i) << endl;
			cout << "     and: " << path + to_string(i) << endl;
			system("pause");
			return -1;
		}

		Mat descriptors2;
		vector<KeyPoint> keypoints2;
		detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

		vector<Point2f> finalPoint1, finalPoint2;
		vector<KeyPoint> keypointsOut;
		errorCode = findMatches(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, matcher, finalPoint1, finalPoint2, 1 - 1.8*(1 / ((double)i)), keypointsOut);
		errorCode = findExpansion(finalPoint1, finalPoint2);
		errorCode = findMinMaxX(finalPoint1, finalPoint2);
		//errorCode = drawExpansion(img1, img2, finalPoint1, finalPoint2);

		// Resetting variable for next loop
		keypoints1 = keypointsOut;
		detector->compute(img2, keypoints1, descriptors1);
		img2.copyTo(img1);
		

	}
	ofs.close();
	system("pause");
	return errorCode;	// 0 is fine, -1 means failure somewhere
}
