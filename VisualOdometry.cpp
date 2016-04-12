
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


//#define DEBUG1
#define DEBUG2


//===================================================================
//						IMAGE LOADING....
//===================================================================

void loadImgs(String foldername, int idx, Mat& img1, Mat& img2, 
	Mat& img3, Mat& img4) {

	// Read in images
	vector<Mat> imgs;
	for (int i = 0; i < 4; i++) {
		char OrigPath[50];
		sprintf_s(OrigPath, "%06d.png", idx+i);

		cout << foldername + OrigPath << endl;

		imgs.push_back(imread(foldername + OrigPath, CV_LOAD_IMAGE_GRAYSCALE));
		if (imgs[i].empty()) {
			cout << "(ERROR) Images read incorrectly at: " << idx + i << endl;
		}
	}

	// Copy images to output variables
	imgs[0].copyTo(img1);
	imgs[1].copyTo(img2);
	imgs[2].copyTo(img3);
	imgs[3].copyTo(img4);
}



//===================================================================
//					FEATURE MATCHING (SURF)
//===================================================================
void findkeypoints(Mat img, vector<KeyPoint>& keypoints, 
	Mat& descriptors) {

	Ptr<SURF> detector = SURF::create(300);
	detector->detectAndCompute(img, Mat(), keypoints, descriptors);
}



//===================================================================
//					 MATCHING (BFMATCHER)
//===================================================================
void matchBestPoints(vector<DMatch>& matches, Mat desc1, Mat desc2, 
	double pass, Mat img1, Mat img2, vector<KeyPoint> key1, 
	vector<KeyPoint> key2, vector<Point2f> fp1, vector<Point2f> fp2) {

	BFMatcher matcher = BFMatcher(4, true);
	matcher.match(desc1, desc2, matches);

	// find max euclidean distance in matches
	double maxDistance = 0;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance > maxDistance)
			maxDistance = matches[i].distance;
	}

	// Make a mask for passing matches
	vector<char> matchesMask(matches.size(), 0);
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < (maxDistance*pass))
			matchesMask[i] = 1;
	}

	for (int idx = 0; idx < matchesMask.size(); idx++) {
		if (matchesMask[idx]) {
			fp1.push_back(key1[matches[idx].queryIdx].pt);
			fp2.push_back(key2[matches[idx].trainIdx].pt);
		}
	}

#ifdef DEBUG1
	namedWindow("Matches", CV_WINDOW_AUTOSIZE);
	Mat img_matches;
	drawMatches(img1, key1, img2, key2, matches, img_matches, 
		Scalar::all(-1), Scalar::all(-1), matchesMask, 2);

	//while (1) {
		imshow("Matches", img_matches);
		int keypress = waitKey(5);
		//if (keypress == 32) {
		//	break;
		//}
	//}
#endif
#ifdef DEBUG2
		namedWindow("Matches", CV_WINDOW_AUTOSIZE);
		Mat img_matches;
		img1.copyTo(img_matches);
		cvtColor(img_matches, img_matches, CV_GRAY2RGB);
		for (int i = 0; i < fp1.size(); i++) {
			arrowedLine(img_matches, fp1[i], fp2[i], 
				Scalar(0, 0, 255), 2);
		}
		imshow("Matches", img_matches);
		waitKey(5);
#endif
}



//===================================================================
//			  CALCULATE TRANSLATION AND ROTATION
//===================================================================

void calculateTandR(String foldername, int idx) {
	// Load first four images
	Mat img1, img2, img3, img4;
	loadImgs(foldername, idx, img1, img2, img3, img4);

	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	findkeypoints(img1, keypoints1, descriptors1);
	findkeypoints(img2, keypoints2, descriptors2);

	vector<DMatch> matches;
	vector<Point2f> finalPoint1, finalPoint2;
	double pass = 0.1;
	matchBestPoints(matches, descriptors1, descriptors2, pass, img1,
		img2, keypoints1, keypoints2, finalPoint1, finalPoint2);
}



//===================================================================
//					OUTPUTTING TO FILE (.TXT)
//===================================================================



//===================================================================
//                    MAIN FUNCTION CALLS
//===================================================================
int main() {
	// Global Parameters
	int numIm = 701-2;
	String foldername = "VO Practice Sequence\\";

	// Go through all pictures
	for (int idx = 0; idx < numIm; idx++) {
		// Load 4 images and find matching points through sequence
		//	of all images as well as 1-2, 2-3 and 3-4.
		calculateTandR(foldername, idx);

		// calculate Rotation and Translation Vectors


		// output values in format: R11 R12 R13 Tx R21 R22 R23 Ty...
		//	R31 R32 R33 Tz
	}

	system("pause");
	return 0;
}
