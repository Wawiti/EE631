
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
#define P_SEQ	// Practice sequence

double xPos = 0, yPos = 0;


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

		//cout << foldername + OrigPath << endl;

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

void matchBestPoints(vector<DMatch>& matches, vector<DMatch>& matches2,
	Mat& desc1, Mat& desc2, Mat& desc3, double pass, double pass2, 
	Mat img1, Mat img2, Mat img3, vector<KeyPoint>& key1, 
	vector<KeyPoint>& key2, vector<KeyPoint>& key3, vector<Point2f>& fp1, 
	vector<Point2f>& fp2, vector<Point2f>&fp3, vector<KeyPoint>& keyOut) {

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

	vector<KeyPoint> temp;
	for (int i = 0; i < matchesMask.size(); i++) {
		if (matchesMask[i]) {
			double distance;
			double xDist = key2[matches[i].trainIdx].pt.x-
				key1[matches[i].queryIdx].pt.x;
			double yDist = key2[matches[i].trainIdx].pt.y -
				key1[matches[i].queryIdx].pt.y;
			distance = sqrt(pow(xDist,2) + pow(yDist,2));
			if (distance < 100) {
				temp.push_back(key1[matches[i].queryIdx]);
				//fp2.push_back(key2[matches[i].trainIdx].pt);
				keyOut.push_back(key2[matches[i].trainIdx]);
			}
		}
	}

	// GIVEN MATCHES FROM 1 TO 2 FIND 2 TO 3
	Ptr<SURF> detector = SURF::create(300);
	Mat desc2_2;
	detector->compute(img2, keyOut, desc2_2);

	matcher.match(desc2_2, desc3, matches2);

	double maxDistance2 = 0;
	for (int i = 0; i < matches2.size(); i++) {
		if (matches2[i].distance > maxDistance2)
			maxDistance2 = matches2[i].distance;
	}
	// Make a mask for passing matches
	vector<char> matchesMask2(matches2.size(), 0);
	for (int i = 0; i < matches2.size(); i++) {
		if (matches2[i].distance < (maxDistance2*pass2))
			matchesMask2[i] = 1;
	}
	for (int i = 0; i < matchesMask2.size(); i++) {
		if (matchesMask2[i]) {
			double distance;
			double xDist = key3[matches2[i].trainIdx].pt.x -
				keyOut[matches2[i].queryIdx].pt.x;
			double yDist = key3[matches2[i].trainIdx].pt.y -
				keyOut[matches2[i].queryIdx].pt.y;
			distance = sqrt(pow(xDist, 2) + pow(yDist, 2));
			if (distance < 100) {
				fp1.push_back(temp[matches2[i].queryIdx].pt);
				fp2.push_back(keyOut[matches2[i].queryIdx].pt);
				fp3.push_back(key3[matches2[i].trainIdx].pt);
			}
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
			arrowedLine(img_matches, fp1[i], fp3[i], 
				Scalar(0, 0, 255), 2);
		}
		imshow("Matches", img_matches);
		waitKey(5);
#endif
}



//===================================================================
//			  CALCULATE TRANSLATION AND ROTATION
//===================================================================
void calculateTandR(String foldername, int idx, Mat& RPrev, 
	Mat& TPrev) {

	// Load first four images
	Mat img1, img2, img3, img4;
	loadImgs(foldername, idx, img1, img2, img3, img4);

	vector<KeyPoint> keypoints1, keypoints2, keypoints3;
	vector<KeyPoint> keyOut;
	Mat descriptors1, descriptors2, descriptors3;
	findkeypoints(img1, keypoints1, descriptors1);
	findkeypoints(img2, keypoints2, descriptors2);
	findkeypoints(img3, keypoints3, descriptors3);

	vector<DMatch> matches, matches2;
	vector<Point2f> finalPoint1, finalPoint2, finalPoint3;
	double pass = 1;
	double pass2 = 1;
	matchBestPoints(matches, matches2, descriptors1, descriptors2, 
		descriptors3, pass, pass2, img1, img2, img3, keypoints1, 
		keypoints2, keypoints3, finalPoint1, finalPoint2,
		finalPoint3, keyOut);

	// input camera matrix
	Mat M, F, E, W, U, Vt, D, Enorm, R, T;
#ifdef P_SEQ
	M = (Mat_<double>(3, 3) << 
		707.0912,   0,          601.8873, 
		0,          707.0912,   183.1104, 
		0,          0,          1);
#endif

	// filter points from fundamental matrix
	vector<uchar> status, status2;
	F = findFundamentalMat(finalPoint1, finalPoint3, 8, 3.0,
		0.989999999999911, status);
	E = findEssentialMat(finalPoint1, finalPoint3, M, RANSAC,
		0.998999999999911, 1.0, status2);
	vector<Point2f> featFilt1, featFilt2;
	for (int i = 0; i < status.size(); i++) {
		if (status[i] && status2[i]) {
			featFilt1.push_back(finalPoint1[i]);
			featFilt2.push_back(finalPoint3[i]);
		}
	}

	//E = M.t()*F*M;
	SVDecomp(E, W, U, Vt);
	D = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 0);
	Enorm = U*D*Vt;
	recoverPose(Enorm, featFilt1, featFilt2, R, T);
	T = 2.15*T;

	ofstream ofs;
	ofs.open("data.txt", std::ios_base::app);
	/*ofs <<  R.at<double>(Point(0, 0)) << " " <<
			R.at<double>(Point(1, 0)) << " " << 
			R.at<double>(Point(2, 0)) << " " <<
			T.at<double>(Point(0, 0)) << " " <<
			R.at<double>(Point(0, 1)) << " " <<
			R.at<double>(Point(1, 1)) << " " <<
			R.at<double>(Point(2, 1)) << " " <<
			T.at<double>(Point(0, 1)) << " " <<
			R.at<double>(Point(0, 2)) << " " <<
			R.at<double>(Point(1, 2)) << " " <<
			R.at<double>(Point(2, 2)) << " " <<
			T.at<double>(Point(0, 2)) << endl;*/
	xPos += T.at<double>(Point(0, 0));
	yPos += T.at<double>(Point(0, 2));
	ofs << xPos << ", " << yPos << endl;

	R.copyTo(RPrev);
	T.copyTo(TPrev);
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
	Mat RPrev = (Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);;
	Mat TPrev = (Mat_<double>(1, 3) << 0, 0, 0);;

	// Clear previous file
	ofstream ofs;
	ofs.open("data.txt");
	ofs << "";
	ofs.close();


	// Go through all pictures
	for (int idx = 0; idx < numIm; idx+=2) {
		// Load 4 images and find matching points through sequence
		//	of all images as well as 1-2, 2-3 and 3-4.
		cout << "Calculating image " << idx << endl;
		calculateTandR(foldername, idx, RPrev, TPrev);

		// calculate Rotation and Translation Vectors


		// output values in format: R11 R12 R13 Tx R21 R22 R23 Ty...
		//	R31 R32 R33 Tz
	}

	system("pause");
	return 0;
}
