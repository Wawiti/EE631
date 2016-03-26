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

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define DEBUG

int main() {
#ifdef DEBUG
	namedWindow("Matches", CV_WINDOW_AUTOSIZE);
#endif
	
	// CREATE A DETECTOR AND A MATCHER
	Ptr<SURF> detector = SURF::create(300);
	BFMatcher matcher = BFMatcher(NORM_L2, true);

	for (int i = 1; i < 18; i++) {
		// Read in images for processing
		Mat img1, img2;
		String path = "Images//T";
		imread(path + to_string(i) + ".jpg", IMREAD_GRAYSCALE).copyTo(img1);
		imread(path + to_string(i + 1) + ".jpg", IMREAD_GRAYSCALE).copyTo(img2);
		if (img1.empty() || img2.empty()) {
			cout << "images didn't read correctly, please try again" << endl;
			cout << "     crashed while reading: " << path + to_string(i) << endl;
			cout << "     and: " << path + to_string(i) << endl;
			system("pause");
			return -1;
		}

		vector<KeyPoint> keypoints1, keypoints2;
		vector<DMatch> matches;
		Mat descriptors1, descriptors2;

		// Find all keypoints and find matching points
		detector->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
		detector->detectAndCompute(img2, Mat(), keypoints2, descriptors2);
		matcher.match(descriptors1, descriptors2, matches);

		// Filter large hamming distances
		vector<char> matchesMask(matches.size(), 0);
		double maxDistance = 0;
		for (int idx = 0; idx < matches.size(); idx++) {
			if (matches[idx].distance > maxDistance)
				maxDistance = matches[idx].distance;
		}
		for (int idx = 0; idx < matches.size(); idx++) {
			if (matches[idx].distance < (maxDistance*0.1))
				matchesMask[idx] = 1;
		}

#ifdef DEBUG
		cout << "FOUND " << keypoints1.size() << " keypoints on the first image" << endl;
		cout << "FOUND " << keypoints2.size() << " keypoints on the second image" << endl;

		Mat img_matches;
		//drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
		drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, Scalar::all(-1), Scalar::all(-1), matchesMask, 2);

		while (1) {
			imshow("Matches", img_matches);
			int keypress = waitKey(30);
			if (keypress == 32) {
				break;
			}
		}
#endif
		
		// Return vectors of good points
		vector<Point2f> finalPoint1, finalPoint2;
		for (int idx = 0; idx < matches.size(); idx++) {
			if (matchesMask[idx]) {
				finalPoint1.push_back(keypoints1[matches[idx].queryIdx].pt);
				finalPoint2.push_back(keypoints2[matches[idx].trainIdx].pt);
			}
		}
	}
	
	return 0;
}
