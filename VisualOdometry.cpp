
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




//===================================================================
//					FEATURE MATCHING (SURF)
//===================================================================

void calculateTandR(String foldername, int idx) {
	// Load first four images
	Mat img1, img2, img3, img4;
	loadImgs(foldername, idx, img1, img2, img3, img4);
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
