#include "stdafx.h"
#include "time.h"
#include "math.h"
#include "Hardware.h"
#include <fstream>
#include <sstream>

ImagingResources	CTCSys::IR;

CTCSys::CTCSys()
{
	EventEndProcess = TRUE;
	EventEndMove = TRUE;
	IR.Acquisition = TRUE;
	IR.UpdateImage = TRUE;
	IR.CaptureSequence = FALSE;
	IR.DisplaySequence = FALSE;
	IR.PlayDelay = 30;
	IR.CaptureDelay = 30;
	IR.FrameID = 0;
	IR.CatchBall = FALSE;
	OPENF("c:\\Projects\\RunTest.txt");
}

CTCSys::~CTCSys()
{
	CLOSEF;
}

void CTCSys::QSStartThread()
{
	EventEndProcess = FALSE;
	EventEndMove = FALSE;
	QSMoveEvent = CreateEvent(NULL, TRUE, FALSE, NULL);		// Create a manual-reset and initially nosignaled event handler to control move event
	ASSERT(QSMoveEvent != NULL);

	// Image Processing Thread
	QSProcessThreadHandle = CreateThread(NULL, 0L,
		(LPTHREAD_START_ROUTINE)QSProcessThreadFunc,
		this, NULL, (LPDWORD) &QSProcessThreadHandleID);
	ASSERT(QSProcessThreadHandle != NULL);
	SetThreadPriority(QSProcessThreadHandle, THREAD_PRIORITY_HIGHEST);

	QSMoveThreadHandle = CreateThread(NULL, 0L,
		(LPTHREAD_START_ROUTINE)QSMoveThreadFunc,
		this, NULL, (LPDWORD) &QSMoveThreadHandleID);
	ASSERT(QSMoveThreadHandle != NULL);
	SetThreadPriority(QSMoveThreadHandle, THREAD_PRIORITY_HIGHEST);
}

void CTCSys::QSStopThread()
{
	// Must close the move event first
	EventEndMove = TRUE;				// Set the falg to true first
	SetEvent(QSMoveEvent);				// must set event to complete the while loop so the flag can be checked
	do { 
		Sleep(100);
		// SetEvent(QSProcessEvent);
	} while(EventEndProcess == TRUE);
	CloseHandle(QSMoveThreadHandle);

	// need to make sure camera acquisiton has stopped
	EventEndProcess = TRUE;
	do { 
		Sleep(100);
		// SetEvent(QSProcessEvent);
	} while(EventEndProcess == TRUE);
	CloseHandle(QSProcessThreadHandle);
}

long QSMoveThreadFunc(CTCSys *QS)
{
	while (QS->EventEndMove == FALSE) {
		WaitForSingleObject(QS->QSMoveEvent, INFINITE);
		if (QS->EventEndMove == FALSE) QS->Move(QS->Move_X, QS->Move_Y);
		ResetEvent(QS->QSMoveEvent);
	}
	QS->EventEndMove = FALSE;
	return 0;
}

//======================== X approximation =====================
double approximateX(vector<Point3f> &Vec3DPt) {
	if (Vec3DPt.size() < 2) {
		cout << "Infinite number of lines... ERROR" << endl;
	}

	double sumX = 0, sumZ = 0, sumXZ = 0, sumZZ = 0;
	int nPoint = Vec3DPt.size();
	for (int r = 0; r < nPoint; r++) {
		sumX += Vec3DPt[r].x;
		sumZ += Vec3DPt[r].z;
		sumXZ += Vec3DPt[r].x * Vec3DPt[r].z;
		sumZZ += Vec3DPt[r].z * Vec3DPt[r].z;
	}

	double xMean = sumX / nPoint;
	double zMean = sumZ / nPoint;
	double denominator = sumZZ - sumZ * zMean;
	double slope = (sumXZ - sumZ * xMean) / denominator;
	double xInt = xMean - slope * zMean;

	return xInt;
}

//======================== Y approximation =====================
double approximateY(vector<Point3f> &Vec3DPt/*, double* coefficients*/) {
	int nPoints = Vec3DPt.size();
	Mat z, y;
	for (int r = 0; r < nPoints; r++) {
		z.push_back(Vec3DPt[r].z);
		y.push_back(Vec3DPt[r].y);
	}

	double yInt;
	int order = 2;
	Mat Z;
	Z = Mat::zeros(z.rows, order + 1, CV_32FC1);
	Mat copy;
	for (int r = 0; r <= order; r++)
	{
		copy = z.clone();
		pow(copy, r, copy);
		Mat M1 = Z.col(r);
		copy.col(0).copyTo(M1);
	}
	Mat Z_t, Z_inv;
	transpose(Z, Z_t);
	Mat temp = Z_t*Z;
	Mat temp2;
	invert(temp, temp2);
	Mat temp3 = temp2*Z_t;
	Mat W = temp3*y;

	//get final y
	cout << " W =" << endl << W << endl;

	return W.at<float>(0, 0);
}

//======================== Bound Movement ======================
void Bounding(double &x, double &y){
	if (x >= 9){
		x = 8.8;
	}
	else if (x <= -9){
		x = -8.8;
	}
	if (y >= 8){
		y = 7.8;
	}
	else if (y <= -8){
		y = -7.8;
	}
}

//======================== MY CODE =============================
long QSProcessThreadFunc(CTCSys *QS)
{
	int     i;
	int     BufID = 0;

	// --------------- Import data from file ------------------------
	Mat LCamMat, LDisCoef, RCamMat, RDisCoef, R1, R2, P1, P2, Q;
	FileStorage fs("AllParameters.yml", FileStorage::READ);
	fs["LcameraMatrix"] >> LCamMat;
	fs["LdistCoeffs"] >> LDisCoef;
	fs["RcameraMatrix"] >> RCamMat;
	fs["RdistCoeffs"] >> RDisCoef;
	fs["R1"] >> R1;
	fs["R2"] >> R2;
	fs["P1"] >> P1;
	fs["P2"] >> P2;
	fs["Q"] >> Q;
	fs.release();

	ofstream ofs;
	ofs.open("Test5.txt", ofstream::out);

	Mat LOrig, LOrigCrop;				// Get an initial image for background removal
	Mat ROrig, ROrigCrop;	
	Mat Im[2];							// storing images for processing

	bool calibrated = false;			// Have you stored your initial images?
	bool PositionReseting = false;
	bool foundfirst = false;			// Have you found the first image with both balls?

	vector<Point2f> LCent, RCent;		// Stores points for right and left undistortion

	//Rect myROIL(300, 10, 150, 250);		// x, y, width, height
	//Rect myROIR(210, 10, 150, 250);		// x, y, width, height
	Rect myROIL(300, 10, 140, 250);		// x, y, width, height THIS ONE WORKS BETTER
	Rect myROIR(210, 10, 140, 250);		// x, y, width, height
	Point2f LROICent = Point2f(370, 90);
	Point2f RROICent = Point2f(300, 90);
	Size ROISize = Size(100, 100);
	
	int count = 0;						// Number of frames with ball in both
	int resetcount = 0;					// number of frames to reset

	float xOffset = 9;			// THESE MAY NEED TO BE CHANGED TO FIT ACTUAL REAL LIFE !!!!!!!!!!!!!!!!!!!!!!!!!!!!
	float yOffset = 21;

	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 20;
	params.filterByArea = true;
	params.minArea = 60;
	params.maxArea = 7000;
	params.blobColor = 255;
	params.filterByConvexity = false;
	params.filterByCircularity = false;
	params.filterByInertia = false;
	

	char    str[32];
    long	FrameStamp;
    
    FrameStamp = 0;


#ifndef PTGREY			// If the cameras aren't attached			
	int imCnt = 1;		// count to choose next image
	int ImageSet = 7;	// Set of images to read in from
#endif

	while (QS->EventEndProcess == FALSE) {
#ifdef PTGREY		// Image Acquisition
		if (QS->IR.Acquisition == TRUE) {
			for(i=0; i < QS->IR.NumCameras; i++) {
				QS->IR.PGRError = QS->IR.pgrCamera[i]->RetrieveBuffer(&QS->IR.PtGBuf[i]);
				// Get frame timestamp if exact frame time is needed.  Divide FrameStamp by 32768 to get frame time stamp in mSec
                QS->IR.metaData[i] = QS->IR.PtGBuf[i].GetMetadata();
				FrameStamp = QS->IR.metaData[i].embeddedTimeStamp;               
				if(QS->IR.PGRError == PGRERROR_OK){
					QS->QSSysConvertToOpenCV(&QS->IR.AcqBuf[i], QS->IR.PtGBuf[i]);		// copy image data pointer to OpenCV Mat structure
				}
			}
			for(i=0; i < QS->IR.NumCameras; i++) {
				if (QS->IR.CaptureSequence) {
#ifdef PTG_COLOR
					mixChannels(&QS->IR.AcqBuf[i], 1, &QS->IR.SaveBuf[i][QS->IR.FrameID], 1, QS->IR.from_to, 3); // Swap B and R channels anc=d copy out the image at the same time.
#else
					QS->IR.AcqBuf[i].copyTo(QS->IR.SaveBuf[i][QS->IR.FrameID]);
#endif
				} else {
#ifdef PTG_COLOR
					mixChannels(&QS->IR.AcqBuf[i], 1, &QS->IR.ProcBuf[i][BufID], 1, QS->IR.from_to, 3); // Swap B and R channels anc=d copy out the image at the same time.
#else
					QS->IR.AcqBuf[i].copyTo(QS->IR.ProcBuf[i]);	// Has to be copied out of acquisition buffer before processing
#endif
				}
			}
		}
#else
		Sleep (100);
#endif
		// Process Image ProcBuf
		if (QS->IR.CatchBall) {  	// Click on "Catch" button to toggle the CatchBall flag when done catching
			// Images are acquired into ProcBuf[0] for left and ProcBuf[1] for right camera
			// Need to create child image or small region of interest for processing to exclude background and speed up processing
			// Mat child = QS->IR.ProcBuf[i](Rect(x, y, width, height));

			
#ifdef PTGREY		// Grab images from the camera if the camera is attached
			for (i = 0; i < QS->IR.NumCameras; i++) {
				QS->IR.ProcBuf[i].copyTo(Im[i]);		// Get a copy of the image for processing
				//QS->IR.OutBuf1[i] = QS->IR.ProcBuf[i];	// copy image to output (allow to see final image)
			}
#else				// Import images from file if the camera is not attached
				char LPathOrig[50];
				char RPathOrig[50];
				//snprintf(LPathOrig, 1024, "Images\\Set%01d\\Set%01dL%02d.bmp", ImageSet, ImageSet, imCnt);
				//snprintf(RPathOrig, 1024, "Images\\Set%01d\\Set%01dR%02d.bmp", ImageSet, ImageSet, imCnt);
				sprintf_s(LPathOrig, "Images\\Set%01d\\Set%01dL%02d.bmp", ImageSet, ImageSet, imCnt);
				sprintf_s(RPathOrig, "Images\\Set%01d\\Set%01dR%02d.bmp", ImageSet, ImageSet, imCnt);
				Im[0] = imread(LPathOrig, CV_LOAD_IMAGE_GRAYSCALE);
				Im[1] = imread(RPathOrig, CV_LOAD_IMAGE_GRAYSCALE);
				imCnt++;
				if (imCnt >= 90) {
					system("pause");	// Stop after 100 frames
				}
#endif
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Done grabbing images from camera

			if (!calibrated) {
				count++;
				if (count >= 5) {	// Don't grab the first frame, wait a bit for things to stabilize
					Im[0].copyTo(LOrig);
					Im[1].copyTo(ROrig);

					calibrated = true;
					count = 0;
				}
			}

			else {
				Mat LImCrop, LImThresh;
				Mat RImCrop, RImThresh;

				// Image Loaded, crop
				//MOGMaskL = QS->IR.ProcBuf[0];	// copy image to output (allow to see final image)
				//MOGMaskR = QS->IR.ProcBuf[1];
				Im[0].copyTo(LImCrop);
				Im[1].copyTo(RImCrop);
				getRectSubPix(LImCrop, ROISize, LROICent, LImCrop);
				getRectSubPix(RImCrop, ROISize, RROICent, RImCrop);
				getRectSubPix(LOrig, ROISize, LROICent, LOrigCrop);
				getRectSubPix(ROrig, ROISize, RROICent, ROrigCrop);

				// Remove background from image
				//LImThresh = LImCrop - LOrigCrop;	// Subtract very first image for background removal
				//RImThresh = RImCrop - ROrigCrop;
				absdiff(LImCrop, LOrigCrop, LImThresh);
				absdiff(RImCrop, ROrigCrop, RImThresh);

				// Blur image to reduce noise
				GaussianBlur(LImThresh, LImThresh, Size(9, 9), 2, 2);
				GaussianBlur(RImThresh, RImThresh, Size(9, 9), 2, 2);

				// Image Loaded, threshold image
				//threshold(LImThresh, LImThresh, 8, 255, THRESH_BINARY);
				//threshold(RImThresh, RImThresh, 8, 255, THRESH_BINARY);
				inRange(LImThresh, Scalar(8), Scalar(180), LImThresh);
				inRange(RImThresh, Scalar(8), Scalar(180), RImThresh);


				// Find the White pixels
				vector<KeyPoint> Lkeypoints, Rkeypoints;
				Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
				detector->detect(LImThresh, Lkeypoints);
				detector->detect(RImThresh, Rkeypoints);
#ifndef PTGREY
				Mat Lim_with_keypoints, Rim_with_keypoints;
				drawKeypoints(LImThresh, Lkeypoints, Lim_with_keypoints, Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				drawKeypoints(RImThresh, Rkeypoints, Rim_with_keypoints, Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
#endif

				if (Lkeypoints.size() >= 1){
					LROICent = Point2f(LROICent.x - (ROISize.width / 2) + Lkeypoints[0].pt.x, LROICent.y - (ROISize.height / 2) + Lkeypoints[0].pt.y);
					if (Lkeypoints[0].size >= (ROISize.width*0.4)){
						ROISize = Size(ROISize.width * 1.5, ROISize.height *1.5);
					}
					if ((LROICent.x + (ROISize.width / 2)) >= 640){
						LROICent.x = 640 - (ROISize.width / 2);
					}
					else if ((LROICent.x - (ROISize.width / 2)) <= 0){
						LROICent.x = ROISize.width / 2;
					}
					if ((LROICent.y + (ROISize.height / 2)) >= 480){
						LROICent.y = 480 - (ROISize.height / 2);
					}
					else if ((LROICent.y - (ROISize.height / 2)) <= 0){
						LROICent.y = ROISize.height / 2;
					}
				}
				if (Rkeypoints.size() >= 1){
					RROICent = Point2f(RROICent.x - (ROISize.width / 2) + Rkeypoints[0].pt.x, RROICent.y - (ROISize.height / 2) + Rkeypoints[0].pt.y);
					if ((RROICent.x + (ROISize.width / 2)) >= 640){
						RROICent.x = 640 - (ROISize.width / 2);
					}
					else if ((RROICent.x - (ROISize.width / 2)) <= 0){
						RROICent.x = ROISize.width / 2;
					}
					if ((RROICent.y + (ROISize.height / 2)) >= 480){
						RROICent.y = 480 - (ROISize.height / 2);
					}
					else if ((RROICent.y - (ROISize.height / 2)) <= 0){
						RROICent.y = ROISize.height / 2;
					}
				}

				cvtColor(Im[0], Im[0], CV_GRAY2BGR);
				cvtColor(Im[1], Im[1], CV_GRAY2BGR);
				//rectangle(Im[0], Rect(LROICent.x, LROICent.y, ROISize.width, ROISize.height), Scalar(0, 0, 255), 2);
				if (Lkeypoints.size() >= 1){
					Lim_with_keypoints.copyTo(Im[0](Rect(LROICent.x - (ROISize.width / 2), LROICent.y - (ROISize.height / 2), Lim_with_keypoints.cols, Lim_with_keypoints.rows)));
				}
				//rectangle(Im[1], Rect(RROICent.x, RROICent.y, ROISize.width, ROISize.height), Scalar(0, 0, 255), 2);
				if (Rkeypoints.size() >= 1){
					Rim_with_keypoints.copyTo(Im[1](Rect(RROICent.x - (ROISize.width / 2), RROICent.y - (ROISize.height / 2), Rim_with_keypoints.cols, Rim_with_keypoints.rows)));
				}
				Mat LWhiteLocations, RWhiteLocations;
				findNonZero(LImThresh, LWhiteLocations);
				findNonZero(RImThresh, RWhiteLocations);

				if (PositionReseting) {
					resetcount++;
					if (resetcount >= 150) {
						double xPos = 0;
						double yPos = 0;
						Bounding(xPos, yPos);
#ifdef USE_STAGE
						QS->Move_X = xPos;
						QS->Move_Y = yPos;
						SetEvent(QS->QSMoveEvent);		// Signal the move event to move catcher. The event will be reset in the move thread.
#endif
						resetcount = 0;
						count = 0;
						PositionReseting = false;
						foundfirst = false;
						calibrated = false;

						// Clear center points
						vector<Point2f> empt1, empt2;		// Empty points to replace center points
						LCent.swap(empt1);
						RCent.swap(empt2);
#ifndef PTGREY
						// Reinitialize image counter
						imCnt = 0;
#endif
					}
				}

				if (((LWhiteLocations.total() > 50) && (RWhiteLocations.total() > 50)) && !PositionReseting) {		// Only come here if both balls are available
					count++;
					foundfirst = true;

					// Find center of ball in Left image
					float LxCent = 0, LyCent = 0;
					for (int r = 0; r < LWhiteLocations.total(); r++) {
						LxCent = LxCent + LWhiteLocations.at<Point>(r).x;
						LyCent = LyCent + LWhiteLocations.at<Point>(r).y;
					}
					LxCent = LxCent / LWhiteLocations.total();
					LyCent = LyCent / LWhiteLocations.total();

					// Find center of ball in Right image
					float RxCent = 0, RyCent = 0;
					for (int r = 0; r < RWhiteLocations.total(); r++) {
						RxCent = RxCent + RWhiteLocations.at<Point>(r).x;
						RyCent = RyCent + RWhiteLocations.at<Point>(r).y;
					}
					RxCent = RxCent / RWhiteLocations.total();
					RyCent = RyCent / RWhiteLocations.total();

					// Undistort the points
					LCent.push_back(Point2f(LxCent + myROIL.x, LyCent + myROIL.y));
					RCent.push_back(Point2f(RxCent + myROIR.x, RyCent + myROIR.y));

					// DEBUG
					//ofs << "LCamMat = " << endl << LCamMat << endl << endl;
					//ofs << "RCamMat = " << endl << RCamMat << endl << endl;
					//ofs << "LDisCoef = " << endl << LDisCoef << endl << endl;
					//ofs << "RDisCoef = " << endl << RDisCoef << endl << endl;
					//ofs << "LCent = " << endl << LCent << endl << endl;
					//ofs << "RCent = " << endl << RCent << endl << endl;

					//ofs << "R1" << endl << R1 << endl << "R2" << endl << R2 << endl;
					//ofs << "P1" << endl << P1 << endl << "P2" << endl << P2 << endl;

					//imwrite("LeftOrig.jpg", Im[0]);
					//imwrite("RightOrig.jpg", Im[1]);

					//imwrite("Left.jpg", LImThresh);
					//imwrite("Right.jpg", RImThresh);

					// If we've reached a point to calculate 3D info
					if (count == 4 || count == 12){
						// Rectify and undistort points
						vector<Point2f> LCentUndist, RCentUndist;
						undistortPoints(LCent, LCentUndist, LCamMat, LDisCoef, R1, P1);
						undistortPoints(RCent, RCentUndist, RCamMat, RDisCoef, R2, P2);

						// Create vectors of point3f(x, y, disparity)
						vector<Point3f> Lvec, Rvec;
						vector<Point3f> LPoints, RPoints;	// Stores points for 3D location
						for (int r = 0; r < LCentUndist.size(); r++) {
							Lvec.push_back(Point3f(LCentUndist[r].x, LCentUndist[r].y, LCentUndist[r].x - RCentUndist[r].x));
							Rvec.push_back(Point3f(RCentUndist[r].x, RCentUndist[r].y, LCentUndist[r].x - RCentUndist[r].x));
							
							// Calcuate 3D info
							vector<Point3f> L3D, R3D;
							perspectiveTransform(Lvec, L3D, Q);
							perspectiveTransform(Rvec, R3D, Q);
							LPoints.push_back(L3D[r]);
							RPoints.push_back(R3D[r]);

							// DEBUG
							ofs << i << ", " << L3D[r].x << ", " << L3D[r].y << ", " << L3D[r].z << ", " << R3D[r].x << ", " << R3D[r].y << ", " << R3D[r].z << endl;
						}

						// If we've reached four good frames than move the x axis
						if (count == 4) {
							double xPos = approximateX(LPoints);
							xPos = (-xPos) + xOffset;
							double yPos = 0;
							Bounding(xPos, yPos);
#ifdef USE_STAGE
							QS->Move_X = xPos;
							QS->Move_Y = yPos;
							SetEvent(QS->QSMoveEvent);		// Signal the move event to move catcher. The event will be reset in the move thread.
							xPos = 0.00;
							yPos = 0.00;
#endif
						}

						// If we've reached 12 good frames than move the x and y axis
						else if (count == 12) {
							double xPos = approximateX(LPoints);
							double yPos = approximateY(LPoints);
							xPos = (-xPos) + xOffset;
							yPos = (-yPos) + yOffset;
							Bounding(xPos, yPos);
#ifdef USE_STAGE
							QS->Move_X = xPos;
							QS->Move_Y = yPos;
							SetEvent(QS->QSMoveEvent);		// Signal the move event to move catcher. The event will be reset in the move thread.
							xPos = 0.00;
							yPos = 0.00;
#endif
							// If we're done calculating trajectory, initiate time delay
							PositionReseting = true;
						}
					}
				}

				// If the ball leaves the frame before reaching 12 good frames, calculate anyway
				else if(foundfirst && !PositionReseting){
					// Undistort and rectify the points
					vector<Point2f> LCentUndist, RCentUndist;
					vector<Point3f> LPoints, RPoints;	// Stores points for 3D location
					undistortPoints(LCent, LCentUndist, LCamMat, LDisCoef, R1, P1);
					undistortPoints(RCent, RCentUndist, RCamMat, RDisCoef, R2, P2);

					// Create vectors of point3f(x, y, disparity)
					vector<Point3f> Lvec, Rvec;
					for (int r = 0; r < LCentUndist.size(); r++) {
						Lvec.push_back(Point3f(LCentUndist[r].x, LCentUndist[r].y, LCentUndist[r].x - RCentUndist[r].x));
						Rvec.push_back(Point3f(RCentUndist[r].x, RCentUndist[r].y, LCentUndist[r].x - RCentUndist[r].x));

						// Calcuate 3D info
						vector<Point3f> L3D, R3D;
						perspectiveTransform(Lvec, L3D, Q);
						perspectiveTransform(Rvec, R3D, Q);
						LPoints.push_back(L3D[r]);
						RPoints.push_back(R3D[r]);

						// DEBUG
						ofs << i << ", " << L3D[r].x << ", " << L3D[r].y << ", " << L3D[r].z << ", " << R3D[r].x << ", " << R3D[r].y << ", " << R3D[r].z << endl;
					}
					// Tell the stage to reset (initiate time delay)
					PositionReseting = true;

					// Move x and y axis
					double xPos = approximateX(LPoints);
					double yPos = approximateY(LPoints);
					xPos = (-xPos) + xOffset;
					yPos = (-(yPos-yOffset));
					Bounding(xPos, yPos);
#ifdef USE_STAGE
					QS->Move_X = xPos;
					QS->Move_Y = yPos;
					SetEvent(QS->QSMoveEvent);		// Signal the move event to move catcher. The event will be reset in the move thread.
					xPos = 0.00;
					yPos = 0.00;
#endif
				}
			}
			// End of our code
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		}
		// Display Image
		if (QS->IR.UpdateImage) {
			for (i=0; i<QS->IR.NumCameras; i++) {
				if (QS->IR.CaptureSequence || QS->IR.DisplaySequence) {
#ifdef PTG_COLOR
					QS->IR.SaveBuf[i][QS->IR.FrameID].copyTo(QS->IR.DispBuf[i]);
#else
					QS->IR.OutBuf[0] = QS->IR.OutBuf[1] = QS->IR.OutBuf[2] = QS->IR.SaveBuf[i][QS->IR.FrameID];
					merge(QS->IR.OutBuf, 3, QS->IR.DispBuf[i]);
#endif
					sprintf_s(str,"%d",QS->IR.FrameID);
					putText(QS->IR.DispBuf[0], str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2);
					if (QS->IR.PlayDelay) Sleep(QS->IR.PlayDelay);
				} else {
#ifdef PTG_COLOR
					QS->IR.ProcBuf[i][BufID].copyTo(QS->IR.DispBuf[i]);
#else
					// Display OutBuf1 when Catch Ball, otherwise display the input image
					QS->IR.OutBuf[0] = QS->IR.OutBuf[1] = QS->IR.OutBuf[2] = (QS->IR.CatchBall) ? QS->IR.OutBuf1[i] : QS->IR.ProcBuf[i];
					merge(QS->IR.OutBuf, 3, QS->IR.DispBuf[i]);
					// line(QS->IR.DispBuf[i], Point(0, 400), Point(640, 400), Scalar(0, 255, 0), 1, 8, 0);
#endif
				}
				QS->QSSysDisplayImage();
			}
		}
		if (QS->IR.CaptureSequence || QS->IR.DisplaySequence) {
			QS->IR.FrameID++;
			if (QS->IR.FrameID == MAX_BUFFER) {				// Sequence if filled
				QS->IR.CaptureSequence = FALSE;
				QS->IR.DisplaySequence = FALSE;
			} else {
				QS->IR.FrameID %= MAX_BUFFER;
			}
		}
		BufID = 1 - BufID;
	} 
	QS->EventEndProcess = FALSE;
	return 0;
}

void CTCSys::QSSysInit()
{
	long i, j;
	IR.DigSizeX = 640;
	IR.DigSizeY = 480;
	initBitmapStruct(IR.DigSizeX, IR.DigSizeY);

	// Camera Initialization
#ifdef PTGREY
	IR.cameraConfig.asyncBusSpeed = BUSSPEED_S800;
	IR.cameraConfig.isochBusSpeed = BUSSPEED_S800;
	IR.cameraConfig.grabMode = DROP_FRAMES;			// take the last one, block grabbing, same as flycaptureLockLatest
	IR.cameraConfig.grabTimeout = TIMEOUT_INFINITE;	// wait indefinitely
	IR.cameraConfig.numBuffers = 4;					// really does not matter since DROP_FRAMES is set not to accumulate buffers
	char ErrorMsg[64];

	// How many cameras are on the bus?
	if(IR.busMgr.GetNumOfCameras((unsigned int *)&IR.NumCameras) != PGRERROR_OK){	// something didn't work correctly - print error message
        sprintf_s(ErrorMsg, "Connect Failure: %s", IR.PGRError.GetDescription());
        AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP );
	} else {
		IR.NumCameras = (IR.NumCameras > MAX_CAMERA) ? MAX_CAMERA : IR.NumCameras;
		for(i = 0; i < IR.NumCameras; i++) {		
			// Get PGRGuid
			if (IR.busMgr.GetCameraFromIndex(i, &IR.prgGuid[i]) != PGRERROR_OK) {    // change to 1-i is cameras are swapped after powered up
				sprintf_s(ErrorMsg, "PGRGuID Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			IR.pgrCamera[i] = new Camera;
			if (IR.pgrCamera[i]->Connect(&IR.prgGuid[i]) != PGRERROR_OK) { 
				sprintf_s(ErrorMsg, "PConnect Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			// Set video mode and frame rate
			if (IR.pgrCamera[i]->SetVideoModeAndFrameRate(VIDEO_FORMAT, CAMERA_FPS) != PGRERROR_OK) { 
				sprintf_s(ErrorMsg, "Video Format Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			// Set all camera configuration parameters
			if (IR.pgrCamera[i]->SetConfiguration(&IR.cameraConfig) != PGRERROR_OK) { 
				sprintf_s(ErrorMsg, "Set Configuration Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			// Sets the onePush option off, Turns the control on/off on, disables auto control.  These are applied to all properties.
			IR.cameraProperty.onePush = false;
			IR.cameraProperty.autoManualMode = false;
			IR.cameraProperty.absControl = true;
			IR.cameraProperty.onOff = true;
			// Set shutter sppeed
			IR.cameraProperty.type = SHUTTER;
			IR.cameraProperty.absValue = SHUTTER_SPEED;
			if(IR.pgrCamera[i]->SetProperty(&IR.cameraProperty, false) != PGRERROR_OK){	
				sprintf_s(ErrorMsg, "Shutter Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			// Set gamma value
			IR.cameraProperty.type = GAMMA;
			IR.cameraProperty.absValue = 1.0;
			if(IR.pgrCamera[i]->SetProperty(&IR.cameraProperty, false) != PGRERROR_OK){	
				sprintf_s(ErrorMsg, "Gamma Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			// Set sharpness value
			IR.cameraProperty.type = SHARPNESS;
			IR.cameraProperty.absControl = false;
			IR.cameraProperty.valueA = 2000;
			if(IR.pgrCamera[i]->SetProperty(&IR.cameraProperty, false) != PGRERROR_OK){	
				sprintf_s(ErrorMsg, "Sharpness Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
#ifdef  PTG_COLOR
			// Set white balance (R and B values)
			IR.cameraProperty = WHITE_BALANCE;
			IR.cameraProperty.absControl = false;
			IR.cameraProperty.onOff = true;
			IR.cameraProperty.valueA = WHITE_BALANCE_R;
			IR.cameraProperty.valueB = WHITE_BALANCE_B;
			if(IR.pgrCamera[i]->SetProperty(&IR.cameraProperty, false) != PGRERROR_OK){	
				ErrorMsg.Format("White Balance Failure: %s",IR.PGRError.GetDescription());
				AfxMessageBox( ErrorMsg, MB_ICONSTOP );
			}
#endif
			// Set gain values (350 here gives 12.32dB, varies linearly)
			IR.cameraProperty = GAIN;
			IR.cameraProperty.absControl = false;
			IR.cameraProperty.onOff = true;
			IR.cameraProperty.valueA = GAIN_VALUE_A;
			IR.cameraProperty.valueB = GAIN_VALUE_B;
			if(IR.pgrCamera[i]->SetProperty(&IR.cameraProperty, false) != PGRERROR_OK){	
				sprintf_s(ErrorMsg, "Gain Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
			// Set trigger state
			IR.cameraTrigger.mode = 0;
			IR.cameraTrigger.onOff = TRIGGER_ON;
			IR.cameraTrigger.polarity = 0;
			IR.cameraTrigger.source = 0;
			IR.cameraTrigger.parameter = 0;
			if(IR.pgrCamera[i]->SetTriggerMode(&IR.cameraTrigger, false) != PGRERROR_OK){	
				sprintf_s(ErrorMsg, "Trigger Failure: %s", IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
            IR.embeddedInfo[i].frameCounter.onOff = true;
            IR.embeddedInfo[i].timestamp.onOff = true;
            IR.pgrCamera[i]->SetEmbeddedImageInfo(&IR.embeddedInfo[i]);
			// Start Capture Individually
			if (IR.pgrCamera[i]->StartCapture() != PGRERROR_OK) {
				sprintf_s(ErrorMsg, "Start Capture Camera %d Failure: %s", i, IR.PGRError.GetDescription());
				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP);
			}
		}
		// Start Sync Capture (only need to do it with one camera)
//		if (IR.pgrCamera[0]->StartSyncCapture(IR.NumCameras, (const Camera**)IR.pgrCamera, NULL, NULL) != PGRERROR_OK) {
//				sprintf_s(ErrorMsg, "Start Sync Capture Failure: %s", IR.PGRError.GetDescription());
//				AfxMessageBox(CA2W(ErrorMsg), MB_ICONSTOP );
//		}
	}

#else
	IR.NumCameras = MAX_CAMERA;
#endif
	Rect R = Rect(0, 0, 640, 480);
	// create openCV image
	for(i=0; i<IR.NumCameras; i++) {
#ifdef PTG_COLOR
		IR.AcqBuf[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC3);
		IR.DispBuf[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC3);
		IR.ProcBuf[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC3);
		for (j=0; j<MAX_BUFFER; j++) 
			IR.SaveBuf[i][j].create(IR.DigSizeY, IR.DigSizeX, CV_8UC3);
#else
		IR.AcqBuf[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC1);
		IR.DispBuf[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC1);
		IR.ProcBuf[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC1);
		for (j=0; j<MAX_BUFFER; j++) 
			IR.SaveBuf[i][j].create(IR.DigSizeY, IR.DigSizeX, CV_8UC1);
#endif
		IR.AcqPtr[i] = IR.AcqBuf[i].data;
		IR.DispROI[i] = IR.DispBuf[i](R); 
		IR.ProcROI[i] = IR.ProcBuf[i](R); 

		IR.OutBuf1[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC1);
		IR.OutBuf2[i].create(IR.DigSizeY, IR.DigSizeX, CV_8UC1);
		IR.OutROI1[i] = IR.OutBuf1[i](R); 
		IR.OutROI2[i] = IR.OutBuf2[i](R); 
		IR.DispBuf[i] = Scalar(0);
		IR.ProcBuf[i] = Scalar(0);
	}
	IR.from_to[0] = 0;
	IR.from_to[1] = 2;
	IR.from_to[2] = 1;
	IR.from_to[3] = 1;
	IR.from_to[4] = 2;
	IR.from_to[5] = 0;
	QSStartThread();
}

void CTCSys::QSSysFree()
{
	QSStopThread(); // Move to below PTGREY if on Windows Vista
#ifdef PTGREY
	for(int i=0; i<IR.NumCameras; i++) {
		if (IR.pgrCamera[i]) {
			IR.pgrCamera[i]->StopCapture();
			IR.pgrCamera[i]->Disconnect();
			delete IR.pgrCamera[i];
		}
	}
#endif
}

void CTCSys::initBitmapStruct(long iCols, long iRows)
{
	m_bitmapInfo.bmiHeader.biSize			= sizeof( BITMAPINFOHEADER );
	m_bitmapInfo.bmiHeader.biPlanes			= 1;
	m_bitmapInfo.bmiHeader.biCompression	= BI_RGB;
	m_bitmapInfo.bmiHeader.biXPelsPerMeter	= 120;
	m_bitmapInfo.bmiHeader.biYPelsPerMeter	= 120;
    m_bitmapInfo.bmiHeader.biClrUsed		= 0;
    m_bitmapInfo.bmiHeader.biClrImportant	= 0;
    m_bitmapInfo.bmiHeader.biWidth			= iCols;
    m_bitmapInfo.bmiHeader.biHeight			= -iRows;
    m_bitmapInfo.bmiHeader.biBitCount		= 24;
	m_bitmapInfo.bmiHeader.biSizeImage = 
      m_bitmapInfo.bmiHeader.biWidth * m_bitmapInfo.bmiHeader.biHeight * (m_bitmapInfo.bmiHeader.biBitCount / 8 );
}

void CTCSys::QSSysDisplayImage()
{
	for (int i = 0; i < 2; i++) {
		::SetDIBitsToDevice(
			ImageDC[i]->GetSafeHdc(), 1, 1,
			m_bitmapInfo.bmiHeader.biWidth,
			::abs(m_bitmapInfo.bmiHeader.biHeight),
			0, 0, 0,
			::abs(m_bitmapInfo.bmiHeader.biHeight),
			IR.DispBuf[i].data,
			&m_bitmapInfo, DIB_RGB_COLORS);
	}
}

#ifdef PTGREY
void CTCSys::QSSysConvertToOpenCV(Mat* openCV_image, Image PGR_image)
{
	openCV_image->data = PGR_image.GetData();	// Pointer to image data
	openCV_image->cols = PGR_image.GetCols();	// Image width in pixels
	openCV_image->rows = PGR_image.GetRows();	// Image height in pixels
	openCV_image->step = PGR_image.GetStride(); // Size of aligned image row in bytes
}
#endif
