/*************************************************

Copyright: Guangyu Zhong all rights reserved

Author: Guangyu Zhong

Date:2014-09-27

Description: codes for Manifold Ranking Saliency Detection
Reference http://ice.dlut.edu.cn/lu/Project/CVPR13[yangchuan]/cvprsaliency.htm

**************************************************/
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include"PreGraph.h"
using namespace std;
using namespace cv;

Mat Gradient_Map(const Mat Src)
{
    Mat gray_img, dst;
    cvtColor(Src, gray_img, CV_RGB2GRAY);

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel(gray_img, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x); //convert to CV_8U
    Sobel(gray_img, grad_y, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y); //convert to CV_8U

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);

    return dst;

}    

void Contoures_Map(const Mat Src)
{
    // Find contours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(Src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Calculate convex hull based on contours
    // vector<vector<Point> > hull(contours.size());

    // Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f> center( contours.size() );
    vector<float> radius( contours.size() );

    for (size_t i = 0, max = contours.size(); i < max; ++i) {
        // Approximate polygon of a contour
  	    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        // Calculate the bounding box for the contour
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        // Calculate the bounding circle and store in center/radius
	    minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        // Calculate convex hull and store in hull
        // convexHull(Mat(contours[i]), hull[i], false);
    }

    // Find the biggest area of all contours
    int big_id = 0;
    double big_area = 0;
    for (size_t i = 0, max = contours.size(); i < max; ++i) {
  	    // Contour area
  	    double area = contourArea(contours[i]);
  	    if (area > big_area) {
  		    big_id = i;
  		    big_area = area;
  	    }
    }

    // Group all bounding rects into one, good for superimposition elimination
    // Vector<Rect> allRect = boundRect;
    // groupRectangles(boundRect, 0, INFINITY);
    // cout << boundRect.size() << endl;

    // Group bounding rects into one
    float xmin, xmax, ymin, ymax;
    xmax = 0;
    ymax = 0;
    xmin = INFINITY;
    ymin = INFINITY;
    if (boundRect.size() > 0) {
        for (size_t j=0, max = boundRect.size(); j<max; ++j) {
            int xminB = boundRect[j].x;
            int yminB = boundRect[j].y;
            int xmaxB = boundRect[j].x + boundRect[j].width;
            int ymaxB = boundRect[j].y + boundRect[j].height;
            if (xminB < xmin)
                xmin = xminB;
            if (yminB < ymin)
                ymin = yminB;
            if (xmaxB > xmax)
                xmax = xmaxB;
            if (ymaxB > ymax)
                ymax = ymaxB;
        }
    } else {
        xmin = 0;
        ymin = 0;
        xmax = 0;
        ymax = 0;
    }
    Rect bigRect = Rect(xmin, ymin, xmax-xmin, ymax-ymin);
//#ifdef DEBUG
    // Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( Src.size(), CV_8UC1 );
    for (size_t i=0, max=boundRect.size(); i<max; ++i) {
  	    //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  	    Scalar color = Scalar(255);
        //drawContours( drawing, contours_poly, i, color, 2, 8, vector<Vec4i>(), 0, Point() );
        drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy);
        //rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(0,200,0), 2, 8, 0 );
        //circle( drawing, center[i], (int)radius[i], Scalar(0,200,0), 2, 8, 0 );
        // Center point
        //circle( drawing, center[i], 3, Scalar(0,200,0), 2, 0, 0);
        // Contour points
        //for (size_t j=0, max = contours_poly[i].size(); j<max; ++j) {
    	//    circle( drawing, contours_poly[i][j], 3, Scalar(200,0,0), 2, 0, 0);
        //}
        // Convex hull points
        //drawContours(drawing, hull, i, color, 2, 8, hierarchy);
    }
    // Draw the big rectangle
    //rectangle( drawing, bigRect.tl(), bigRect.br(), Scalar(255,200,255), 2, 8, 0 );

//    #endif

    imshow("contours", drawing);
}    

int main(int argc, char* argv[])
{
	
	Mat img = imread(argv[1]);
    if(img.empty())
        return 0;
    //resize(img, img, Size(600, 400));
    Mat dst = Mat::zeros(img.size(), CV_8UC1);
    //Mat gradient = Gradient_Map(img);
    imshow("src", img);
	PreGraph SpMat;
	Mat superpixels = SpMat.GeneSp(img);
	Mat sal = SpMat.GeneSal(img);
	cout << sal;
	Mat salMap = SpMat.Sal2Img(superpixels, sal);
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(dst, CV_8U, 1.0);
	//gradient.convertTo(gradient, CV_8UC3, 1.0);
    //imshow("gradient", gradient);
    //printf("tmp.channels: %d\n", tmpsuperpixels.channels());
    //printf("gradient.channels: %d\n", gradient.channels());
    //waitKey(0);
    //multiply(tmpsuperpixels, gradient, dst, 0.04 , -1);
	//imshow("Saliency",  tmpsuperpixels);
	//imshow("tmp", dst);
    //Mat contoursImg = Contoures_Map(dst);
    //addWeighted(tmpsuperpixels, 0.8, gradient, 0.2, 0, dst);
    //imshow("dst result", dst);
    //imshow("dst contours", contoursImg);
    Mat blur, most_salient;
    bilateralFilter(dst, blur, 12, 24, 6);
    
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    minMaxLoc(blur, &minVal, &maxVal, &minLoc, &maxLoc);

    imshow("blur", blur);

    //threshold(blur, most_salient, 0, 255, THRESH_BINARY + THRESH_OTSU);
    inRange(blur, Scalar(maxVal - maxVal/5 ), Scalar(maxVal), most_salient);
    imshow("most_salient", most_salient);
	// Eliminate small regions (Mat() == default 3x3 kernel)
	Mat filtered;
    //filtered = most_salient;
    // Another option is to use dilate/erode/dilate:
	// dilate(most_salient, filtered, Mat(), Point(-1, -1), 2, 1, 1);
	// erode(filtered, filtered, Mat(), Point(-1, -1), 4, 1, 1);
	// dilate(filtered, filtered, Mat(), Point(-1, -1), 2, 1, 1);
	// sprintf(file_path, "%s_filtered.png", original_image_path);
	// imwrite(file_path, filtered);

	int morph_operator = 1; // 0: opening, 1: closing, 2: gradient, 3: top hat, 4: black hat
	int morph_elem = 2; // 0: rect, 1: cross, 2: ellipse
	int morph_size = 10; // 2*n + 1
    int operation = morph_operator + 2;

    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  // Apply the specified morphology operation
    morphologyEx( most_salient, filtered, operation, element );
    imshow("filtered", filtered);
    Contoures_Map(filtered);
	waitKey(0);
	return 0;
}
