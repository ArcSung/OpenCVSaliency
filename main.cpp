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
    // Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( Src.size(), CV_8UC1 );
    for (size_t i=0, max=boundRect.size(); i<max; ++i) {
  	    Scalar color = Scalar(255);
        drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy);
    }

    imshow("contours", drawing);
}    

int main(int argc, char* argv[])
{
	
	Mat img = imread(argv[1]);
    if(img.empty())
        return 0;
    Mat dst = Mat::zeros(img.size(), CV_8UC1);
    imshow("src", img);
	PreGraph SpMat;
	Mat superpixels = SpMat.GeneSp(img);
	Mat sal = SpMat.GeneSal(img);
	Mat salMap = SpMat.Sal2Img(superpixels, sal);
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(dst, CV_8U, 1.0);
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
    // Another option is to use dilate/erode/dilate:
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
