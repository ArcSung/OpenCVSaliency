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

Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
Point origin;
Rect selection;

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

Mat JudgeColor(Mat imgOriginal, Mat Mask, Mat Mask2)
{
    Mat imgHSV;
    std::vector<Mat> vectorOfHSVImages;
    Mat imgValue;
    Mat HValue;
    Mat SValue;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    Mat hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    int hsize = 16;
    cvtColor(imgOriginal, imgHSV, CV_RGB2HSV);

    split(imgHSV, vectorOfHSVImages);

    HValue = vectorOfHSVImages[0];
    
    calcHist(&HValue, 1, 0, Mask, hist, 1, &hsize, &phranges);
    normalize(hist, hist, 0, 255, NORM_MINMAX);

    histimg = Scalar::all(0);
    int binW = histimg.cols / hsize;
    Mat buf(1, hsize, CV_8UC3);
    for( int i = 0; i < hsize; i++ )
        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
    cvtColor(buf, buf, COLOR_HSV2RGB);

    for( int i = 0; i < hsize; i++ )
    {
        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
        rectangle( histimg, Point(i*binW,histimg.rows),
        Point((i+1)*binW,histimg.rows - val),
        Scalar(buf.at<Vec3b>(i)), -1, 8 );
    }
    calcBackProject(&HValue, 1, 0, hist, backproj, &phranges, 1, true);
    //imshow("histimg", histimg);
    //imshow("backproj", backproj);

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    Mat backproj2, backproj3;
    minMaxLoc(backproj, &minVal, &maxVal, &minLoc, &maxLoc);
    //inRange(backproj, Scalar(maxVal - maxVal/5 ), Scalar(maxVal), backproj2);
    threshold(backproj, backproj2, 0, 255, THRESH_BINARY + THRESH_OTSU);
    //imshow("backproj2", backproj2);
	int morph_operator = 1; // 0: opening, 1: closing, 2: gradient, 3: top hat, 4: black hat
	int morph_elem = 2; // 0: rect, 1: cross, 2: ellipse
	int morph_size = 10; // 2*n + 1
    int operation = morph_operator + 2;

    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  // Apply the specified morphology operation
    morphologyEx( backproj2, backproj3, operation, element );
    backproj3 = backproj3 & Mask2;
    imshow("backproj3", backproj3);
    return backproj3;
}

void ConnectedComponentCompare(Mat img1, Mat img2)
{
    Mat labelImage(img1.size(), CV_32S);
    int nLabels = cv::connectedComponents(img2, labelImage, 8);
    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0);
    for(int label = 1; label < nLabels; ++label)
    {
        colors[label] = cv::Vec3b((rand()&255), (rand()&255), (rand()&255));
    }

    // ラベリング結果の描画
    cv::Mat dst(img1.size(), CV_8UC3);
    for(int y = 0; y < dst.rows; ++y)
    {
        for(int x = 0; x < dst.cols; ++x)
        {
            int label = labelImage.at<int>(y, x);
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);
            pixel = colors[label];
        }
    }

    Mat dst2(img1.size(), CV_8UC1);
    cvtColor(dst, dst2, CV_RGB2GRAY);

    //cv::namedWindow( "Connected Components", cv::WINDOW_AUTOSIZE );
    //cv::imshow( "Connected Components", dst );

    bitwise_and(dst2, img1, dst2);

    cv::namedWindow( "Connected Components2", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Connected Components2", dst2 );

}

static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, image.cols, image.rows);
        printf("selectObject\n");
    }

    switch( event )
    {
    case EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        printf("ButtonDown\n");
        break;
    case EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        printf("ButtonUP\n");
        break;
    }
}

int main(int argc, char* argv[])
{
	
	image = imread(argv[1]);
    if(image.empty())
        return 0;
    Mat proc_img = Mat::zeros(image.size(), CV_8UC3);
    image.copyTo(proc_img);
    namedWindow( "src", 0 );
    setMouseCallback( "src", onMouse, 0 );
    PreGraph SpMat;
	Mat superpixels = SpMat.GeneSp(proc_img);
	Mat sal = SpMat.GeneSal(proc_img);
	Mat salMap = SpMat.Sal2Img(superpixels, sal);
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(proc_img, CV_8U, 1.0);
    Mat blur, most_salient2, backproj;
    bilateralFilter(proc_img, blur, 12, 24, 6);
    
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    minMaxLoc(blur, &minVal, &maxVal, &minLoc, &maxLoc);


    threshold(blur, most_salient2, 0, 255, THRESH_BINARY + THRESH_OTSU);
    //imshow("most_salient2", most_salient2);
	Mat filtered2;
    // Another option is to use dilate/erode/dilate:
	int morph_operator = 1; // 0: opening, 1: closing, 2: gradient, 3: top hat, 4: black hat
	int morph_elem = 2; // 0: rect, 1: cross, 2: ellipse
	int morph_size = 10; // 2*n + 1
    int operation = morph_operator + 2;

    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

  // Apply the specified morphology operation
    morphologyEx( most_salient2, filtered2, operation, element );
    imshow("filtered2", filtered2);
    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3);

    int ch[] = {0, 0};
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    cvtColor(image, hsv, COLOR_RGB2HSV);
    hue.create(hsv.size(), hsv.depth());
    mixChannels(&hsv, 1, &hue, 1, ch, 1);
    while(1)
    {    
        if( trackObject < 0 )
        {
            Mat roi(hue, selection), maskroi(filtered2, selection);
            calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
            //trackWindow = selection;
            trackObject = 1;

            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            backproj = backproj & filtered2;
            imshow("backproj", backproj);
            /*if( trackWindow.area() <= 1 )
            {
                int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                 trackWindow.x + r, trackWindow.y + r) &
                                 Rect(0, 0, cols, rows);
            }*/

        }
       

        imshow( "src", image );
        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c)
        {
        case 'c':
            trackObject = 0;
            break;
        default:
            ;
        }
    }
	return 0;
}
