//Lane line dection using OpenCV in C++
//From the Python version: https://github.com/georgesung/road_lane_line_detection.git
//author: liuwein@126.com
/*
1. ./lane_line without filename will open the first camera
2. ./lane_line <video_filename/ pic_filename> will open a video or a picture
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;

//Canny Edge Detector
double low_threshold = 50;
double high_threshold = 150;

// Region-of-interest vertices
// We want a trapezoid shape, with bottom edge at the bottom of the image
double trap_bottom_width = 0.85;  // width of bottom edge of trapezoid, expressed as percentage of image width
double trap_top_width = 0.07;  // ditto for top edge of trapezoid
double trap_height = 0.4;  //height of the trapezoid expressed as percentage of image height

// Hough Transform 
double rho = 2;  //distance resolution in pixels of the Hough grid
double theta = CV_PI/180;  //angular resolution in radians of the Hough grid
double threshold = 15;	  //minimum number of votes (intersections in Hough grid cell)
double min_line_length = 10;  //minimum number of pixels making up a line
double max_line_gap = 20;	 // maximum gap in pixels between connectable line segments

void fiter_colors(cv::InputArray , cv::OutputArray );
void draw_straight_lines(cv::InputOutputArray , vector<cv::Vec4i>,  const cv::Scalar color=cv::Scalar(0,0,255), int thickness=10);

int main( int argc, char** argv )
{
    cv::VideoCapture cap;
    if (argc==1)
    {
        cap.open(0); // open the first camera if no input filename
    } 
    else
    {
        cap.open(argv[1]);
    }
    if( !cap.isOpened() ) 
    { // check if we succeeded
        std::cerr << "Couldn't open capture." << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;
    for(;;) 
    {
        if( frame.empty() ) break; // Ran out of film

        cv::Mat new_img, img_gray;
        fiter_colors(frame,new_img);
        cv::cvtColor(new_img, img_gray, CV_BGR2GRAY);

        cv::Mat blur_gray;
        cv::Size ksize = cv::Size(3,3);
        cv::GaussianBlur(img_gray, blur_gray, ksize, 0);

        cv::Mat edges;
        cv::Canny(blur_gray, edges, low_threshold, high_threshold);

        cv::Size imshape = edges.size();
        vector<vector<cv::Point2i> > roi_vertices;
        vector<cv::Point2i> roi_points(4);
        roi_points[0] = cvPoint((imshape.width * (1 - trap_bottom_width)) / 2, imshape.height);  
        roi_points[1]= cvPoint(imshape.width * (1 - trap_top_width) / 2, imshape.height - imshape.height * trap_height); 
        roi_points[2]= cvPoint(imshape.width - (imshape.width * (1 - trap_top_width)) / 2, imshape.height - imshape.height * trap_height); 
        roi_points[3]= cvPoint(imshape.width - (imshape.width * (1 - trap_bottom_width)) / 2, imshape.height);

        roi_vertices.push_back(roi_points) ; 
        cv::Mat mask =  cv::Mat(edges.size(), edges.type(), cv::Scalar(0));
        cv::fillPoly(mask, roi_vertices, 255);
        cv::Mat masked_edges;
        cv::bitwise_and(mask, edges, masked_edges);

        vector<cv::Vec4i> lines;
        cv::HoughLinesP(masked_edges, lines, rho, theta, threshold, min_line_length, max_line_gap);

        cv::Mat line_img = cv::Mat(frame.size(), frame.type(), cv::Scalar(0,0,0));
        draw_straight_lines(line_img, lines);
        cv::Mat annotated_image;
        cv::addWeighted(line_img, 0.8, frame, 1.0, 0, annotated_image);
        
        cv::imshow("new_img", annotated_image);

        if( cv::waitKey(30) >= 0 ) break;
        cap>> frame; //get a new frame
        if(frame.empty()) cv::waitKey(0);// wait here if it is a picture or the last frame of the video
    }
    return 0;
}

void fiter_colors(cv::InputArray img, cv::OutputArray new_img)
{
    //Filter white pixels
    cv::Scalar lower_white = cv::Scalar(200,200,200) ;
    cv::Scalar upper_white = cv::Scalar(255, 255, 255);
    cv::Mat white_mask, white_img;
    cv::inRange(img, lower_white, upper_white, white_mask);
    // cv::imshow("white_mask", white_mask);
    cv::bitwise_and(img, img, white_img, white_mask);

    //filter yellow pixels
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);  //python cv2.COLOR_BGR2HSV  not the same

    cv::Scalar lower_yellow = cv::Scalar(20,100,100);
    cv::Scalar upper_yellow = cv::Scalar(50,255,255);// They are different from the ranges in the original python code.
                                                        // I don't known why.
    cv::Mat yellow_mask, yellow_img;
    
    cv::inRange(hsv, lower_yellow, upper_yellow, yellow_mask);  //input hsv
    // cv::imshow("yellow_mask", yellow_mask);
    cv::bitwise_and(img, img, yellow_img, yellow_mask);
    // cv::namedWindow("yellow_img", CV_WINDOW_AUTOSIZE);
    // cv::imshow("yellow_img", yellow_img); //yelow_img has problem!!!
    cv::addWeighted(white_img, 1.0, yellow_img, 1.0, 0, new_img);    
}

void draw_straight_lines(cv::InputOutputArray img, vector<cv::Vec4i> lines,  const cv::Scalar color , int thickness)
{
    /*
    NOTE: this is the function you might want to use as a starting point once you want to 
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).  
	
	Think about things like separating line segments by their 
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of 
	the lines and extrapolate to the top and bottom of the lane.
	
	This function draws `lines` with `color` and `thickness`.	
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
    */
   if(lines.empty() || lines.size()==0) return;
   bool draw_right = true;
   bool draw_left = true;
   // Find slopes of all lines
   // But only care about lines where abs(slope) > slope_threshold
   double slope_threshold = 0.5;
   vector<double> slopes;
   vector<cv::Vec4i> new_lines;
   for( size_t i = 0; i < lines.size(); i++ )
   {
       int x1 = lines[i][0];
       int y1 = lines[i][1];
       int x2 = lines[i][2];
       int y2 = lines[i][3];
       double slope = 0;
       if(x1 == x2) slope = 999;
       else slope = (double)(y2 - y1) / (x2 - x1);
       
       if(abs(slope) > slope_threshold)
       {
           slopes.push_back(slope);
           new_lines.push_back(lines[i]);
       }
   }
   lines = new_lines;

   //Split lines into right_lines and left_lines, representing the right and left lane lines
   //Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
   vector<cv::Vec4i> right_lines, left_lines;
   for( size_t i = 0; i < lines.size(); i++ )
   {
       int x1 = lines[i][0];
       int y1 = lines[i][1];
       int x2 = lines[i][2];
       int y2 = lines[i][3];
       int img_x_center = img.size().width/2;
       if(slopes[i]>0 && x1 > img_x_center && x2 > img_x_center) right_lines.push_back(lines[i]);
       if(slopes[i]<0 && x1 < img_x_center && x2 < img_x_center) left_lines.push_back(lines[i]);
   }
   //Run linear regression to find best fit line for right and left lane lines
   //Right lane lines
   vector<cv::Point2i> right_lines_points;

   for( size_t i = 0; i < right_lines.size(); i++ )
   {
       int x1 = right_lines[i][0];
       int y1 = right_lines[i][1];
       int x2 = right_lines[i][2];
       int y2 = right_lines[i][3];
       right_lines_points.push_back(cvPoint(x1,y1));
       right_lines_points.push_back(cvPoint(x2,y2));
   }
   
   float right_m, right_b; //y = m*x + b
   cv::Vec4f fit_right_line;
   double param =0, reps=0.01, aeps =0.01;
    if(right_lines_points.size() > 0)
    {
        cv::fitLine(right_lines_points, fit_right_line, CV_DIST_L2, param, reps, aeps);
        right_m = fit_right_line[1]/fit_right_line[0];
        right_b = fit_right_line[3] - right_m * fit_right_line[2];
    }
    else
    {
        draw_right = false;
        right_m =1;
        right_b =1;
    }
    //Left lane lines
   vector<cv::Point2i> left_lines_points;

   for( size_t i = 0; i < left_lines.size(); i++ )
   {
       int x1 = left_lines[i][0];
       int y1 = left_lines[i][1];
       int x2 = left_lines[i][2];
       int y2 = left_lines[i][3];
       left_lines_points.push_back(cvPoint(x1,y1));
       left_lines_points.push_back(cvPoint(x2,y2));
   }
   
    float left_m, left_b; //y = m*x + b
    cv::Vec4f fit_left_line;
    if(left_lines_points.size() > 0)
    {
        cv::fitLine(left_lines_points, fit_left_line, CV_DIST_L2, param,reps, aeps);
        left_m = fit_left_line[1]/fit_left_line[0];
        left_b = fit_left_line[3] - left_m * fit_left_line[2];
    }
    else
    {
        draw_left = false;
        left_m =1;
        left_b =1;
    }

    // Find 2 end points for right and left lines, used for drawing the line
	// y = m*x + b --> x = (y - b)/m
    int y1 = img.size().height;
    int y2 = img.size().height * (1 - trap_height);

    int right_x1 = (int)((y1 - right_b) / right_m);
    int right_x2 = (int)((y2 - right_b) / right_m);

    int left_x1 = (int)((y1 - left_b) / left_m);
	int left_x2 = (int)((y2 - left_b) / left_m);

    if(draw_right) cv::line(img, cvPoint(right_x1, y1), cvPoint(right_x2, y2), color, thickness);
    if(draw_left) cv::line(img, cvPoint(left_x1, y1), cvPoint(left_x2, y2), color, thickness);

}
