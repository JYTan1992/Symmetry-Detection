#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>
#include <iostream>
#include "SIFT.h"

using namespace cv;

float U(const Point2f& p1, const Point2f& p2) {
	float r_2 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
	if (r_2 < 1e-9) {
		return 0.0;
	}
	return r_2 * std::log(r_2);
}

void TPS(const Mat& srcImage, Mat& dstImage, const Size& dstSize,
	const std::vector<Point2f>& srcPoints, const std::vector<Point2f>& dstPoints,
	const float lambda = 0.0, const int interpolation = INTER_LINEAR)
{
	int n = dstPoints.size();

	Mat_<float> K = Mat::eye(n, n, CV_32F) * lambda;
	Mat_<float> P = Mat::ones(n, 3, CV_32F);
	Mat_<float> L = Mat::zeros(n+3, n+3, CV_32F);
	Mat_<float> V = Mat::zeros(n+3, 2, CV_32F);

	for (int y = 0; y < n; y++) {
		for (int x = 0; x < n; x++) {
			if (y != x) {
				K(y, x) = U(dstPoints[y], dstPoints[x]);
			}
		}
		P(y, 1) = dstPoints[y].x;
		P(y, 2) = dstPoints[y].y;
		V(y, 0) = srcPoints[y].x;
		V(y, 1) = srcPoints[y].y;
	}

	K.copyTo(L(Range(0, n), Range(0, n)));
	P.copyTo(L(Range(0, n), Range(n, n+3)));
	L(Range(n, n+3), Range(0, n)) = P.t();

	Mat_<float> C = L.inv() * V;
	/*
	std::cout << "K:" << std::endl << K << std::endl;
	std::cout << "P:" << std::endl << P << std::endl;
	std::cout << "L:" << std::endl << L << std::endl;
	std::cout << "V:" << std::endl << V << std::endl;
	std::cout << "C:" << std::endl << C << std::endl;
	*/
	Size srcSize = srcImage.size();
	int x_max = dstSize.width > srcSize.width ? dstSize.width : srcSize.width;
	int y_max = dstSize.height > srcSize.height ? dstSize.height : srcSize.height;

	Size newSize = Size(x_max, y_max);
	Mat newImage = Mat(newSize, srcImage.type(), Scalar());
	srcImage.copyTo(newImage(Range(0, srcSize.height), Range(0, srcSize.width)));

	Mat_<float> mapx = Mat(newSize, CV_32F, Scalar(-1.0));
	Mat_<float> mapy = Mat(newSize, CV_32F, Scalar(-1.0));

	for (int y = 0; y < dstSize.height; y++) {
		for (int x = 0; x < dstSize.width; x++) {
			for (int j = 0; j < 2; j++) {
				float a1 = C(n  , j);
				float ax = C(n+1, j);
				float ay = C(n+2, j);
				float sum1 = a1 + ax*x + ay*y;
				float sum2 = 0.0;
				for (int i = 0; i < n; i++) {
					sum2 += C(i, j) * U(Point2f(x, y), dstPoints[i]);
				}
				if (j == 0) {
					mapx(y, x) = sum1 + sum2;
				} else {
					mapy(y, x) = sum1 + sum2;
				}
			}
		}
	}
	/*
	std::cout << "mapx:" << std::endl << mapx << std::endl;
	std::cout << "mapy:" << std::endl << mapy << std::endl;
	*/
	remap(newImage, dstImage, mapx, mapy, interpolation);
}

//float NCC(const Mat& image, const std::vector<Point2f>& points, bool show = false) {
	//std::cout << image << std::endl << std::endl;
	//std::cout << mean(image) << std::endl << std::endl;
	//Mat normedImage = image - mean(image);
	//std::cout << normedImage << std::endl << std::endl;
	//Mat normedImage = image.clone();
	//Mat_<float> normedImage = Mat(image.size(), CV_32FC3, Scalar());
	//std::cout << image.depth() << std::endl;
	//image.convertTo(normedImage, CV_32F);
	/*
	namedWindow("NCC0", WINDOW_AUTOSIZE);
	imshow("NCC0", normedImage);
	waitKey();
	*/
	/*
	int n = points.size() / 3;
	for (int i = 0; i < n-1; i++) {
		float y1  = points[3*i    ].y;
		float y2  = points[3*i + 3].y;
		float x11 = points[3*i    ].x;
		float x12 = points[3*i + 2].x;
		float x21 = points[3*i + 3].x;
		float x22 = points[3*i + 5].x;
		for (int y = std::ceil(y1); y < std::ceil(y2); y++) {
			for (int x = 0; (x - x11) * (y2 - y1) < (x21 - x11) * (y - y1); x++) {
				normedImage.at<Vec3b>(y, x) = Vec3b(255,255,255);
			}
			for (int x = normedImage.size().width - 1; (x - x22) * (y2 - y1) > (x12 - x22) * (y2 - y); x--) {
				normedImage.at<Vec3b>(y, x) = Vec3b(255,255,255);
			}
		}
	}
	*/
	/*
	namedWindow("NCC", WINDOW_AUTOSIZE);
	imshow("NCC", normedImage);
	waitKey();
	*/
	//Mat_<float> mirrorImage = Mat(image.size(), CV_32FC3, Scalar());
	//flip(normedImage, mirrorImage, 1);
	//Mat_<float> product = normedImage.mul(mirrorImage);
	//Mat_<float> square;
	//pow(normedImage, 2, square);
	//Scalar sum1 = sum(product);
	//Scalar sum2 = sum(square);
	/*
	if (show) {
		std::cout << normedImage << std::endl << std::endl;
		namedWindow("NCC1", WINDOW_AUTOSIZE);
		imshow("NCC1", normedImage);
		waitKey();
		std::cout << mirrorImage << std::endl << std::endl;
		namedWindow("NCC2", WINDOW_AUTOSIZE);
		imshow("NCC2", mirrorImage);
		waitKey();
		std::cout << product << std::endl << std::endl;
		std::cout << square << std::endl << std::endl;
	}
	*/
	/*
	if (sum2[0] + sum2[1] + sum2[2] == 0) {
		return 0.8;
	}
	return 1.0 * (sum1[0] + sum1[1] + sum1[2]) / (sum2[0] + sum2[1] + sum2[2]);
	*/
//}

float NCC(const Mat& image, bool show = false, bool trape = false, const std::vector<Point2f>& points = std::vector<Point2f>(1)) {
	if (show) {
		namedWindow("NCC", WINDOW_AUTOSIZE);
		imshow("NCC", image);
		waitKey(10);
	}
	Scalar m = mean(image);
	double num = 0.0, den = 0.0;
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int c = 0; c < image.channels(); c++) {
				num += (image.at<Vec3b>(y, x)[c] - m[c]) * (image.at<Vec3b>(y, image.cols-1-x)[c] - m[c]);
				den += (image.at<Vec3b>(y, x)[c] - m[c]) * (image.at<Vec3b>(y, x)[c] - m[c]);
			}
		}
	}
	if (trape) {
		int n = points.size() / 3;
		for (int i = 0; i < n-1; i++) {
			float y1  = points[3*i    ].y;
			float y2  = points[3*i + 3].y;
			float x11 = points[3*i    ].x;
			float x12 = points[3*i + 2].x;
			float x21 = points[3*i + 3].x;
			float x22 = points[3*i + 5].x;
			for (int y = std::ceil(y1); y < std::ceil(y2); y++) {
				for (int x = 0; (x - x11) * (y2 - y1) < (x21 - x11) * (y - y1); x++) {
					for (int c = 0; c < image.channels(); c++) {
						num -= (image.at<Vec3b>(y, x)[c] - m[c]) * (image.at<Vec3b>(y, image.cols-1-x)[c] - m[c]);
						den -= (image.at<Vec3b>(y, x)[c] - m[c]) * (image.at<Vec3b>(y, x)[c] - m[c]);
					}
				}
				for (int x = image.cols-1; (x - x22) * (y2 - y1) > (x12 - x22) * (y2 - y); x--) {
					for (int c = 0; c < image.channels(); c++) {
						num -= (image.at<Vec3b>(y, x)[c] - m[c]) * (image.at<Vec3b>(y, image.cols-1-x)[c] - m[c]);
						den -= (image.at<Vec3b>(y, x)[c] - m[c]) * (image.at<Vec3b>(y, x)[c] - m[c]);
					}
				}
			}
		}
	}
	if (den < 1e-9) {
		return 1.0;
	}
	return 1.0 * num / den;
}

bool left(const Point2f& p, const Point2f& p1, const Point2f& p2) {
	return (p1.x - p.x) * (p2.y - p1.y) > (p1.y - p.y) * (p2.x - p1.x);
}

float evaluate(const Mat& srcImage, Particle& ptc1, Particle& ptc2, bool show = false) {
	if (!left(ptc1.p1, ptc1.pm, ptc2.pm)) {
		ptc1.flip();
	}
	if (!left(ptc2.p1, ptc1.pm, ptc2.pm)) {
		ptc2.flip();
	}

	std::vector<Point2f> srcPoints(6), dstPoints(6);

	srcPoints[0] = ptc1.p1;
	srcPoints[1] = ptc1.pm;
	srcPoints[2] = ptc1.p2;
	srcPoints[3] = ptc2.p1;
	srcPoints[4] = ptc2.pm;
	srcPoints[5] = ptc2.p2;

	int width = std::ceil(ptc1.l > ptc2.l ? ptc1.l : ptc2.l);
	
	if (width < 4) {
		width = 4;
	}

	float height = norm(ptc1.pm - ptc2.pm);
	Size dstSize = Size(width+1, std::ceil(height)+1);

	dstPoints[0] = Point2f(1.0*width/2 - 1.0*ptc1.l/2, 0.0);
	dstPoints[1] = Point2f(1.0*width/2               , 0.0);
	dstPoints[2] = Point2f(1.0*width/2 + 1.0*ptc1.l/2, 0.0);
	dstPoints[3] = Point2f(1.0*width/2 - 1.0*ptc2.l/2, height);
	dstPoints[4] = Point2f(1.0*width/2               , height);
	dstPoints[5] = Point2f(1.0*width/2 + 1.0*ptc2.l/2, height);

	Mat dstImage;
	TPS(srcImage, dstImage, dstSize, srcPoints, dstPoints);
	if (show) {
		namedWindow("evaluate", WINDOW_AUTOSIZE);
		imshow("evaluate", dstImage);
		waitKey(10);
	}
	/*namedWindow("evaluate", WINDOW_AUTOSIZE);
	imshow("evaluate", dstImage);
	waitKey();*/
	return NCC(dstImage(Range(0, dstSize.height), Range(0, dstSize.width)), show);
}

float rectify(const Mat& srcImage, std::vector<Particle>& particles, bool show = false) {
	int n = particles.size();

	if (!left(particles[0].p1, particles[0].pm, particles[1].pm)) {
		particles[0].flip();
	}
	for (int i = 1; i < n; i++) {
		if (!left(particles[i].p1, particles[i-1].pm, particles[i].pm)) {
			particles[i].flip();
		}
	}

	std::vector<Point2f> srcPoints(3*n), dstPoints(3*n);
	float l_max = 0.0;

	for (int i = 0; i < n; i++) {
		srcPoints[3*i    ] = particles[i].p1;
		srcPoints[3*i + 1] = particles[i].pm;
		srcPoints[3*i + 2] = particles[i].p2;
		l_max = particles[i].l > l_max ? particles[i].l : l_max;
	}

	int width = std::ceil(l_max);
	float height = 0.0;

	dstPoints[0] = Point2f(1.0*width/2 - 1.0*particles[0].l/2, 0.0);
	dstPoints[1] = Point2f(1.0*width/2                       , 0.0);
	dstPoints[2] = Point2f(1.0*width/2 + 1.0*particles[0].l/2, 0.0);
	for (int i = 1; i < n; i++) {
		height += norm(particles[i].pm - particles[i-1].pm);
		dstPoints[3*i    ] = Point2f(1.0*width/2 - 1.0*particles[i].l/2, height);
		dstPoints[3*i + 1] = Point2f(1.0*width/2                       , height);
		dstPoints[3*i + 2] = Point2f(1.0*width/2 + 1.0*particles[i].l/2, height);
	}

	Size dstSize = Size(width+1, std::ceil(height)+1);

	Mat dstImage;
	TPS(srcImage, dstImage, dstSize, srcPoints, dstPoints);
	if (show) {
		namedWindow("rectify", WINDOW_AUTOSIZE);
		imshow("rectify", dstImage);
		waitKey();
	}
	/*namedWindow("rectify", WINDOW_AUTOSIZE);
	imshow("rectify", dstImage);
	waitKey();*/
	return NCC(dstImage(Range(0, dstSize.height), Range(0, dstSize.width)), show, true, dstPoints);
}


