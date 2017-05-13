#pragma once

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/nonfree/nonfree.hpp"  
#include "opencv2/nonfree/features2d.hpp" 
#include <stdio.h>  
#include <stdlib.h>
#include <iostream>

using namespace cv;  
using namespace std;  


#define PI 3.1415926535898

class Particle {
public:
	Point2f p1, p2, pm;
	float angle;
	float l;
	Particle() {};
	Particle(Point2f p1, Point2f p2, Point2f pm, float angle) {
		this->p1 = p1;
		this->p2 = p2;
		this->pm = pm;
		this->angle = angle;
		this->l = norm(p1 - p2);
	}
	void flip() {
		Point2f pt = p1;
		p1 = p2;
		p2 = pt;
	}
	void paint(Mat Img) {
		circle(Img,pm,3,Scalar(0,255,255),-1);
		float x=pm.x,y=pm.y;
		x+=7*cos(angle);
		y+=7*sin(angle);
		line(Img,pm,Point2f(x,y),Scalar(0,0,255));
	}
};

class keynode{
public:
	int top;
	double d;
	keynode(){top=-1;d=1e10;}
	keynode(double d1,int t1){top=t1;d=d1;}
};

float toAngle(float a,int k=1){return a-floor(a/k/PI)*k*PI;}

float toInclinedAngle(float a){
	if (a>PI){
		return 2*PI-a;
	}
	return a;
}

vector<Particle> creatParticle(const Mat &OriImg){
	initModule_nonfree();//初始化模块，使用SIFT或SURF时用到  
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );//创建SIFT特征检测器  
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );//创建特征向量生成器  
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//创建特征匹配器  
	if(detector.empty()||descriptor_extractor.empty()){
		cout<<"fail to create detector!\n";
	}

	/*生成镜像*/
	Mat MirImg = OriImg.clone();
	int rows=OriImg.rows;
	int cols=OriImg.cols;
	for (int i = 0;i < rows;++i)
	{
		for (int j = 0;j < cols;++j)
		{
			MirImg.at<Vec3b>(i,j) = OriImg.at<Vec3b>(i,cols-j-1);
		}
	}

	/*imshow("镜像",MirImg);
	imshow("原图像",OriImg);
	waitKey();*/

	/*特征点检测*/
	vector<KeyPoint> keypoints1,keypoints2;
	detector->detect( OriImg, keypoints1 );
	detector->detect( MirImg, keypoints2 );
	
	//画出特征点
	cout<<keypoints1.size()<<","<<keypoints2.size()<<endl;
    Mat img_keypoints1,img_keypoints2;
    drawKeypoints(OriImg,keypoints1,img_keypoints1,Scalar::all(-1),0);
    drawKeypoints(MirImg,keypoints2,img_keypoints2,Scalar::all(-1),0);
	imshow("Src1",img_keypoints1);
    imshow("Src2",img_keypoints2);	
	waitKey();

	/*根据特征点计算特征描述子矩阵，即特征向量矩阵*/
    Mat descriptors1,descriptors2;
    descriptor_extractor->compute( OriImg, keypoints1, descriptors1 );
    descriptor_extractor->compute( MirImg, keypoints2, descriptors2 );

	/*寻找局部对称点，生成particles*/
	vector<Particle> particles;	
	int num1=keypoints1.size();
	for(int i=0;i<num1;i++){
		vector<keynode> keylist(3);
		Mat descriptor1=descriptors1.row(i);
		int num2=keypoints2.size();
		for(int j=0;j<num2;j++){
			Mat descriptor2=descriptors2.row(j);
			float D=norm(descriptor1,descriptor2,CV_L2);
			if(D<keylist[1].d){
				keynode nk(D,j);
				if(D<keylist[0].d){
					keylist.insert(keylist.begin(),nk);
				}
				else{
					keylist.insert(keylist.begin()+1,nk);
				}
				keylist.pop_back();
			}
			else if(D<keylist[2].d){
				keynode nk(D,j);
				keylist.insert(keylist.begin()+2,nk);
				keylist.pop_back();
			}					
		}
		
		KeyPoint &keypoint1=keypoints1[i];
		float size1=keypoint1.size;
		float x1=keypoint1.pt.x,y1=keypoint1.pt.y;
		float angle1=keypoint1.angle;
		for(int j=0;j<3;j++){
			int k=keylist[j].top;
			if(k!=-1){
				KeyPoint &keypoint2=keypoints2[k];
				if(abs(size1/keypoint2.size)<2&&abs(size1/keypoint2.size)>0.5){
					float x2=float (cols)-keypoint2.pt.x,y2=keypoint2.pt.y;
					float angle2=keypoint2.angle;
					float orientation=toAngle((angle1+angle2)*PI/180/2);
 					if(sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))>0.5&&abs(PI/2-abs(orientation-toAngle(atan2(y2-y1,x2-x1))))<0.5){
							Point2f pm((x1+x2)/2,(y1+y2)/2);
							int psize=particles.size();
							int t;
							for(t=0;t<psize;t++)
							{
								if(norm(particles[t].pm-pm)<1){
									break;
								}
							}
							if(t==psize){
								Particle p(keypoint1.pt,Point2f(x2,y2),pm,orientation);
								particles.push_back(p);
							}
							//j=3;
					}
				}
			}
		}		
	}	
	int s=particles.size();
	cout<<s<<endl;
	Mat ParImg=OriImg.clone();
	for(int i=0;i<s;i++){
		particles[i].paint(ParImg);
	}	
	imshow("Particles",ParImg);
	waitKey();
	return particles;
}