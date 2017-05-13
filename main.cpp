#include "TPS.h"
#include "graph.h"
#include<queue>
#include <string>

using namespace cv;  
using namespace std; 

const int K = 100;

void dashline(Mat Img,Point2f v1,Point2f v2,Scalar color,float solid=4,float space=4){
	int n=norm(v1-v2)/(solid+space);
	float ang=atan2(v2.y-v1.y,v2.x-v1.x);
	Point2f t1=v1,t2=v1+Point2f(solid*cos(ang),solid*sin(ang));
	Point2f d((solid+space)*cos(ang),(solid+space)*sin(ang));
	for(int i=0;i<n;i++){
		line(Img,t1,t2,color,1,CV_AA);
		t1+=d;
		t2+=d;
	}
	line(Img,t2,v2,color,1,CV_AA);
}


int main(){
	Mat OriImg = imread("back.jpg");
	if(OriImg.empty()){
		cout<<"Fail to load back.jpg!\n";
	}
	vector<Particle> particles=creatParticle(OriImg);
	int l=particles.size();
	Graph<Particle> graph(particles);
	//建立边
	for(int i=0;i<l;i++){
		Particle P1=graph.getVertex(i);
		for (int j=i+1;j<l;j++){			
			Particle P2=graph.getVertex(j);;
			float a1=P1.angle,a2=P2.angle;
			Point2f pm1=P1.pm,pm2=P2.pm;
			float a12=atan2(pm2.y-pm1.y,pm2.x-pm1.x);
			if (abs(a1-a2)<PI/6&&abs(toAngle(a12)-0.5*(a1+a2))<PI/6){
				float NCC=evaluate(OriImg,P1,P2);
				//cout<<NCC<<endl;
				if(NCC>0.5){
					Cost c(NCC,0.5*(P1.l+P2.l)*norm(pm1-pm2));
					graph.addEdge(i,j,c,toAngle(a12,2));
				}
			}
		}
	}
	Graph<Particle> maxGraph=graph.getConnectedDomain();
	l=maxGraph.size();
	//选择初始点
	int n=-1;
	float f=65536;
	Point2f hypoPoint(0.52*OriImg.cols,0.32*OriImg.rows);
	for (int i=0;i<l;i++){		
		float f1=norm(hypoPoint-maxGraph.getVertex(i).pm);
		if (f1<f){
			f=f1;
			n=i;
		}
	}


	Mat ParImg;
	resize(OriImg, ParImg, Size(3*OriImg.size().width, 3*OriImg.size().height));
	for(int i=0;i<l;i++){
		Particle P1=maxGraph.getVertex(i);
		Point2f v1=P1.pm;
		circle(ParImg,v1*3,3,Scalar(255,0,255),-1); 
		vector<int> v2s=maxGraph.getConnectedEdges(i);
		int num=v2s.size();
		for(int j=0;j<num;j++){
			int id2=v2s[j];
			if(i==n){
				Point2f v2=maxGraph.getVertex(id2).pm;
				cout<<"("<<i<<","<<id2<<")"<<" "<<maxGraph.getEdge(i,id2).dir<<endl;
				dashline(ParImg,v1*3,v2*3,Scalar(0,255,0),4,20);
			}
		}
	}	
	circle(ParImg,hypoPoint*3,5,Scalar(255,255,255));  
	namedWindow("new Particles");
	imshow("new Particles",ParImg);
	waitKey();

	Path h(n);
	priority_queue<Path> hypo;
	hypo.push(h);
	bool forward_completed=false,backward_completed=false;
	int direction=0;
	while (!forward_completed||!backward_completed)
	{
		//all_completed=true;
		if (direction%2==0){
			backward_completed=true;
		}
		else{
			forward_completed=true;
		}
		int s=hypo.size();
		priority_queue<Path> temp;
		for (int i=0;i<s&&i<K;i++)
		{
			Path pt1=hypo.top();
			hypo.pop();
			if ((direction%2==1&&!pt1.forward)||(direction%2==0&&!pt1.backward)){
				temp.push(pt1);
			}
			else{
				bool backward_extendable,forward_extendable;
				int v1;
				if (direction%2==0){
					v1=pt1.getBack();
					backward_extendable=false;
				}
				else{
					v1=pt1.getFront();
					forward_extendable=false;
				}

				vector<int> v2s=maxGraph.getConnectedEdges(v1);
				int l=v2s.size();
				for(int j=0;j<l;j++){
					int v2=v2s[j];
					if(!pt1.contain(v2)){
						Path pt=pt1;
						Edge e=maxGraph.getEdge(v1,v2);
						if(direction%2==0){
							if(toInclinedAngle(abs(e.dir-pt1.back))<PI/5){
								pt.push_back(e);
								temp.push(pt);
								backward_extendable=true;
								backward_completed=false;
							}
						}
						else{
							if(toInclinedAngle(abs(e.dir-pt1.front))<PI/5){
								pt.push_front(e);
								temp.push(pt);
								forward_extendable=true;
								forward_completed=false;
							}
						}
					}
				}
				if (direction%2==0){
					pt1.backward=backward_extendable;
					if(!pt1.backward){
						temp.push(pt1);
					}					
				}
				else{
					pt1.forward=forward_extendable;
					if (!pt1.forward){
						temp.push(pt1);
					}
				}				
			}			
		}
		hypo.swap(temp);
		direction++;
	}

	priority_queue<Path> result;
	Path po=hypo.top();
	while (!hypo.empty()){		
		Path p=hypo.top();
		hypo.pop();
		list<int> lis=p.nodes;
		vector<Particle> df;
		list<int>::iterator ite;
		for (ite=lis.begin(); ite != lis.end(); ++ite)
		{
			df.push_back(maxGraph.getVertex(*ite));
		} 
		float new_NCC=rectify(OriImg,df);
		p.changeNCC(new_NCC);
		result.push(p);
	}

	Mat FinImg=OriImg.clone();	
	list<int> li=po.nodes;
	Point2f p1=maxGraph.getVertex(li.front()).pm;
	li.pop_front();
	circle(FinImg,p1,3,Scalar(0,255,255));
	while (!li.empty())	{
		Point2f p2=maxGraph.getVertex(li.front()).pm;
		li.pop_front();
		circle(FinImg,p2,3,Scalar(0,255,255));
		line(FinImg,p1,p2,Scalar(255,255,0));
		p1=p2;
	}
	imshow("result",FinImg);

	Mat ResImg;
	int cols=OriImg.cols,rows=OriImg.rows;
	ResImg.create(Size(cols*10, rows*3), CV_8UC3);
	for (int i=0;i<30&&!result.empty();i++){
	Path fp=result.top();
	result.pop();
	li=fp.nodes;
	p1=maxGraph.getVertex(li.front()).pm;
	li.pop_front();
	Mat top10 = OriImg.clone();
	circle(top10,p1,3,Scalar(0,255,255));
	while (!li.empty())	{
		Point2f p2=maxGraph.getVertex(li.front()).pm;
		li.pop_front();
		circle(top10,p2,3,Scalar(0,255,255));
		line(top10,p1,p2,Scalar(255,255,255));
		p1=p2;
	}	
	Mat imgROI = ResImg(Rect((i%10)*cols, int(i/10)*rows, cols, rows));  
	resize(top10, imgROI, Size(cols, rows));  	
	}
	imshow("Top",ResImg);
	waitKey();
	/*
	Mat ResImg;
	list<int> li;
	Point2f p1;
	int cols=OriImg.cols,rows=OriImg.rows;
	ResImg.create(Size(cols*10, rows*3), CV_8UC3);
//	for(int i=0;i<30&&!hypo.empty();i++) hypo.pop();
	for (int i=0;i<30&&!hypo.empty();i++){
		Path fp=hypo.top();
		hypo.pop();
		li=fp.nodes;
		p1=maxGraph.getVertex(li.front()).pm;
		li.pop_front();
		Mat top10 = OriImg.clone();
		circle(top10,p1,3,Scalar(0,255,255));
		while (!li.empty())	{
			Point2f p2=maxGraph.getVertex(li.front()).pm;
			li.pop_front();
			circle(top10,p2,3,Scalar(0,255,255));
			line(top10,p1,p2,Scalar(255,255,255));
			p1=p2;
		}	
		Mat imgROI = ResImg(Rect((i%10)*cols, int(i/10)*rows, cols, rows));  
		resize(top10, imgROI, Size(cols, rows));  	
	}
	imshow("Top",ResImg);
	waitKey();*/
}