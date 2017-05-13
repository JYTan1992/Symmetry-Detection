#pragma once
#include <vector>
#include <list>
#include <iostream>
using namespace std;

class Cost{
public:	
	float ncc,s;
	Cost(float ncc=0,float s=0){
		this->ncc=ncc;
		this->s=s;
	}
	float getValue(){
		return ncc+0.3*log(s);
	}
	Cost operator + (const Cost &c){
		float s1=c.s+s;
		float ncc1=(c.ncc*c.s+ncc*s)/s1;
		Cost c1(ncc1,s1);		
		return c1;
	}
};

class Edge{
public:
	bool connected;
	Cost c;
	int v1,v2;
	float dir;
	Edge(){connected=false;v1=-1;v2=-1;}
	void set(int i,int j,Cost c1,float d){
		connected=true;
		c=c1;
		v1=i;
		v2=j;
		dir=d;
	}
	void change(int i,int j){
		v1=i;
		v2=j;
	}
};

class Path{
public:
	float front,back;
	Cost c;
	list<int> nodes;
	bool forward,backward;
	Path(int v){
		back=1.5*PI;
		front=0.5*PI;		
		nodes.push_back(v);
		forward=true;
		backward=true;
	}
	void push_back(Edge e){
		nodes.push_back(e.v2);
		c=c+e.c;
		back=e.dir;
	}
	void push_front(Edge e){
		nodes.push_front(e.v2);
		c=c+e.c;
		front=e.dir;
	}
	friend bool operator < (Path p1,Path p2){
		return p1.c.getValue()<p2.c.getValue();
	}
	int getBack(){return nodes.back();}
	int getFront(){return nodes.front();}
	bool contain(int n){
		list<int>::iterator it = find(nodes.begin(), nodes.end(),n); 
		return it != nodes.end();
	}
	int size(){return nodes.size();}
	void changeNCC(float ncc){c.ncc=ncc;}
};

template<class T>
class Graph{
private:
	vector<T> vertex; 
	Edge **edge;
	int d;
public:
	Graph(vector<T> v);
	int size(){return d;};
	T getVertex(int n){return vertex[n];};	
	Edge getEdge(int i,int j){return edge[i][j];};
	void addEdge(int i,int j,Cost c,float a);
	vector<int> getConnectedEdges(int i);
	void setEdge(Edge **e);
	Graph getConnectedDomain(); 
};

template<class T>
Graph<T>::Graph(vector<T> v){	
	d=v.size();
	vertex=v;
	edge = new Edge *[d];
	for(int i=0;i<d;i++) edge[i]=new Edge[d];
}

template<class T>
void Graph<T>::addEdge(int i,int j,Cost c,float d){
	edge[i][j].set(i,j,c,toAngle(d,2));
	edge[j][i].set(j,i,c,toAngle(d+PI,2));
}

template<class T>
vector<int> Graph<T>::getConnectedEdges(int i){	
	vector<int> nodes;
	for (int j=0;j<d;j++){
		if (edge[i][j].connected){
			nodes.push_back(j);
		}
	}
	return nodes;
}

template<class T>
void Graph<T>::setEdge(Edge **e){
	for (int i=0;i<d;i++)
	{
		for (int j=0;j<d;j++)
		{
			edge[i][j]=e[i][j];
		}
	}
}

template<class T>
Graph<T> Graph<T>::getConnectedDomain(){
	vector<int> note(d,0);
	vector<int> id;
	int num=0;
	for(int i=0;i<d;i++){
		if(note[i]==0){
			int num1=0;
			vector<int> id1;
			list<int> nodes;
			nodes.push_back(i);
			while(!nodes.empty()){
				int node1=nodes.front();
				nodes.pop_front();
				if(note[node1]==0){
					note[node1]=1;
					num1++;	
					id1.push_back(node1);
					vector<int> n1=this->getConnectedEdges(node1);
					nodes.insert(nodes.end(),n1.begin(),n1.end());
				}
			}		
			if (num1>num)
			{
				num=num1;
				id=id1;
			}
		}
	}
	int l=id.size();
	vector<T> v;
	Edge **e = new Edge *[l];
	for(int i=0;i<l;i++) e[i]=new Edge[l];
	for(int i=0;i<l;i++){
		int v1=id[i];
		v.push_back(vertex[v1]);		
		for(int j=0;j<l;j++){
			int v2=id[j];
			Edge et=edge[v1][v2];
			et.change(i,j);
			e[i][j]=et;
		}
	}	
	Graph<T> g(v);
	g.setEdge(e);
	return g;
}

