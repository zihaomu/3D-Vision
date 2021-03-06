#include <iostream>
#include "octree.h"
#include <opencv2/viz/viz3d.hpp>
#include <fstream>

using namespace std;
using namespace cv;

vector<Point3f> loadPointCloud(string fileName)
{
    vector<Point3f> pointCloud;

    ifstream ifs(fileName);
    string str;

    for(size_t i = 0; i < 12; ++i)
        getline(ifs, str);

    Point3f temp;
    float dummy1, dummy2;

    for(size_t i = 0; i < 1889; ++i)
    {
        ifs >> temp.x >> temp.y >> temp.z >> dummy1 >> dummy2;
        temp *= 5.0f;
        pointCloud.push_back(temp);
    }

    return pointCloud;
}

void Traverse( OctreeNode*& node, viz::WWidgetMerger& merger){

    if(node == nullptr)
    {
        std::cerr<<"node empty"<<std::endl;
        return;
    }
    else
    {
    viz::WCube cubeW(node->origin, node->origin + Point3f(node->size, node->size, node->size), !(node->isLeaf), viz::Color::white());
    merger.addWidget(cubeW);

    for(size_t child_index = 0; child_index < 8; ++child_index)
    {
        if(node->children[child_index] != nullptr)
            Traverse(node->children[child_index], merger);
    }
    }
}

int main()
{
    string bunny_name = "../data/bunny.ply";
    vector<Point3f> data = loadPointCloud(bunny_name);
    Octree tree(6, data);
    cout<<"load point cloud successfully."<<endl;

    // test

//    if(tree.index(data[0], tree.rootNode) != nullptr)
//    {
//        cout<<"node is in bound!!"<<endl;
//    } else{
//        cout<<"node is not in bound!!"<<endl;
//    }
//
//    if(tree.deletePoint(data[0]))
//    {
//        cout<<"delete success!!"<<endl;
//    } else{
//        cout<<"delete fail!!"<<endl;
//    }
//
//    if(tree.deletePoint(data[0]))
//    {
//        cout<<"delete success!!"<<endl;
//    } else{
//        cout<<"delete fail!!"<<endl;
//    }

    // Visualization
    viz::Viz3d myWindow("Octree");  // create window

    viz::WWidgetMerger merger;

    Traverse(tree.rootNode, merger);

    merger.finalize();
    myWindow.showWidget("Cube Widget", merger);

    myWindow.spin();

    return 0;
} 