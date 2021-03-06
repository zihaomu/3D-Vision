// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <vector>
#include "octree.h"
#include "opencv2/core.hpp"

namespace cv{

    OctreeNode::OctreeNode(int _depth, double _size, Point3f _origin, int _parentIndex):children(8),depth(_depth),size(_size),origin(
            _origin),parentIndex(_parentIndex)
    {
    }

    void OctreeNode::clear()
    {
        if(isLeaf)
        {
            if(parentIndex != -1)
            {
                parent->children[parentIndex] = nullptr;
            }
            delete this;
        }
        else
        {
            for(int i = 0; i<childNum;i++)
            {
                if(children[i] != nullptr)
                {
                    children[i]->clear();
                }
            }
            if(parentIndex != -1)
            {
                parent->children[parentIndex] = nullptr;
            }
            delete this;
        }
    }

    Octree::Octree(int _maxDepth, double _size, Point3f _origin ):maxDepth(_maxDepth),size(_size),origin(_origin)
    {
    }

    Octree::Octree(int _maxDepth, std::vector<Point3f>& _pointCloud):maxDepth(_maxDepth),size(0)
    {
        convertFromPointCloud(_pointCloud);
    }

    Octree::Octree(int _maxDepth):maxDepth(_maxDepth), size(0), origin(0,0,0)
    {
    }

    void Octree::insertPoint(OctreeNode*& node, Point3f &point)
    {
        if(node == nullptr)
        {
            node = new OctreeNode( 0, size, origin, -1);
        }

        insertPointRecurse(node, point);
    }

    bool Octree::convertFromPointCloud(std::vector<Point3f> &pointCloud)
    {
        // Find center coordinate of PointCloud data.
        Point3f center = cv::Octree::findCenterInPointCloud(pointCloud);

        float halfSize = std::max(center.x, std::max(center.y, center.z));
        this->origin = center - Point3f(halfSize, halfSize, halfSize);
        this->size = 2 * halfSize;

        // Insert every point in PointCloud data.
        for(size_t idx = 0; idx< pointCloud.size(); idx++ )
        {
            insertPoint(rootNode, pointCloud[idx]);
        }
        return true;
    }


    Point3f Octree::findCenterInPointCloud(std::vector<Point3f> &pointCloud)
    {
        Point3f maxBound(pointCloud[0]);
        Point3f minBound(pointCloud[0]);

        for(size_t idx = 0; idx <pointCloud.size(); idx++)
        {
            maxBound.x = max(pointCloud[idx].x, maxBound.x);
            maxBound.y = max(pointCloud[idx].y, maxBound.y);
            maxBound.z = max(pointCloud[idx].z, maxBound.z);

            minBound.x = min(pointCloud[idx].x, minBound.x);
            minBound.y = min(pointCloud[idx].y, minBound.y);
            minBound.z = min(pointCloud[idx].z, minBound.z);
        }
        return (maxBound+minBound)/2.0;
    }

    bool Octree::isPointInBound(const Point3f& _point, OctreeNode*& _node)
    {
        if((_point.x > _node->origin.x && _point.y > _node->origin.y && _point.z > _node->origin.z)
            && (_point.x < _node->origin.x + _node->size && _point.y < _node->origin.y + _node->size && _point.z < _node->origin.z + _node->size))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Octree::isPointInBound(const Point3f &_point) const
    {
        if((_point.x > origin.x && _point.y > origin.y && _point.z > origin.z) && (_point.x < origin.x + size && _point.y < origin.y + size && _point.z < origin.z + size))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Octree::isPointInBound(const Point3f &_point, Point3f &_origin, double _size)
    {
        if((_point.x > _origin.x && _point.y > _origin.y && _point.z > _origin.z) && (_point.x < _origin.x + _size && _point.y < _origin.y + _size && _point.z < _origin.z + _size))
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void Octree::clear()
    {
        if(rootNode != nullptr)
        {
            rootNode->clear();
        }

        size = 0;
        maxDepth = 0;
        origin = Point3f (0,0,0); // origin coordinate
    }

    bool Octree::isEmpty() const
    {
     return rootNode == nullptr;
    }

    void Octree::traverseRecurseBFS( OctreeNode*&node, const std::function<bool ( OctreeNode*&)>&f )
    {

        if(node == nullptr)
        {
            return;
        }
        else
        {
            for(size_t childIndex = 0; childIndex < Octree::childNum; ++childIndex)
            {
                traverseRecurseBFS(node->children[childIndex], f);
            }

            if(!f(node)) return;
        }
    }

    void Octree::traverseRecurseDFS( OctreeNode* &node, const std::function<bool ( OctreeNode*&)>&f )
    {

        if(node == nullptr)
        {
            return;
        }
        else
        {
            if(!f(node)) return;

            for(size_t childIndex = 0; childIndex < childNum; ++childIndex)
            {
                traverseRecurseDFS(node->children[childIndex], f);
            }
        }
    }

    OctreeNode* Octree::index(const Point3f& point)
    {
        if(isPointInBound(point))
        {
            return this->index(point, rootNode);
        }
        else
        {
            return nullptr;
        }
    }

    OctreeNode* Octree::index(const Point3f& point, OctreeNode*& node) const
    {

        if(node == nullptr)
        {
         return nullptr;
        }

        if(node->isLeaf)
        {
            for(int i = 0; i < node->pointList.size(); i++ )
            {
                if((point.x == node->pointList[i]->x) &&
                        (point.y == node->pointList[i]->y) &&
                        (point.z == node->pointList[i]->z)
                )
                {
                    return node;
                }
            }
            return nullptr;
        }

        if(this->isPointInBound(point, node->origin, node->size))
        {
            double childSize = node->size / 2.0;
            size_t xIndex = point.x < node->origin.x+ childSize ? 0 : 1;
            size_t yIndex = point.y < node->origin.y + childSize ? 0 : 1;
            size_t zIndex = point.z < node->origin.z + childSize ? 0 : 1;
            size_t childIndex = xIndex + yIndex * 2 + zIndex * 4;

            if(node->children[childIndex] != nullptr)
            {
                return this->index(point, node->children[childIndex]);
            }
        }
        return nullptr;
    }

    bool Octree::deletePoint(Point3f& point)
    {
        OctreeNode* node = index(point, rootNode);

        if(node != nullptr)
        {
            for(int i = 0; i < node->pointList.size(); i++)
            {
                if((point.x == node->pointList[i]->x) &&
                   (point.y == node->pointList[i]->y) &&
                   (point.z == node->pointList[i]->z)
                        )
                {
                    node->pointList.erase(node->pointList.begin() + i);
                }
            }
            return deletePointRecurse(node);
        }
        else
        {
            return false;
        }
    }


    bool Octree::deletePointRecurse(OctreeNode*& node)
    {
        if(node == nullptr)
            return false;
        if(node->isLeaf)
        {
            if( !node->pointList.empty())
            {
                OctreeNode* parent = node->parent;
                parent->children[node->parentIndex] = nullptr;
                delete node;

                return deletePointRecurse(parent);
            }
            else
            {
                return true;
            }
        }
        else
        {
            bool deleteFlag = true;

            for(int i = 0; i< childNum; i++) // only all children was deleted, can we delete the tree node.
            {
                if(node->children[i] != nullptr)
                {
                    deleteFlag = false;
                    break;
                }
            }

            if(deleteFlag)
            {
                OctreeNode* parent = node->parent;
                node = nullptr;
                return deletePointRecurse(parent);
            }
            else
            {
                return true;
            }
        }
    }

    void Octree::insertPointRecurse( OctreeNode*& node,  Point3f& point)
    {
        if(!isPointInBound(point, node->origin, node->size))
        {
            CV_Error(Error::StsBadArg, "The point is out of boundary!");
        }

        if(node->depth == maxDepth)
        {
            node->isLeaf = true;
            node->pointList.push_back(&point);
            return;
        }

        double childSize = node->size / 2.0;
        size_t xIndex = point.x < node->origin.x+ childSize ? 0 : 1;
        size_t yIndex = point.y < node->origin.y + childSize ? 0 : 1;
        size_t zIndex = point.z < node->origin.z + childSize ? 0 : 1;
        size_t childIndex = xIndex + yIndex * 2 + zIndex * 4;

        if(node->children[childIndex] == nullptr)
        {
            Point3f childOrigin = node->origin + Point3f(xIndex * childSize,yIndex * childSize, zIndex * childSize);
            node->children[childIndex] = new OctreeNode(node->depth + 1, childSize, childOrigin, childIndex);
            node->children[childIndex]->parent = node;
        }
        insertPointRecurse(node->children[childIndex], point);

    }
}