// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2021, Huawei Technologies Co., Ltd. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Zihao Mu <zihaomu6@gmail.com>
//         Liangqian Kong <chargerKong@126.com>
//         Longbu Wang <riskiest@gmail.com>

#ifndef OPENCV_OCTREE_OCTREE_H
#define OPENCV_OCTREE_OCTREE_H

#include <vector>
#include "opencv2/core.hpp"

namespace cv {
//! @addtogroup 3d
//! @{

    /** @brief OctreeNode for Octree.

    The class OctreeNode represents the node of the octree. Each node contains 8 children, which are used to divide the
    space cube into eight parts. Each octree node represents a cube.
    And these eight children will have a fixed order, the order is described as follows:

    For illustration, assume,
        rootNode: origin == (0, 0, 0), size == 2
     Then,
        children[0]: origin == (0, 0, 0), size == 1
        children[1]: origin == (1, 0, 0), size == 1, along X-axis next to child 0
        children[2]: origin == (0, 1, 0), size == 1, along Y-axis next to child 0
        children[3]: origin == (1, 1, 0), size == 1, in X-Y plane
        children[4]: origin == (0, 0, 1), size == 1, along Z-axis next to child 0
        children[5]: origin == (1, 0, 1), size == 1, in X-Z plane
        children[6]: origin == (0, 1, 1), size == 1, in Y-Z plane
        children[7]: origin == (1, 1, 1), size == 1, furthest from child 0

    There are two kinds of nodes in an octree, intermediate nodes and leaf nodes, which are distinguished by isLeaf.
    Intermediate nodes are used to contain leaf nodes, and leaf nodes will contain pointers to all pointcloud data
    within the node, which will be used for octree indexing and mapping from point clouds to octree. Note that,
    in an octree, each leaf node contains at least one point cloud data. Similarly, every intermediate OctreeNode
    contains at least one non-empty child pointer, except for the root node.
    */
    class CV_EXPORTS OctreeNode{
    public:

        /**
         * There are multiple constructors to create OctreeNode.
         * */
        OctreeNode():children(8, nullptr), depth(0), size(0), origin(0,0,0), parentIndex(-1){}


        /** @overload
         *
         * @param _depth The depth of the current node. The depth of the root node is 0, and the leaf node is equal
         * to the depth of Octree.
         * @param _size The length of the OctreeNode. In space, every OctreeNode represents a cube.
         * @param _origin The absolute coordinates of the center of the cube.
         * @param _parentIndex The serial number of the child of the current node in the parent node,
         * the range is (-1~7). Among them, only the root node's _parentIndex is -1.
         */
        OctreeNode(int _depth, double _size, Point3f _origin, int _parentIndex);

        //! destructor - calls release()
        ~OctreeNode()= default;

        /** @brief clear the OctreeNode and its children.
         * This function will delete the current node and all child nodes. And set the pointer to itself
         * in its parent node to NULL.
         */
        void clear();

        //! Contains 8 pointers to its 8 children.
        std::vector<OctreeNode *> children;

        //! Point to the parent node of the current node. The root node has no parent node and the value is NULL.
        OctreeNode* parent{};

        /** @brief The serial number of the child of the current node in the parent node,
         * the range is (-1~7). Among them, only the root node's _parentIndex is -1.
         */
        int parentIndex;

        //! Each OctreeNode will contain eight child nodes.
        const static int childNum = 8;

        //! The depth of the current node. The depth of the root node is 0, and the leaf node is equal to the depth of Octree.
        int depth;

        //! The length of the OctreeNode. In space, every OctreeNode represents a cube.
        double size;

        //! The absolute coordinates of the center of the cube.
        Point3f origin;

        //! If the OctreeNode is LeafNode.
        bool isLeaf = false;

        //! Contains pointers to all point cloud data in this node.
        std::vector<Point3f *> pointList;
    };


    /** @brief Octree for 3D vision.
   In 3D vision filed, the Octree is used to process and accelerate the pointcloud data. The class Octree represents
   the Octree data structure. Each Octree will have a fixed depth. The depth of Octree refers to the distance from
   the root node to the leaf node.All OctreeNodes will not exceed this depth.Increasing the depth will increase
   the amount of calculation exponentially. And the small number of depth refers low resolution of Octree.
   */
    class CV_EXPORTS Octree{

    public:

        //! Default constructor.
        Octree():maxDepth(0), size(0), origin(0,0,0){}

        /** @overload
         * @brief Create an empty Octree and set the maximum depth.
         * @param The max depth of the Octree.
         */
        explicit Octree(int _maxDepth);

        /** @overload
         *  @brief Create an Octree from the PointCloud data with the specific max depth.
         * @param _maxDepth The max depth of the Octree.
         * @param _pointCloud Point cloud data.
         */
        Octree(int _maxDepth, std::vector<Point3f>& _pointCloud);

        /** @overload
         * @brief Deep copy a new tree with the same structure.
         * @param src Source Octree
         */
        Octree(Octree& src);

        /** @overload
         * @brief Create an empty Octree.
         * @param _maxDepth Max depth.
         * @param _size Initial Cube size.
         * @param _origin Initial center coordinate.
         */
        Octree(int _maxDepth, double _size, Point3f _origin);

        //! destructor - calls release()
        ~Octree()= default;;


        /** @brief Insert a point data to a OctreeNode.
         *
         * @param node A pointer to a specific OctreeNode
         * @param point The point data in Point3f format.
         */
        void insertPoint(OctreeNode*& node, Point3f& point);


        /** @brief Read point cloud data and create OctreeNode.
         * This function is only called when the octree is being created.
         * @param pointCloud PointCloud data.
         * @return Returns whether the creation is successful.
         */
        bool convertFromPointCloud(std::vector<Point3f> &pointCloud);

        /** @brief
         *
         * @param pointCloud
         * @return The coordinate of center point of the PointCloud.
         */
        static Point3f findCenterInPointCloud(std::vector<Point3f> &pointCloud) ;

        /** @brief Determine whether the point is within the space range of the specific cube.
         *
         * @param point The point coordinates.
         * @param origin The coordinate of cube.
         * @param size The size of cube.
         * @return If point is in bound, return ture. Otherwise, false.
         */
        static bool isPointInBound(const Point3f& point, Point3f& origin, double size) ;

        /** @overload
         * @brief Determine whether the point is within the space range of the OctreeNode.
         * @param point The point coordinates.
         * @param node The pointer to OctreeNode
         * @return If point is in bound, return ture. Otherwise, false.
         */
        static bool isPointInBound(const Point3f &point, OctreeNode*& node) ;

        /** @overload
         * @brief  @brief Determine whether the point is within the space range of the octree.
         * @param point The point coordinates.
         * @return If point is in bound, return ture. Otherwise, false.
         */
        bool isPointInBound(const Point3f &point) const;

        //! returns true if the rootnode is NULL.
        bool isEmpty() const;

        /** @brief
         *  Reset all octree parameterDeleting a point from the octree actually deletes the corresponding element
         *  from the pointList in the corresponding leaf node. If the leaf node does not contain other points after
         *  deletion, this node will be deleted. In the same way, its parent node may also be deleted if its
         *  last child is deleted. and delete all its OctreeNode.
         */
        void clear();

        /** @brief Locate the OctreeNode corresponding to the input point.
         *
         * @param point The point to be located.
         * @param node OctreeNode.
         * @return The pointer to the located OctreeNode.
         */
        OctreeNode* index(const Point3f& point, OctreeNode*& node) const;

        /** @overload
         *  @brief The default search range is in the entire tree.
         * @param point
         * @return The pointer to the located OctreeNode.
         */
        OctreeNode* index(const Point3f& point);

        /** @brief Delete a given point from the Octree.
         * Delete the corresponding element from the pointList in the corresponding leaf node. If the leaf node
         * does not contain other points after deletion, this node will be deleted. In the same way,
         * its parent node may also be deleted if its last child is deleted.
         * @param point The point coordinates.
         * @return return ture if the point is deleted successfully.
         */
        bool deletePoint(Point3f& point);

        /** @brief Traverse OctreeNode in BFS.
         *
         * @param node When node is the rootNode, traverse the entire Octree.
         * @param f The operations on the node.
         */
        void traverseRecurseBFS( OctreeNode*& node, const std::function<bool ( OctreeNode*&)>&f );

        //! Traverse OctreeNode in DFS.
        void traverseRecurseDFS( OctreeNode*& node, const std::function<bool ( OctreeNode*&)>&f );

        //! The pointer to Octree root node.
        OctreeNode* rootNode = nullptr;

        //!
        const static int childNum = 8;

        //! The size of the cube of the .
        double size;

        //! Max depth of the Octree.
        int maxDepth;

        //! The origin coordinate of root node.
        Point3f origin;

    private:

        /** @brief Insert node recursively.
         * If the OctreeNode to be inserted does not exist, a new OctreeNode is created. If it exists,
         * add the point information to the pointList of the corresponding leaf node.
         * @param node
         * @param point
         */
        void insertPointRecurse( OctreeNode*& node, Point3f& point);

        /** @brief Delete a node from tree.
         * Used to delete the node. If its parent node may also be deleted if its last child is deleted.
         * @param node OctreeNode to be deleted.
         * @return return ture if the deletion is completed.
         */
        bool deletePointRecurse( OctreeNode*& node);

    };
//! @} 3d
}

#endif //OPENCV_OCTREE_OCTREE_H
