//
// C++ Implementation: phylonode
//
// Description: 
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "phylonode.h"


void PhyloNeighbor::clearForwardPartialLh(Node *dad) {
	clearPartialLh();
	for (NeighborVec::iterator it = node->neighbors.begin(); it != node->neighbors.end(); it ++)
		if ((*it)->node != dad)
			((PhyloNeighbor*)*it)->clearForwardPartialLh(node);
}


void PhyloNode::clearReversePartialLh(PhyloNode *dad) {
	PhyloNeighbor *node_nei = (PhyloNeighbor*)findNeighbor(dad);
	assert(node_nei);
	node_nei->partial_lh_computed = 0;
	for (NeighborVec::iterator it = neighbors.begin(); it != neighbors.end(); it ++)
		if ((*it)->node != dad)
			((PhyloNode*)(*it)->node)->clearReversePartialLh(this);
}

void PhyloNode::clearAllPartialLh(PhyloNode *dad) {
	PhyloNeighbor *node_nei = (PhyloNeighbor*)findNeighbor(dad);
	node_nei->partial_lh_computed = 0;
	node_nei = (PhyloNeighbor*)dad->findNeighbor(this);
	node_nei->partial_lh_computed = 0;
	for (NeighborVec::iterator it = neighbors.begin(); it != neighbors.end(); it ++)
		if ((*it)->node != dad)
			((PhyloNode*)(*it)->node)->clearAllPartialLh(this);
}


PhyloNode::PhyloNode()
 : Node()
{
	init();
}


PhyloNode::PhyloNode(int aid) : Node(aid)
{
	init();
}

PhyloNode::PhyloNode(int aid, int aname) : Node (aid, aname) {
	init();
}


PhyloNode::PhyloNode(int aid, const char *aname) : Node(aid, aname) {
	init();
}

void PhyloNode::init() {
	//partial_lh = NULL;
}


void PhyloNode::addNeighbor(Node *node, double length, int id) {
	neighbors.push_back(new PhyloNeighbor(node, length, id));
}
