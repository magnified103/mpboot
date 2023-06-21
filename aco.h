/*
 * aco.h
 * Ant Colony Optimization
 */

#ifndef ACO_H_
#define ACO_H_
#include "iqtree.h"
#include "pllrepo/src/pll.h"
#include "tools.h"
#include <ctime>
#include <vector>

class ACOAlgo {
  public:
    enum NodeTag {
        ROOT,
        RATCHET,
        IQP,
        RANDOM_NNI,
        NNI,
        SPR,
        TBR,
    };
    struct ACONode {
        int tag;
        vector<int> adj;
        ACONode(NodeTag _tag) { tag = _tag; }
    };
    struct ACOEdge {
        int fromNode;
        int toNode;
        double pheromone;
        double prior;
        double deltaPhero;
        ACOEdge(int _fromNode, int _toNode, double _prior) {
            fromNode = _fromNode;
            toNode = _toNode;
            prior = _prior;
            pheromone = 1;
            deltaPhero = 0;
        }
        void updatePheroDelta(int diffMP, double t) {
            // diffMP / time (minutes)
            deltaPhero += 60 * diffMP / t;
        }
    };

    const int MAX_ITER = 100;
    const int UPDATE_ITER = 30;
    const double EVAPORATION_RATE = 0.6;
    int curNode;
    int curIter;
    clock_t curTime;
    vector<int> par;
    vector<ACONode> nodes;
    vector<ACOEdge> edges;
    ACOAlgo();
    void setUpACOGraph();
    int moveNextNode();
    void addNode(NodeTag tag);
    void addEdge(int from, int to, double prior);
    void updatePheromoneDelta(int diffMP);
    void updatePheromone();
    void registerTime();
    double getTimeFromLastRegistered();
};
#endif /* ACO_H_ */
