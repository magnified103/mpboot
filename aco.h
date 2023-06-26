/*
 * aco.h
 * Ant Colony Optimization
 */

#ifndef ACO_H_
#define ACO_H_
#include "iqtree.h"
#include "pllrepo/src/pll.h"
#include "tools.h"

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

        NodeTag tag;
        vector<int> adj;
        ACONode(NodeTag _tag) { tag = _tag; }
    };
    struct ACOEdge {
        int fromNode;
        int toNode;
        double pheromone;
        double prior;
        double newPheromone;
        ACOEdge(int _fromNode, int _toNode, double _prior) {
            fromNode = _fromNode;
            toNode = _toNode;
            prior = _prior;
            pheromone = PHERO_MAX;
            newPheromone = pheromone;
        }
        void updateNewPhero(bool isOnPath, double EVAPORATION_RATE,
                            double c = 1.0) {
            // Smooth Max-Min Ant System (Huan, Linh-Trung, et al. 2012).
            // Default:           c = 1.0
            // Modified (TODO):   c = newScore / bestScore
            // (the better the score, the more it converges to PHERO_MAX)
            if (isOnPath) {
                newPheromone = (1 - EVAPORATION_RATE) * newPheromone +
                               EVAPORATION_RATE * PHERO_MAX * c;
            } else {
                newPheromone = (1 - EVAPORATION_RATE) * newPheromone +
                               EVAPORATION_RATE * PHERO_MIN;
            }
        }
    };

    // TODO: Try UPDATE_ITER 10% of num stop conditions
    int UPDATE_ITER = 20;
    double EVAPORATION_RATE = 0.5;
    static constexpr double PHERO_MAX = 1;
    static constexpr double PHERO_MIN = 0.001;
    int curNode;
    int curIter;
    clock_t curTime;
    vector<int> par;
    vector<ACONode> nodes;
    vector<ACOEdge> edges;
    vector<pair<double, vector<int>>> savedPath;
    ACOAlgo();
    void setUpParamsAndGraph(Params *params);
    int moveNextNode();
    void addNode(NodeTag tag);
    void addEdge(int from, int to, double prior);
    void updateNewPheromonePath(vector<int> &edgesOnPath);
    void updateNewPheromone(int diffMP);
    void applyNewPheromone();
    void registerTime();
    double getTimeFromLastRegistered();

    NodeTag getNodeTag(int u) { return nodes[u].tag; }
};
#endif /* ACO_H_ */
