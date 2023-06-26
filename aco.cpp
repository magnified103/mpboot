#include "aco.h"
ACOAlgo::ACOAlgo() {}
void ACOAlgo::setUpParamsAndGraph(Params *params) {
    UPDATE_ITER = params->aco_update_iter;
    EVAPORATION_RATE = params->aco_evaporation_rate;
    int RATCHET_PRIOR = params->aco_ratchet_prior;
    int IQP_PRIOR = params->aco_iqp_prior;
    int RANDOM_NNI_PRIOR = params->aco_random_nni_prior;
    int NNI_PRIOR = params->aco_nni_prior;
    int SPR_PRIOR = params->aco_spr_prior;
    int TBR_PRIOR = params->aco_tbr_prior;

    addNode(ROOT);
    addNode(RATCHET);
    addNode(IQP);
    addNode(RANDOM_NNI);
    addNode(NNI);
    addNode(SPR);
    addNode(TBR);

    addEdge(0, 1, RATCHET_PRIOR);
    addEdge(0, 2, IQP_PRIOR);
    addEdge(0, 3, RANDOM_NNI_PRIOR);

    for (int i = 1; i <= 3; ++i) {
        addEdge(i, 4, NNI_PRIOR);
        addEdge(i, 5, SPR_PRIOR);
        addEdge(i, 6, TBR_PRIOR);
    }
    curIter = 0;
    curNode = 0;
}
void ACOAlgo::addNode(NodeTag tag) {
    nodes.push_back(ACONode(tag));
    par.push_back(0);
}

void ACOAlgo::addEdge(int from, int to, double prior) {
    int edgeId = ACOAlgo::edges.size();
    edges.push_back(ACOEdge(from, to, prior));
    nodes[from].adj.push_back(edgeId);
}

void ACOAlgo::registerTime() { curTime = clock(); }

double ACOAlgo::getTimeFromLastRegistered() {
    clock_t nowTime = clock();
    double elapsed_time = ((double)(nowTime - curTime)) / CLOCKS_PER_SEC;
    return elapsed_time;
}

int ACOAlgo::moveNextNode() {
    double sum = 0;
    int u = curNode;
    for (int i = 0; i < nodes[u].adj.size(); ++i) {
        int E = nodes[u].adj[i];
        double prob = edges[E].pheromone * edges[E].prior;
        sum += prob;
    }
    double random = random_double() * sum;
    sum = 0;
    for (int i = 0; i < nodes[u].adj.size(); ++i) {
        int E = nodes[u].adj[i];
        double prob = edges[E].pheromone * edges[E].prior;
        sum += prob;
        if (random < sum || i == nodes[u].adj.size() - 1) {
            curNode = edges[E].toNode;
            par[curNode] = E;
            return curNode;
        }
    }
    assert(0);
    return 0;
}

void ACOAlgo::updateNewPheromonePath(vector<int> &edgesOnPath) {
    vector<bool> isOnPath(edges.size(), false);
    for (int E : edgesOnPath) {
        isOnPath[E] = true;
    }
    for (int i = 0; i < edges.size(); ++i) {
        edges[i].updateNewPhero(isOnPath[i], EVAPORATION_RATE);
    }
}

void ACOAlgo::updateNewPheromone(int diffMP) {
    double t = getTimeFromLastRegistered();
    vector<int> edgesOnPath;
    int u = curNode;
    while (u) {
        int E = par[u];
        edgesOnPath.push_back(E);
        u = edges[E].fromNode;
    }
    if (diffMP > 0) {
        updateNewPheromonePath(edgesOnPath);
    } else {
        // Didn't improve the tree
        // -> Save the paths, get the fastest ones to update
        savedPath.push_back({t, edgesOnPath});
    }
    curNode = 0;
    curIter++;
    if (curIter == UPDATE_ITER) {
        applyNewPheromone();
        curIter = 0;
    }
}

void ACOAlgo::applyNewPheromone() {
    // Get the paths that is fastest
    sort(savedPath.begin(), savedPath.end(),
         [&](const pair<double, vector<int>> &A,
             const pair<double, vector<int>> &B) { return A.first < B.first; });
    // If there are less than half of UPDATE_ITER paths that have diffMP > 0,
    // Update using savedPath until there are half of paths updated
    for (int i = 0;
         i < min((int)savedPath.size(),
                 UPDATE_ITER / 2 - (UPDATE_ITER - (int)savedPath.size()));
         ++i) {
        updateNewPheromonePath(savedPath[i].second);
    }
    savedPath.clear();
    for (auto E : edges) {
        E.pheromone = E.newPheromone;
    }
}
