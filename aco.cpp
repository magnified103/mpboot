#include "aco.h"
ACOAlgo::ACOAlgo() {
    setUpACOGraph();
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

void ACOAlgo::setUpACOGraph() {
    addNode(ROOT);
    addNode(RATCHET);
    addNode(IQP);
    addNode(RANDOM_NNI);
    addNode(NNI);
    addNode(SPR);
    addNode(TBR);

    addEdge(0, 1, 2);
    addEdge(0, 2, 1);
    addEdge(0, 3, 2);

    for (int i = 1; i <= 3; ++i) {
        addEdge(i, 4, 0.1);
        addEdge(i, 5, 0.2);
        addEdge(i, 6, 0.3);
    }
    curIter = 0;
    curNode = 0;
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
    return 0;
}

void ACOAlgo::updatePheromoneDelta(int diffMP) {
    double t = getTimeFromLastRegistered();
    int u = curNode;
    while (u) {
        int E = par[u];
        edges[E].updatePheroDelta(diffMP, t);
        u = edges[E].fromNode;
    }
    curNode = 0;
    curIter++;
    if (curIter == UPDATE_ITER) {
        updatePheromone();
        curIter = 0;
    }
}

void ACOAlgo::updatePheromone() {
    for (auto E : edges) {
        E.pheromone = (1 - EVAPORATION_RATE) * E.pheromone + E.deltaPhero;
        E.deltaPhero = 0;
    }
}
