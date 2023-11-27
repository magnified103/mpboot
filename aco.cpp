#include "aco.h"
#include <iomanip>
#include <sstream>
ACOAlgo::ACOAlgo() {}
void ACOAlgo::setUpParamsAndGraph(Params *params) {
    // UPDATE_ITER = params->aco_update_iter;
    UPDATE_ITER =
        params->aco_update_iter + (int)params->unsuccess_iteration * 2 / 100;
    EVAPORATION_RATE = params->aco_evaporation_rate;
    double NNI_PRIOR = params->aco_nni_prior;
    double SPR_PRIOR = params->aco_spr_prior;
    double TBR_PRIOR = params->aco_tbr_prior;
    cout << "ACO Params: \n";
    cout << "UPDATE_ITER = " << UPDATE_ITER << '\n';
    cout << "EVAPORATION_RATE = " << EVAPORATION_RATE << '\n';
    cout << "NNI_PRIOR = " << NNI_PRIOR << '\n';
    cout << "SPR_PRIOR = " << SPR_PRIOR << '\n';
    cout << "TBR_PRIOR = " << TBR_PRIOR << '\n';

    addNode(ROOT);
    addNode(NNI);
    addNode(SPR);
    addNode(TBR);

    addEdge(ROOT, NNI, NNI_PRIOR);
    addEdge(ROOT, SPR, SPR_PRIOR);
    addEdge(ROOT, TBR, TBR_PRIOR);

    curIter = 0;
    curNode = ROOT;
    curCounter = 0;
    foundBetterScore = false;

    isOnPath.assign(edges.size(), false);
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

void ACOAlgo::registerCounter() { lastCounter = curCounter; }

long long ACOAlgo::getNumCounters() { return curCounter - lastCounter; }

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
            nodes[curNode].cnt++;
            return curNode;
        }
    }
    assert(0);
    return 0;
}

void ACOAlgo::updateNewPheromone(int oldScore, int newScore) {
    // numCounters measures how long the chosen hill-climbing procedure ran
    long long numCounters = getNumCounters();
    vector<int> edgesOnPath;
    int u = curNode;
    while (u) {
        int E = par[u];
        edgesOnPath.push_back(E);
        u = edges[E].fromNode;
    }
    if (newScore < curBestScore) {
        // cout << "P0\n";
        for (int i = 0; i < edges.size(); ++i) {
            isOnPath[i] = false;
        }
        for (int E : edgesOnPath) {
            isOnPath[E] = true;
            edges[E].updateNewPhero(true, EVAPORATION_RATE);
        }
        for (int i = 0; i < edges.size(); ++i) {
            if (!isOnPath[i]) {
                edges[i].updateNewPhero(false, EVAPORATION_RATE);
            }
            isOnPath[i] = false;
        }
        savedPath.clear();
        curIter = 0;
        curBestScore = newScore;
        foundBetterScore = true;
        reportPheroPercentage();
    } else if (foundBetterScore && newScore == curBestScore) {
        // cout << "P1\n";
        for (int E : edgesOnPath) {
            isOnPath[E] = true;
            edges[E].updateNewPhero(true, EVAPORATION_RATE);
        }
        // } else if (oldScore - newScore >= newScore - curBestScore) {
        //     cout << "P2\n";
        //     for (int E : edgesOnPath) {
        //         isOnPath[E] = true;
        //     }
    } else {
        // cout << "P3\n";
        savedPath.push_back({numCounters, edgesOnPath});
    }
    curNode = ROOT;
    curIter++;
    if (curIter == UPDATE_ITER) {
        applyNewPheromone();
        curIter = 0;
    }
}

void ACOAlgo::applyNewPheromone() {
    // Get the paths that is fastest
    sort(savedPath.begin(), savedPath.end(),
         [&](const pair<long long, vector<int>> &A,
             const pair<long long, vector<int>> &B) {
             return A.first < B.first;
         });
    // If there are less than half of UPDATE_ITER paths that have diffMP > 0,
    // Update using savedPath until there are half of paths updated
    // cout << "foundBetterScore = " << foundBetterScore << '\n';
    for (int i = 0;
         i < min((int)savedPath.size(),
                 UPDATE_ITER / 2 - (UPDATE_ITER - (int)savedPath.size()));
         ++i) {
        if (foundBetterScore) {
            for (int E : savedPath[i].second) {
                isOnPath[E] = true;
                // edges[E].updateNewPhero(true, EVAPORATION_RATE);
            }
        } else {
            for (int E : savedPath[i].second) {
                isOnPath[E] = true;
                edges[E].updateNewPhero(true, EVAPORATION_RATE);
            }
        }
    }
    savedPath.clear();
    for (int i = 0; i < edges.size(); ++i) {
        edges[i].updateNewPhero(isOnPath[i], EVAPORATION_RATE);
        isOnPath[i] = false;
    }
    reportPheroPercentage();
}

void ACOAlgo::reportUsage() {
    for (int i = 1; i < (int)nodes.size(); ++i) {
        cout << nodeTagToString(getNodeTag(i)) << " : " << nodes[i].cnt << '\n';
    }
}

void ACOAlgo::incCounter() { curCounter++; }

void ACOAlgo::reportPheroPercentage() {
    double p_nni = edges[0].pheromone;
    double p_spr = edges[1].pheromone;
    double p_tbr = edges[2].pheromone;

    double sum = p_nni + p_spr + p_tbr;
    p_nni /= sum;
    p_spr /= sum;
    p_tbr /= sum;
    ostringstream tem;
    tem << "%Phero:\n";
    tem << fixed << setprecision(3);
    tem << "PER_NNI = " << p_nni << '\n';
    tem << "PER_SPR = " << p_spr << '\n';
    tem << "PER_TBR = " << p_tbr << '\n';
    string temStr = tem.str();
    cout << temStr;
}

int ACOAlgo::getNumStopCond(int unsuccess_iters) {
    double p_nni = edges[0].pheromone;
    double p_spr = edges[1].pheromone;
    double p_tbr = edges[2].pheromone;

    double sum = p_nni + p_spr + p_tbr;
    p_nni /= sum;
    p_spr /= sum;
    p_tbr /= sum;
    return int(p_nni * unsuccess_iters + p_spr * unsuccess_iters + p_tbr * 100);
}
