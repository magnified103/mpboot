/*
 * helper.h
 * Contains classes for different purposes
 */

#ifndef HELPER_H_
#define HELPER_H_
#include "iqtree.h"
#include "pllrepo/src/pll.h"
#include "tools.h"
#include <sstream>

class DuplicatedTreeEval {
  private:
    const int MOD = 1e9 + 7;
    const int MOD1 = 1e9 + 123;
    const int BASE = 317;
    const int BASE1 = 371;
    int numDup = 0;
    int mxTips = 0;
    vector<int> mnLeaf;
    map<pair<int, int>, int> mp;
    /**
     * Get the Newick tree string format with sorted leaves id.
     */
    string getTreeString(IQTree *iqtree, pllInstance *tr, partitionList *pr) {

        pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back, PLL_TRUE,
                        PLL_TRUE, 0, 0, 0, PLL_SUMMARIZE_LH, 0, 0);
        string treeString = string(tr->tree_string);
        iqtree->readTreeString(treeString);
        iqtree->initializeAllPartialPars();
        iqtree->clearAllPartialLH();
        stringstream out;
        int options = WT_TAXON_ID | WT_SORT_TAXA;
        iqtree->printTree(out, options);
        return out.str();
    }

    pair<int, int> getHash(string treeString) {
        int hash = 0;
        for (int i = 0; i < treeString.size(); ++i) {
            hash = (hash * 1LL * BASE + treeString[i]) % MOD;
        }
        int hash1 = 0;
        for (int i = 0; i < treeString.size(); ++i) {
            hash1 = (hash1 * 1LL * BASE1 + treeString[i]) % MOD1;
        }
        return {hash, hash1};
    }

    int minLeaf(nodeptr p) {
        if (p->number <= mxTips) {
            return mnLeaf[p->number] = p->number;
        }
        return mnLeaf[p->number] =
                   min(minLeaf(p->next->back), minLeaf(p->next->next->back));
    }
    void addHash(int &hash, int &hash1, int v) {
        hash = (hash * 1LL * BASE + v) % MOD;
        hash1 = (hash1 * 1LL * BASE1 + v) % MOD1;
    }
    void dfs(nodeptr p, int &hash, int &hash1) {
        if (p->number <= mxTips) {
            addHash(hash, hash1, p->number);
            return;
        }
        nodeptr q = p->next->back;
        nodeptr r = p->next->next->back;
        if (mnLeaf[q->number] > mnLeaf[r->number]) {
            swap(q, r);
        }

        addHash(hash, hash1, mnLeaf[p->number]);
        dfs(q, hash, hash1);
        addHash(hash, hash1, mnLeaf[p->number]);
        dfs(r, hash, hash1);
        addHash(hash, hash1, mnLeaf[p->number]);
    }
    pair<int, int> getHashDfs(nodeptr p) {
        minLeaf(p);
        int hash = 0, hash1 = 0;
        dfs(p, hash, hash1);
        return {hash, hash1};
    }

  public:
    DuplicatedTreeEval(int mxTips) {
        this->mxTips = mxTips;
        mnLeaf.assign(mxTips * 2, 0);
    }
    bool keepOptimizeTree(nodeptr p) {
        pair<int, int> hash = getHashDfs(p);
        int numDuplicate = ++mp[hash];
        if (numDuplicate % 1000 == 0) {
            cout << "Num duplicates: " << numDuplicate << '\n';
        }
        return numDuplicate == 1;
        // return random_double() <= 1.0 / numDuplicate;
    }

    bool duplicate(IQTree *iqtree, pllInstance *tr, partitionList *pr) {
        string treeString = getTreeString(iqtree, tr, pr);
        pair<int, int> hash = getHash(treeString);
        if (mp.find(hash) != mp.end()) {
            numDup++;
            cout << "Duplicated tree: " << numDup << '\n';
            return true;
        }
        mp[hash]++;
        return false;
    }
};


class AntColonyAlgo {
public:
    int numNNI, numSPR, numTBR;
    int totNNI, totSPR, totTBR;
    const int MAX_ITER = 100;
    AntColonyAlgo() {
        numNNI = numSPR = numTBR = 0;
        totNNI = totSPR = totTBR = 0;
    }
    /**
     * 0: NNI
     * 1: SPR
     * 2: TBR
     */
    int getMoveType(int curIt) {
        if (curIt <= MAX_ITER) {
            int tem1 = random_int(3);
            if (tem1 == 0) {
                totNNI++;
            } else if (tem1 == 1) {
                totSPR++;
            } else {
                totTBR++;
            }
            return tem1;
        } 
        int tem = random_int(numNNI + numSPR + numTBR);
        if (tem < numNNI) {
            totNNI++;
            return 0;
        } else if (tem < numNNI + numSPR) {
            totSPR++;
            return 1;
        }
        totTBR++;
        return 2;
    }

    void update(int moveType, bool better) {
        if (better) {
            if (moveType == 0) {
                numNNI++;
            } else if (moveType == 1) {
                numSPR += 3;
            } else {
                numTBR += 3;
            }
        } else {
            if (moveType == 0) {
                numSPR += 3;
                numTBR += 3;
            } else if (moveType == 1) {
                numNNI += 1;
                numTBR += 3;
            } else {
                numNNI += 1;
                numSPR += 3;
            }
        }
        // cout << "totNNI: " << totNNI << '\n';
        // cout << "totSPR: " << totSPR << '\n';
        // cout << "totTBR: " << totTBR << '\n';
    }
};
#endif /* HELPER_H_ */
