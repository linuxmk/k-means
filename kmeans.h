#ifndef KMEANS_H
#define KMEANS_H
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>

class KMeans
{
public:
    KMeans(const std::string& filename);
    int readFromFile( const std::string& filename);
    void kMeansAlgo();
    void printClusterCentroids( std::ostream &out = std::cout);
    void printDataset( std::ostream &out = std::cout );

private:
    int M;
    int N;
    int K;
    float d;

    std::vector<std::vector<float>>         mDataset;
    std::vector<std::vector<float>>         mCentroids;
    std::vector<int>                        mAssignments;

    float euclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2);
    int findClosestCentroid(const std::vector<float>& v );
    void updateCentroids();
};

#endif // KMEANS_H
