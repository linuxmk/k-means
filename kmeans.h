#ifndef KMEANS_H
#define KMEANS_H
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <random>
#include <limits>

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

    std::vector<std::vector<double>>         mDataset;
    std::vector<std::vector<double>>         mCentroids;
    std::vector<int>                         mAssignments;

    double euclideanDistance(const std::vector<double>& v1, const std::vector<double>& v2);
    void kMeansPlusPlusInitCentroids();
    void assignPointsToCentroids();
    std::vector<std::vector<double>> updateCentroids();
};

#endif // KMEANS_H
