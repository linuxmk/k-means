#include "kmeans.h"


int main()
{

    try
    {
        KMeans k("input.txt");

        k.kMeansAlgo();

//        std::ofstream out("output.txt");

//        k.printClusterCentroids(out);

//        k.printDataset(out);

        k.printClusterCentroids();

        k.printDataset();

    }
    catch(const std::runtime_error &e)
    {
        std::cerr << e.what() << "\n";
    }

    return 0;
}
