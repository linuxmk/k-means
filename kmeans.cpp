#include "kmeans.h"

KMeans::KMeans(const std::string &filename)
{
    if(readFromFile(filename) == -1 )
        throw std::runtime_error("Can not read from " + filename + " file ");
}

int KMeans::readFromFile( const std::string& filename)
{
    std::ifstream in(filename);
    if(!in)
        return -1;

    in >> M >> N >> K >> d;

//    std::cerr << "M = " << M << " N = " << N << " K = " << K << " d = " << d << "\n";

    mDataset.resize(M,std::vector<double>(N));

    for(int i = 0 ; i < M ; ++i)
    {
        for(int j = 0 ; j < N ; ++j)
        {
            in >> mDataset[i][j];
        }
    }

    in.close();
    return 0;
}

double KMeans::euclideanDistance(const std::vector<double>& v1, const std::vector<double>& v2)
{
    if(v1.size() != v2.size())
        return -1;

    double s = 0.0;

    for (size_t i = 0; i < v1.size(); ++i)
    {
        s += pow( v1[i] - v2[i], 2);
    }

    s = sqrt(s);
    return s;
}



// Function to perform K-means++ initialization for centroids
void KMeans::kMeansPlusPlusInitCentroids()
{
    // Select the first centroid
    mCentroids.push_back(mDataset[0]);

    // Select the rest of the centroids using K-means++ algorithm
    for (int k = 1; k < K; ++k)
    {
        std::vector<double> distSquared(mDataset.size(), std::numeric_limits<double>::max());
        double totalDistSquared = 0.0;
        for (size_t i = 0; i < mDataset.size(); ++i)
        {
            double minDist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < mCentroids.size(); ++j)
            {
                double dist = euclideanDistance(mDataset[i], mCentroids[j]);
                minDist = std::min(minDist, dist);
            }
            distSquared[i] = minDist * minDist;
            totalDistSquared += distSquared[i];
        }

        // Select next centroid with probability proportional to distance squared
        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<double> uniform(0, totalDistSquared);
        double r = uniform(gen);
        double sum = 0.0;
        for (size_t i = 0; i < mDataset.size(); ++i)
        {
            sum += distSquared[i];
            if (sum >= r)
            {
                mCentroids.push_back(mDataset[i]);
                break;
            }
        }
    }
}

// Function to assign each data point to the nearest centroid
void KMeans::assignPointsToCentroids()
{
    for (size_t i = 0; i < mDataset.size(); ++i)
    {
        double minDist = std::numeric_limits<double>::max();
        int minIdx = -1;
        for (size_t j = 0; j < mCentroids.size(); ++j)
        {
            double dist = euclideanDistance(mDataset[i], mCentroids[j]);
            if (dist < minDist)
            {
                minDist = dist;
                minIdx = j;
            }
        }
        mAssignments[i] = minIdx;
    }
}

// Function to compute new centroids based on current assignments
std::vector<std::vector<double> > KMeans::updateCentroids()
{
    std::vector<std::vector<double>> sums(K, std::vector<double>(mDataset[0].size(), 0.0));
    std::vector<int> counts(K, 0);

    for (size_t i = 0; i < mDataset.size(); ++i)
    {
        int cluster = mAssignments[i];
        for (size_t j = 0; j < mDataset[i].size(); ++j)
        {
            sums[cluster][j] += mDataset[i][j];
        }
        counts[cluster]++;
    }

    std::vector<std::vector<double>> newCentroids(K, std::vector<double>(mDataset[0].size(), 0.0));

    for (int i = 0; i < K; ++i)
    {
        for (size_t j = 0; j < mDataset[0].size(); ++j)
        {
            if (counts[i] > 0)
            {
                newCentroids[i][j] = round(sums[i][j] / counts[i]);
            }
        }
    }

    return newCentroids;
}

void KMeans::kMeansAlgo()
{
    bool converged = false;

    mAssignments.resize(M, -1);

    // Different initialization: Use K-means++ to initialize centroids
    kMeansPlusPlusInitCentroids();

    while (!converged)
    {
        // Assign data points to nearest centroids
        assignPointsToCentroids();

        // Compute new centroids based on current assignments
        std::vector<std::vector<double>> newCentroids = updateCentroids();

        // Check for convergence
        double maxChange = 0.0;
        for (int i = 0; i < K; ++i)
        {
            double dist = euclideanDistance(mCentroids[i], newCentroids[i]);
            maxChange = std::max(maxChange, dist);
        }
        if (maxChange <= d) {
            converged = true;
        }

        // Update centroids
        mCentroids = newCentroids;
    }
}

void KMeans::printClusterCentroids( std::ostream& out )
{
    for (size_t i = 0; i < mCentroids.size(); ++i)
    {
        out << (i + 1) << " ";
        for (size_t j = 0; j < mCentroids[i].size(); ++j)
        {
            out << std::fixed << std::setprecision(2) << mCentroids[i][j] << " ";
        }
        out << "\n";
    }
    out << "\n";
}

void KMeans::printDataset(std::ostream &out)
{
    for (int i = 0; i < M; ++i)
    {
        out << (mAssignments[i] + 1) << " ";
        for (int j = 0; j < N; ++j)
        {
            out << std::fixed << std::setprecision(2) << mDataset[i][j] << " ";
        }
        out << "\n";
    }
}
