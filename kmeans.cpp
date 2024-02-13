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

    mDataset.resize(M,std::vector<float>(N));

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

float KMeans::euclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2)
{
    if(v1.size() != v2.size())
        return -1;

    float s = 0.0;

    for (size_t i = 0; i < v1.size(); ++i)
    {
        s += pow( v1[i] - v2[i], 2);
    }

    s = sqrt(s);
    return s;
}

int KMeans::findClosestCentroid(const std::vector<float>& v )
{
    int closestIndex = 0;
    float  minDistance = euclideanDistance(v, mCentroids[0]);
    for (size_t i = 1; i < mCentroids.size(); ++i)
    {
        float distance = euclideanDistance(v, mCentroids[i]);
        if (distance < minDistance)
        {
            minDistance = distance;
            closestIndex = i;
        }
    }
    return closestIndex;
}

void KMeans::updateCentroids( )
{
    std::vector<std::vector<float>> sums(mCentroids.size(), std::vector<float>(mDataset[0].size(), 0.0));
    std::vector<int> counts(mCentroids.size(), 0);

    for (size_t i = 0; i < mDataset.size(); ++i)
    {
        int cluster = mAssignments[i];
        for (size_t j = 0; j < mDataset[i].size(); ++j)
        {
            sums[cluster][j] += mDataset[i][j];
        }
        counts[cluster]++;
    }

    for (size_t i = 0; i < mCentroids.size(); ++i)
    {
        if (counts[i] != 0)
        {
            for (size_t j = 0; j < mCentroids[i].size(); ++j)
            {
                mCentroids[i][j] = sums[i][j] / counts[i];
            }
        }
    }
}

void KMeans::kMeansAlgo()
{
    mAssignments.resize(M, -1);
    bool converged = false;

    mCentroids.resize(K);

    for (int i = 0; i < K; ++i)
    {
        mCentroids[i] = mDataset[i];
    }

    while (!converged)
    {
        for (int i = 0; i < M; ++i)
        {
            mAssignments[i] = findClosestCentroid(mDataset[i]);
        }

        std::vector<std::vector<float>> prevCentroids = mCentroids;
        updateCentroids();

        double maxChange = 0.0;
        for (size_t i = 0; i < mCentroids.size(); ++i)
        {
            float change = euclideanDistance(prevCentroids[i], mCentroids[i]);
            if (change > maxChange)
            {
                maxChange = change;
            }
        }
        if (maxChange <= d)
        {
            converged = true;
        }
    }
}

void KMeans::printClusterCentroids( std::ostream& out )
{
    for (size_t i = 0; i < mCentroids.size(); ++i)
    {
        out << (i + 1) << " ";
        for (size_t j = 0; j < mCentroids[i].size(); ++j)
        {
            out << std::fixed << std::setprecision(6) << mCentroids[i][j] << " ";
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
            out << std::fixed << std::setprecision(6) << mDataset[i][j] << " ";
        }
        out << "\n";
    }
}
