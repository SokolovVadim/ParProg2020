#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

void calc(double* frame, uint32_t ySize, uint32_t xSize, double delta, int rank, int size)
{
  if (rank == 0 && size > 0)
  {
    double diff = 0;
    double* tmpFrame = new double[ySize * xSize];
    // Prepare tmpFrame
    for (uint32_t y = 0; y < ySize; y++)
    {
      tmpFrame[y*xSize] = frame[y*xSize];
      tmpFrame[y*xSize + xSize - 1] = frame[y*xSize + xSize - 1];
    }
    for (uint32_t x = 1; x < xSize - 1; x++)
    {
      tmpFrame[x] = frame[x];
      tmpFrame[(ySize - 1)*xSize + x] = frame[(ySize - 1)*xSize + x];
    }
    // Calculate first iteration
    for (uint32_t y = 1; y < ySize - 1; y++)
    {
      for (uint32_t x = 1; x < xSize - 1; x++)
      {
        tmpFrame[y*xSize + x] = (frame[(y + 1)*xSize + x] + frame[(y - 1)*xSize + x] +\
                                frame[y*xSize + x + 1] + frame[y*xSize + x - 1])/4.0;
        diff += std::abs(tmpFrame[y*xSize + x] - frame[y*xSize + x]);
      }
    }

    double* currFrame = tmpFrame;
    double* nextFrame = frame;
    uint32_t iteration = 1;
    // Calculate frames
    while (diff > delta)
    {
      diff = 0;
      for (uint32_t y = 1; y < ySize - 1; y++)
      {
        for (uint32_t x = 1; x < xSize - 1; x++)
        {
          nextFrame[y*xSize + x] = (currFrame[(y + 1)*xSize + x] + currFrame[(y - 1)*xSize + x] +\
                                  currFrame[y*xSize + x + 1] + currFrame[y*xSize + x - 1])/4.0;
          diff += std::abs(nextFrame[y*xSize + x] - currFrame[y*xSize + x]);
        }
      }
      std::swap(currFrame, nextFrame);
      iteration++;
    }

    // Copy result from tmp
    if (iteration % 2 == 1)
    {
      for (uint32_t i = 0; i < xSize*ySize; i++)
      {
        frame[i] = tmpFrame[i];
      }
    }
    delete tmpFrame;
  }
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  double delta = 0;
  uint32_t ySize = 0, xSize = 0;
  double* frame = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize >> delta;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    frame = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> frame[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(frame, ySize, xSize, delta, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete frame;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << frame[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete frame;
  }

  MPI_Finalize();
  return 0;
}
