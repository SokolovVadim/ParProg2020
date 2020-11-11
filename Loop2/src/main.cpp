#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

enum { OFFSET = 7 };

void calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
  uint32_t token_size = (ySize * xSize) / size;
  int rest_size = xSize * ySize % size;

  int token_offset = rank * token_size;
  int token_offset_end = (rank + 1) * token_size;

  if((rest_size != 0) && (rank == size - 1)) // the rest of data
  {
    token_size += rest_size;
  }
  if(rank != size - 1)
  {
    token_size += OFFSET;
  }

  for(int i(0); i < token_size; ++i)
  {
    int j = token_offset + i;
    if((j % xSize > 2) && (j % xSize < xSize) && (j < xSize * ySize - xSize))
    {
      arr[i] = sin(0.00001 * arr[i + OFFSET]);
    }
  }
}


void print_arr(uint32_t size, double* arr)
{
  std::string msg{};
  for(uint32_t i(0); i < size; ++i)
  {
    msg += std::to_string(arr[i]) + " ";
  }
  msg += "\n";
  std::cout << msg;
}

void send_data_from_root(uint32_t ySize, uint32_t xSize, int size, double* arr)
{
  int token_size = xSize * ySize / size;
  int rest_size = (xSize * ySize) % size;
  if(rest_size == 0) // scalable
  {
    for(int i(1); i < size - 1; ++i)
    {
      double* data = arr + (i * token_size);
      MPI_Send(data, token_size + OFFSET, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    if(size > 1)
    {
      double* data = arr + ((size - 1) * token_size);
      MPI_Send(data, token_size, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
    }
  }
  else
  {
    for(int i(1); i < size - 1; ++i)
    {
      double* data = arr + (i * token_size);
      MPI_Send(data, token_size + OFFSET, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    double* data = arr + ((size - 1) * token_size);
    MPI_Send(data, token_size + rest_size, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
  }
}

double* recv_data_from_root(int rank, uint32_t ySize, uint32_t xSize, int size, double* data)
{
  int token_size = xSize * ySize / size;
  int rest_size = xSize * ySize % size;
 
  if((rank == size - 1) && (rest_size != 0))
  {
    data = new double[token_size + rest_size];
    MPI_Recv(data, token_size + rest_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  else
  {
    if(rank == size - 1)
    {
      data = new double[token_size];
      MPI_Recv(data, token_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else
    {
      data = new double[token_size + OFFSET];
      MPI_Recv(data, token_size + OFFSET, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  return data;
}

void send_data_from_process(int rank, int ySize, int xSize, int size, double* data)
{
  int token_size = xSize * ySize / size;
  int rest_size = (xSize * ySize) % size;
  if((rank == size - 1) && (rest_size != 0))
  {
    MPI_Send(data, token_size + rest_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  else // scalable
  {
    MPI_Send(data, token_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void recv_data_from_process(int ySize, int xSize, int size, double* arr)
{
  int token_size = xSize * ySize / size;
  int rest_size = xSize * ySize % size;
  if(rest_size == 0) // scalable
  {
    for(int i(1); i < size; ++i)
    {
      double* data = arr + i * token_size;
      MPI_Recv(data, token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  else
  {
    for(int i(1); i < size - 1; ++i)
    {
      double* data = arr + i * token_size;
      MPI_Recv(data, token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    double* data = arr + (size - 1) * token_size;
    MPI_Recv(data, token_size + rest_size, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t ySize = 0, xSize = 0;
  double* arr = 0;
  double* data = nullptr;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    

    arr = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> arr[y*xSize + x];
      }
    }
    input.close();
    
    send_data_from_root(xSize, ySize, size, arr);
    data = arr;
    // MPI_Bcast(arr, xSize * ySize, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
    
    MPI_Bcast(&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /*double* data = nullptr;*/
    data = recv_data_from_root(rank, xSize, ySize, size, data);
    
  }

  calc(data, ySize, xSize, rank, size);

  if (rank == 0)
  {
    recv_data_from_process(ySize, xSize, size, data);
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << arr[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }
  else
  {
    send_data_from_process(rank, xSize, ySize, size, data);
  }
  // delete arr;

  MPI_Finalize();
  return 0;
}
