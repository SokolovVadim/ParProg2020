#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cstring>

enum { OFFSET = 4 };

void calculate_token_size(int& token_size, int& rest_size, int xSize, int size);
void print_arr(uint32_t size, double* arr);

void copy_arr_into_data(double* arr, double* data, uint32_t xSize, int token_size)
{
  for(int i(0); i < token_size; ++i)
  {
    for(int j(0); j < int(OFFSET); ++j)
    {
      data[i * (xSize - OFFSET) + j] = arr[i * OFFSET + j];
    }
  }
}

double* calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
  int token_size(0);
  int rest_size(0);
  calculate_token_size(token_size, rest_size, xSize, size);
  if((rest_size != 0) && (rank == size - 1)) // the rest of data
  {
    token_size += rest_size;
  }
  double* data = new double[(xSize - OFFSET) * token_size];
  copy_arr_into_data(arr, data, xSize, token_size);
  if(rank == 0)
  {
    for(int x(0); x < token_size; ++x)
    {
      for (int y = OFFSET; y < int(ySize); y++)
      {
        arr[y*xSize + x] = sin(arr[(y - OFFSET)*xSize + x]);
      }
    }
    
  }
  else
  {
    if((rest_size != 0) && (rank == size - 1)) // the rest of data
    {
      token_size += rest_size;
    }
    // calculate data array
    for(int i(0); i < token_size; ++i)
    {
      for(int j(0); j < int(xSize - OFFSET); ++j)
      {
        if(j < OFFSET)
        {
          /*std::cout << "i = " << i << " j = " << j << " data[" << (i * (xSize - OFFSET) + j) << "] = sin(arr[" <<
          (i * (xSize - OFFSET) + j) << "] = " << data[i * (xSize - OFFSET) + j] << ")";*/
          data[i * (xSize - OFFSET) + j] = sin(data[i * (xSize - OFFSET) + j]);
          // std::cout << " = " << data[i * (xSize - OFFSET) + j] << std::endl;
        }
        else // j >= OFFSET 
        {
          /*std::cout << "i = " << i << " j = " << j << " data[" << (i * (xSize - OFFSET) + j) << "] = sin(arr[" <<
          (i * (xSize - OFFSET) + j - OFFSET) << "] = " << data[i * (xSize - OFFSET) + j - OFFSET] << ")";  */
          data[i * (xSize - OFFSET) + j] = sin(data[i * (xSize - OFFSET) + j - OFFSET]);
          // std::cout << " = " << data[i * (xSize - OFFSET) + j] << std::endl;
        }
      }
    }
  }
  return data;
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

void calculate_token_size(int& token_size, int& rest_size, int xSize, int size)
{
  token_size = xSize / size;
  rest_size  = xSize % size;
}

void send_data_from_root(uint32_t xSize, int size, double* arr)
{
  int token_size(0);
  int rest_size(0);
  calculate_token_size(token_size, rest_size, xSize, size);
  double* data = nullptr;
  if(rest_size == 0) // scalable
  {
    data = new double[OFFSET * token_size];
    for(int i(1); i < size; ++i)
    {
      int didx(0);
      for (int x = i * token_size; x < (i + 1) * token_size; x++)
      {
        for (int y = OFFSET; y < OFFSET + OFFSET; y++)
        {
          // arr[y*xSize + x] = sin(arr[(y - OFFSET)*xSize + x]);
          data[didx] = arr[(y - OFFSET)*xSize + x];
          didx++;
        }
      }
      // print_arr(OFFSET * token_size, data);
      MPI_Send(data, OFFSET * token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      memset(data, 0.0, token_size * sizeof(*data));
    }
  }
  else
  {
    data = new double[OFFSET * (token_size + rest_size)];
    for(int i(1); i < size - 1; ++i)
    {
      int bidx(0);
      for (int x = i * token_size; x < (i + 1) * token_size; x++)
      {
        for (int y = OFFSET; y < OFFSET + OFFSET; y++)
        {
          // arr[y*xSize + x] = sin(arr[(y - OFFSET)*xSize + x]);
          data[bidx] = arr[(y - OFFSET)*xSize + x];
          bidx++;
        }
      }
      MPI_Send(data, OFFSET * token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      memset(data, 0.0, (token_size) * sizeof(*data));
    }
    int bidx = 0;
    for (int x = (size - 1) * token_size; x < int(xSize); x++)
    {
      for (int y = OFFSET; y < OFFSET + OFFSET; y++)
      {
        // arr[y*xSize + x] = sin(arr[(y - OFFSET)*xSize + x]);
        data[bidx] = arr[(y - OFFSET)*xSize + x];
        bidx++;
      }
    }
    MPI_Send(data, OFFSET * (token_size + rest_size), MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
  }
}

double* recv_data_from_root(int rank, uint32_t xSize, int size)
{
  int token_size(0);
  int rest_size(0);
  double* data(nullptr);
  calculate_token_size(token_size, rest_size, xSize, size);
 
  if((rank == size - 1) && (rest_size != 0))
  {
    data = new double[OFFSET * (token_size + rest_size)];
    MPI_Recv(data, OFFSET * (token_size + rest_size), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  else
  {
    data = new double[OFFSET * token_size];
    MPI_Recv(data, OFFSET * token_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // print_arr(OFFSET * token_size, data);
  }
  return data;
}

void send_data_from_process(int rank, int xSize, int size, double* data)
{
  int token_size(0);
  int rest_size(0);
  calculate_token_size(token_size, rest_size, xSize, size);
  if((rank == size - 1) && (rest_size != 0))
  {
    MPI_Send(data, (xSize - OFFSET) * (token_size + rest_size), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  else // scalable
  {
    /*print_arr(OFFSET * token_size, data);*/
    MPI_Send(data, (xSize - OFFSET) * token_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}

void recv_data_from_process(int xSize, int ySize, int size, double* arr)
{
  int token_size(0);
  int rest_size(0);
  calculate_token_size(token_size, rest_size, xSize, size);
  double* data = nullptr;
  if(rest_size == 0) // scalable
  {
    for(int i(1); i < size; ++i)
    {
      data = new double[(xSize - OFFSET) * token_size];// + i * token_size;
      MPI_Recv(data, (xSize - OFFSET) * token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // print_arr((xSize - OFFSET) * token_size, data);
      int idx(0);
      for(int x(i * token_size); x < (i + 1) * token_size; ++x)
      {
        for (int y = OFFSET; y < int(ySize); y++)
        {
          // std::cout << "i = " << i << " x = " << x << " y = " << y << " arr[" << y*xSize + x << "] = data[" << idx << "]\n";
          arr[y*xSize + x] = data[idx];
          idx++;
        }
      }      
    }
  }
  else
  {
    for(int i(1); i < size - 1; ++i)
    {
      data = new double[(xSize - OFFSET) * token_size];
      MPI_Recv(data, (xSize - OFFSET) * token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int idx(0);
      for(int x(i * token_size); x < (i + 1) * token_size; ++x)
      {
        for (int y = OFFSET; y < int(ySize); y++)
        {
          arr[y*xSize + x] = data[idx];
          idx++;
        }
      }  
    }
    data = new double[(xSize - OFFSET) * (token_size + rest_size)];
    MPI_Recv(data, (xSize - OFFSET) * (token_size + rest_size), MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // print_arr((xSize - OFFSET) * token_size, data);
    int idx(0);
    for(int x((size - 1) * token_size); x < xSize; ++x)
    {
      for (int y = OFFSET; y < int(ySize); y++)
      {
        arr[y*xSize + x] = data[idx];
        idx++;
      }
    }  
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

    send_data_from_root(xSize, size, arr);
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
    data = recv_data_from_root(rank, xSize, size);
    
  }

  double* ret_calc = calc(data, ySize, xSize, rank, size);

  if (rank == 0)
  {
    recv_data_from_process(xSize, ySize, size, data);
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
    data = ret_calc;
    send_data_from_process(rank, xSize, size, data);
  }
  // delete arr;

  MPI_Finalize();
  return 0;
}
