#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

double acceleration(double t)
{
    return sin(t);
}

void print_arr(uint32_t size, double* arr);

void calc(double* trace, double v0, uint32_t traceSize, double t0, double dt, double y0, int rank, int size)
{
    int token_size = traceSize / size;
    int rest_size = traceSize % size;
    if((rest_size != 0) && (rank == size - 1)) // last process
    {
        token_size += rest_size;
    }
    
    trace[0] = y0;
    trace[1] = y0 + dt * v0;

    for (uint32_t i = 2; i < uint32_t(token_size); i++)
    {
        trace[i] = dt*dt*acceleration(t0 + (rank * token_size + i - 1)*dt) + 2*trace[i - 1] - trace[i - 2];
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

enum { END_DATA_SIZE = 2 };

struct Node
{
    double a;
    double v;
    double A;
    double V;
    double u;
    double left_border;
    double right_border;
    int token_size;
};

void send_end_data_to_root(double* trace, int traceSize, int rank, int size)
{
    int token_size = traceSize / size;
    int rest_size = traceSize % size;
    std::cout << "token_size = " << token_size << std::endl;
    if((rest_size != 0) && (rank == size - 1)) // last process
    {
        token_size += rest_size;
    }
    double* data = new double[END_DATA_SIZE];
    data[0] = trace[token_size - 2];
    data[1] = trace[token_size - 1];
    MPI_Send(data, END_DATA_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    delete[] data;
}

void recv_end_data_in_root(Node* tokens, int size)
{
    /*int token_size = traceSize / size;
    int rest_size = traceSize % size;*/
    double* data = new double[END_DATA_SIZE];
    //double end_data[END_DATA_SIZE * size];
    for(int i(1); i < size; ++i)
    {
        // double* ptr = end_data + (i - 1) * END_DATA_SIZE;
        MPI_Recv(data, END_DATA_SIZE, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        tokens[i].left_border  = data[0];
        tokens[i].right_border = data[1];
    }
    delete[] data;
}

double calculate_last_point(Node* tokens, double dt, int traceSize, int size)
{
    int token_size = traceSize / size;
    int rest_size = traceSize % size;
    for(int i(0); i < size - 1; ++i)
    {
        tokens[i].token_size = token_size;
    }
    if(rest_size != 0) // last process
    {
        tokens[size - 1].token_size = token_size + rest_size;
    }

    for(int i(0); i < size; ++i)
    {
        tokens[i].u = (tokens[i].right_border - tokens[i].left_border) / dt;
    }
    for(int i(1); i < size; ++i)
    {
        tokens[i].v = tokens[i - 1].u - tokens[i - 1].v;
    }
    for(int i(1); i < size; ++i)
    {
        tokens[i].a = tokens[i - 1].right_border + tokens[i - 1].a + tokens[i - 1].v * (tokens[i - 1].token_size * dt);
    }
    double last_point = tokens[size - 1].right_border + tokens[size - 1].a + tokens[size - 1].v * (tokens[size - 1].token_size * dt);
    return last_point;
}

double calculate_initial_speed(double last_point, double y1, double t0, double t1)
{
    double initial_speed = (y1 - last_point) / (t1 - t0);
    return initial_speed;
}

/*void calculate_speed_and_trace(Node* tokens, double dt, int size)
{
    for(int i(1); i < size; ++i)
    {
        tokens[i].V = tokens[i - 1].u + tokens[i - 1].V;
        tokens[i].A = tokens[i - 1].right_border + tokens[i - 1].A + tokens[i - 1].V * (tokens[i - 1].token_size * dt);
    } 
}*/

void consider_initial_speed(double* trace, int traceSize, double initial_speed, double dt, int rank, int size)
{
    int token_size = traceSize / size;
    int rest_size = traceSize % size;

    for(int i(0); i < token_size + rest_size; ++i)
    {
        // y[i] += i * v0*dt;
        trace[i] += (i + rank * token_size) * initial_speed * dt;
    }
}

void send_result_to_root(double* trace, int traceSize, int rank, int size)
{
    int token_size = traceSize / size;
    int rest_size = traceSize % size;
    if((rank == size - 1) && (rest_size != 0))
    {
      MPI_Send(trace, token_size + rest_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else // scalable
    {
      MPI_Send(trace, token_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void write_result(double* trace, int traceSize, char* filename, int size)
{
    int token_size = traceSize / size;
    int rest_size = traceSize % size;

    std::ofstream output(filename);
    if (!output.is_open())
    {
        std::cout << "[Error] Can't open " << filename << " for read\n";
        delete[] trace;
        return;
    }

    for (int i = 0; i < token_size; i++)
    {
        output << " " << trace[i];
    }

    if(rest_size != 0)
    {
        for(int i(1); i < size - 1; ++i)
        {
            MPI_Recv(trace, token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j(0); j < token_size; ++j)
                output << " " << trace[j];
        }
        double* last_token = new double[token_size + rest_size];
        MPI_Recv(last_token, token_size + rest_size, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int j(0); j < token_size + rest_size; ++j)
            output << " " << trace[j];
        delete[] last_token;
    }
    else
    {
        for(int i(1); i < size; ++i)
        {
            MPI_Recv(trace, token_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j(0); j < token_size; ++j)
                output << " " << trace[j];
        }
    }

    output << std::endl;
    output.close();
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = nullptr;

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
      input >> t0 >> t1 >> dt >> y0 >> y1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      traceSize = (t1 - t0)/dt;
      trace = new double[traceSize / size];
      input.close();

      MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&t1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&traceSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  } else {
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (status != 0)
      {
          return 1;
      }

      MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&t1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&y1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&traceSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      int token_size = traceSize / size;
      int rest_size = traceSize % size;
      if((rest_size != 0) && (rank == size - 1)) // last process
      {
          token_size += rest_size;
      }
      trace = new double[token_size];
  }

  calc(trace, 0, traceSize, t0, dt, y0, rank, size);

  if (rank == 0)
  {
      Node* tokens = new Node[size];
      tokens[0].token_size   = traceSize / size;
      tokens[0].left_border  = trace[0];
      tokens[0].right_border = trace[(traceSize / size) - 1];
      tokens[0].v = 0;
      tokens[0].a = y0;

      recv_end_data_in_root(tokens, size);
      double last_point = calculate_last_point(tokens, dt, traceSize, size);
      double initial_speed = calculate_initial_speed(last_point, y1, t0, t1);
      std::cout << "initial_speed = " << initial_speed << std::endl;

      // send_initial_speed_to_process
      MPI_Bcast(&initial_speed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      /*recv_results();*/
      // tokens[0].V = initial_speed;
      // calculate_speed_and_trace(tokens, dt, size);


      write_result(trace, traceSize, argv[2], size);
      delete[] trace;
      delete[] tokens;
  }
  else
  {

      send_end_data_to_root(trace, traceSize, rank, size);
      double initial_speed(0.0);
      MPI_Bcast(&initial_speed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      consider_initial_speed(trace, traceSize, initial_speed, dt, rank, size);
      send_result_to_root(trace, traceSize, rank, size);
  }

  MPI_Finalize();
  return 0;
}
