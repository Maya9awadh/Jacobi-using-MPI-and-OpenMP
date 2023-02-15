
//include libraries.
#include <iostream> //input output libraries.
#include "mpi.h" //MPI header .
#include <fstream> //file library.
#include <chrono> //library to calculate the time.

using namespace std;
using namespace std::chrono;//uesd for timing.

//constants.
#define MAX_SIZE 11000 //maximum size of array.
#define Copy(a,b) {double* temp; temp = a; a = b; b = temp;} //to copy the contents of vector to another vector.

//global variables.
double A[MAX_SIZE][MAX_SIZE];//the whole matrix A.

double local_A[MAX_SIZE][MAX_SIZE];//the scatred matrix A.

double b[MAX_SIZE];//the whole vactor b.

double local_b[MAX_SIZE];//the scatred vector b.

double local_x[MAX_SIZE];//values of x calculated by each processes.

double solution[MAX_SIZE];//the values of all x.
int rows;//number of rows or equations.

//time variables.
high_resolution_clock::time_point start;
high_resolution_clock::time_point stop;

/*
 This function reads a text file, and checks if the file is empty.
 It fills the matrix A, and the vector b with data from the file
 @param string file name
 return 1 if successful, 0 if the file is not found, 
 and 2 if there is wrong data.
*/
int Read_Data(string file_name);

/*
 Parallel Jacobi function to do parallel Jacobi algorithm via MPI
 @param int the rank of the process.
 @param double the epsilon value.
 @param int maximum number of iterations
 @parm int the number of process.
 return 1 if it converged.
 return 0 if it is not converged.
*/
int Parallel_Jacobi(int my_rank, double epsilon, int max_iter, int p);

int main()
{
	int p;//the number of processes, it will be declered in the command mpiexec -n p .
	int my_rank;//the rank of each process.
	int max_itr = 0;//the maximum iteration.
	double epsilon=0.0;//the tolerance value.
	int converged;//1 if converged, 0 if not.

	//intialize mpi.
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &p);//set the number of the processes.
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);//set the rank of each processes.

	//the rank 0 will read the data.
	if (my_rank == 0) {

		//read epsilon from the user.
		cout << "Enter epsilon: ";
		cin >> epsilon;

		cout << "\n";

		//read the maximum number of iteration from the user.
		cout << "Enter maximun iteration: ";
		cin >> max_itr;

		cout << "\n";

		//read matrix A and vector b from file
		string filename;//variable to store the file name.

		//read the file name from the user.
		cout << "Enter a input file's name: ";
		cin >> filename;

		cout << "\n";

		//read the file, and check for error.
		int read_data = Read_Data(filename);
		    
		if (read_data == 0) {
			cout << "The data file is not exit." << endl;
			exit(0);
		}

		else if (read_data == 2) {
			cout << "Wrong data in matrix." << endl;
			exit(0);
		}

		//convert the matrix A to diagonally dominant, in order to be solved by Jacobi.
		for (int i = 0; i < rows; i++) {
			double sum_raw = 0;
			for (int j = 0; j < rows; j++) {
				if (j != i) sum_raw += abs(A[i][j]);//the value in diagonal must 
				//be bigger than the sum of all values of row.
			}
			if (abs(A[i][i]) < sum_raw) {
				A[i][i] = sum_raw + 1;
			}
		}
	}

	//After reading the data.
	//rank 0, will brodcasts the max_itr,epsilon, and number of rows to all other processes.
	MPI_Bcast(&max_itr, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//scatter matrix A and vector b 
	//each processes will get rows/p rows of A and b.
	int num_rows = rows / p; 

	/* Fill dummy entries in A with zeroes */
	for (int i = 0; i < rows; i++)
		for (int j = rows; j < MAX_SIZE; j++)
			A[i][j] = 0.0;

	//rank 0 will scatter the matrix A and vector b to other processes.
	MPI_Scatter(A, num_rows * MAX_SIZE, MPI_DOUBLE, local_A, num_rows * MAX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(b, num_rows, MPI_DOUBLE, local_b, num_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//start to calculate the time.
	start = high_resolution_clock::now();

	//do parallel Jacobi iteration
	converged = Parallel_Jacobi(my_rank, epsilon, max_itr, p);

	//stop to calculate the time
	stop = high_resolution_clock::now();

	//the inverense of scatter is gather, all other process will gather the values of new_x in rank 0.
	MPI_Gather(local_x, num_rows, MPI_DOUBLE, solution, num_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//rank 0, will shows the result 
	if (my_rank == 0) {

		if (converged)//if the system converged.
			cout << "The solution is:  \n";

		else //the system is not conveged.
			cout << "Maximum iterations reached. The current vector is: " << "\n";

		//display the solution or  current vector of x.
		for (int i = 0;i < rows;i++)
			cout << solution[i] << " ";

		cout << "\n";
		cout << "\n";

		//display the execution time.
		auto execution_time = duration_cast<microseconds>(stop - start);
		cout << "The execution time is: " << execution_time.count() << "  microseconds \n";
	}

	//finalize mpi.
	MPI_Finalize();
}

/*
 Parallel Jacobi function to do parallel Jacobi algorithm via MPI
 @param int the rank of the process.
 @param double the epsilon value.
 @param int maximum number of iterations
 @parm int the number of process.
 return 1 if it converged.
 return 0 if it is not converged.
*/
int Parallel_Jacobi(int my_rank, double epsilon, int max_itr, int p) {

	//local variables
	int  num_rows;//number of rows on each process.
	int num_itr;//iteration number.

	double   temp1_x[MAX_SIZE]; //to store the values of new_x.
	double   temp2_x[MAX_SIZE]; //to store the values of old_x.

	double* old_x;
	double* new_x;

	/*  
	*function to calculate the distance between two vectors.
	* @param double vectore x[].
	* @param double vectore y[].
	* @param int n.
	* return the distance between two vectors.
	*/
	double Distance(double x[], double y[], int n);

	num_rows = rows / p;

	//intialze new_x to local_b.
	MPI_Allgather(local_b, num_rows, MPI_DOUBLE, temp1_x, num_rows, MPI_DOUBLE, MPI_COMM_WORLD);

	
	new_x = temp1_x;
	old_x = temp2_x;

	num_itr = 0;//intialize number of iteration to 0.

	//do Jacobi iteration.
	do {

		//increment the number of iterations.
		num_itr++;

		//interchange the old_x and new_x.
		Copy(old_x, new_x);

		//calculate the values of new x.
		for (int i = 0; i < num_rows; i++) {

			//each process will calculate the value of new x and store it to local_x.
			int k = i + my_rank * num_rows;
			local_x[i] = local_b[i];

			for (int j = 0; j < k; j++)
				
				local_x[i] = local_x[i] -local_A[i][j] * old_x[j];

			for (int j = k + 1; j < rows; j++)

				local_x[i] = local_x[i] -local_A[i][j] * old_x[j];

				local_x[i] = local_x[i] /local_A[i][k];
		}  	   

		//every process collects the values of new_x from all processes.
		MPI_Allgather(local_x, num_rows, MPI_DOUBLE, new_x, num_rows, MPI_DOUBLE, MPI_COMM_WORLD);

	} while ((num_itr < max_itr) &&(Distance(new_x, old_x, rows) >= epsilon));

	//check for convergence.
	if (Distance(new_x, old_x, rows) < epsilon)
		return 1;
	else
		return 0;
}

/*
	*function to calculate the distance between two vectors.
	* @param double vectore x[].
	* @param double vectore y[].
	* @param int n.
	* return the distance between two vectors.
	*/
double Distance(double x[], double y[], int n) {

	double sum = 0.0;
	double diff;
	for (int i = 0; i < n; i++) {
		diff = (x[i] - y[i]);
		sum = sum + diff * diff;
	}
	return sqrt(sum);
}

/*
 This function reads a text file, and checks if the file is empty.
 It fills the matrix A, and the vector b with data from the file
 @param string file name
 return 1 if successful, 0 if the file is not found,
 and 2 if there is wrong data.
*/

int Read_Data(string file_name) {
	fstream file;

	file.open(file_name);//open the input file

	if (!file) {//if file not exit.
		return 0;
	}

	//Reading Data from file (A,b, and rows).

	if (file.eof()) {
		cout << "The data file is empty." << endl;
		exit(0);
	}

	//read rows.
	if (file >> rows) {
		;
	}

	else {//error in number of rows or equations.
		cout << "Wrong data in number of rows or equation." << endl;
		exit(0);
	}

	//read the matrix A, and vector b.
	for (int i = 0;i < rows;i++) {
		for (int j = 0;j < rows;j++) {
			if (file >> A[i][j]) {
				continue;
			}
			else {
				return 2;
			}
		}
		if (file >> b[i]) {
			continue;
		}
	}
	file.close();
	return 1;
}
