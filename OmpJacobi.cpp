//include libraries.
#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <fstream>

//define the constants.
#define MAX_SIZE 11000
#define MAX_ITE 1000
#define EPSILON 0.0000001
#define NUM_OF_THREADS 8

// define variables
double A[MAX_SIZE][MAX_SIZE];
double b[MAX_SIZE];
int rows;

using namespace std;
using namespace std::chrono;// used for timing
/*
This function reads a text file, and checks if the file is
empty.
It fills the matrix A, and the vector b with data from the file
@param string file name
return 1 if successful, 0 if the file is not found,
and 2 if there is wrong data.*/

int Read_Data(string filename);

int main() {

	string filename;  //string variable to store file name

	//promp the user to enter file name
	cout << "Enter a input file's name: ";
	cin >> filename;

	//read the file and check for errors
	int read_data = Read_Data(filename);

	if (read_data == 0) {
		cout << "The data file is not exit." << endl;
		exit(0);
	}
	else if (read_data == 2) {
		cout << "Error in matrix data." << endl;
		exit(0);
	}
	//make the matrix diagonally dominant
	for (int i = 0; i < rows; i++) {
		double sum_raw = 0;
		for (int j = 0; j < rows; j++) {
			if (j != i) sum_raw += abs(A[i][j]);
		}
		if (abs(A[i][i]) < sum_raw) {
			A[i][i] = sum_raw + 1;
		}
	}
	//initialize new x and old x
	double new_x[MAX_SIZE];
	double old_x[MAX_SIZE];

	//store b vector values in new x vector
	for (int i = 0; i < rows; i++) {
		new_x[i] = b[i];
	}
	//start computing the time
	auto start = high_resolution_clock::now();

	//start Jacobi iterations
	for (int i = 0; i < MAX_ITE; i++) {

		//store values of nex x in old x
		for (int m = 0; m < rows; m++) {
			old_x[m] = new_x[m];
		}
		//set the number of threads
		omp_set_num_threads(NUM_OF_THREADS);

		//start the parallel region
#pragma omp parallel for schedule(dynamic, 1)

		//for loop to compute new x
		for (int j = 0; j < rows; j++) {
			double sigma_value = 0.0;
			for (int k = 0; k < rows; k++) {
				if (j != k) {
					sigma_value += A[j][k] * new_x[k];
				}
			}
			new_x[j] = (b[j] - sigma_value) / A[j][j];
		}
		//compute the difference
		double difference = 0.0;
		for (int n = 0; n < rows; n++) {
			difference = difference + pow(old_x[n] - new_x[n], 2);
		}
		difference = sqrt(difference);

		if (difference < EPSILON) break;

	}
	//stop computing the time
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	//display the solution
	std::cout << "The solution is: ";
	for (int l = 0; l < rows; l++) {
		std::cout << new_x[l];
		std::cout << "  ";
	}
	std::cout << "\n";
	std::cout << "\n";
	std::cout << "Time taken : " //display the time
		<< duration.count() << " microseconds" << endl;
}
int Read_Data(string file_name) {
	std::fstream file;

	file.open(file_name); //open the file

	//if file not found
	if (!file) {
		return 0;
	}

	// if the file is empty

	if (file.eof()) {
		cout << "The data file is empty." << endl;
		exit(0);
	}
	// if there is an error in the number of rows

	if (file >> rows) {
		;
	}
	else {
		cout << "Wrong data in number of rows or equation." << endl;
		exit(0);
	}
	// read the matrix A

	for (int i = 0;i < rows;i++) {
		for (int j = 0;j < rows;j++) {
			if (file >> A[i][j]) {
				continue;
			}
			else {
				return 2;
			}
		}
		if (file >> b[i]) { // read the vector b
			continue;
		}
	}
	file.close();// close the file
	return 1;
}