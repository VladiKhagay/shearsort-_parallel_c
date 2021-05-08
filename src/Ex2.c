/*
 ============================================================================
 Name        : Ex2.c
 Authors     : Vladi Khagay 319654497
 	 	 	   Asaf Hen 308024280
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include "exr2.h"


//---------- Main ----------//
int main(int argc, char *argv[]) {
	int my_rank; /* rank of process */
	int numOfProccs; /* number of processes */

	int size; /* */
	int order; /* */
	int coords[DIMS]; /* */

	Cuboid *cubes;
	Cuboid myCuboid; /* */

	MPI_Comm cartesianComm; /* */
	MPI_Datatype cuboidType;

	FILE *fp;

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProccs);

	/* create cuboid type*/
	createCuboidType(&cuboidType);

	cartesianComm = createCartesianComm(my_rank, numOfProccs, coords);

	if (my_rank == MASTER) {
		/* open file pointer*/
		fp = fopen(DAT_FILE_PATH, "r");
		if (fp == NULL) {
			printf("file open failed\n");
			return -1;
		}

		/* read the size and the order*/
		fscanf(fp, "%d", &size);
		fscanf(fp, "%d", &order);

		if (numOfProccs != size) {
			printf("\nInvalid number of processes: %d instead of %d\n",
					numOfProccs, size);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		cubes = (Cuboid*) malloc(sizeof(Cuboid) * size); /* Cuboids array to store the file content*/

		if (!cubes) {
			printf("malloc failed\n");
			return -1;
		}

		/* Reads the data of the cuboids from the file to the cuboids array */
		readFile(cubes, fp, size);

		/* close the file reader*/
		fclose(fp);
	}

	MPI_Bcast(&size, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&order, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

	MPI_Scatter(cubes, 1, cuboidType, &myCuboid, 1, cuboidType, MASTER,
			cartesianComm);

	shearSort(cartesianComm, size, order, &myCuboid, cuboidType);

	MPI_Gather(&myCuboid, 1, cuboidType, cubes, 1, cuboidType, MASTER,
			cartesianComm);

	if (my_rank == MASTER) {

		storeToFile(cubes, size, fp);
	}
	/* shut down MPI */
	MPI_Finalize();

	return 0;
}

//---------- Functions : ------------//

/**
 * Function to create a new communicator of Cartesian topology
 *
 */
MPI_Comm createCartesianComm(int rank, int numOfProcesses, int *coords) {

	MPI_Comm cartesianComm;
	int ndim = DIMS;
	int dim[DIMS], period[DIMS], reorder;
	int n = (int) sqrt(numOfProcesses);
	dim[0] = n;
	dim[1] = n;
	period[0] = 0;
	period[1] = 0;
	reorder = 1;

	MPI_Cart_create(MPI_COMM_WORLD, ndim, dim, period, reorder, &cartesianComm);
	MPI_Cart_coords(cartesianComm, rank, ndim, coords);

	return cartesianComm;

}

/**
 * Function to create a new MPI type for structure Cuboid
 */
void createCuboidType(MPI_Datatype *cuboidType) {

	Cuboid cuboid;
	MPI_Datatype type[4] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[CUBOID_NUM_ATTR] = { 1, 1, 1, 1 }; // id, width, height, length, one integer and three doubles
	MPI_Aint disp[CUBOID_NUM_ATTR];

	// Create MPI user data type for cuboid
	disp[0] = (char*) &cuboid.id - (char*) &cuboid;
	disp[1] = (char*) &cuboid.length - (char*) &cuboid;
	disp[2] = (char*) &cuboid.width - (char*) &cuboid;
	disp[3] = (char*) &cuboid.height - (char*) &cuboid;

	MPI_Type_create_struct(CUBOID_NUM_ATTR, blocklen, disp, type, cuboidType);
	MPI_Type_commit(cuboidType);

}

/**
 * Function that performs the  shear sort
 */
void shearSort(MPI_Comm comm, const int size, const int order, Cuboid *myCuboid,
		MPI_Datatype cuboidType) {

	int rowSize = (int) sqrt(size);

	int iterations = (int) ceil(log2((double) size)) + 1;

	for (int iter = 0; iter < iterations; iter++) {
		if (iter % 2 == 0) {
			oddEvenSort(comm, ROW_DIR, rowSize, order, myCuboid, cuboidType);
		} else {
			oddEvenSort(comm, COL_DIR, rowSize, order, myCuboid, cuboidType);
		}
	}
}
/**
 *  odd even sort
 */
void oddEvenSort(MPI_Comm comm, const int direction, const int rowSize,
		const int order, Cuboid *myCuboid, MPI_Datatype cuboidType) {

	int myRank, source, dest;
	int coords[DIMS];

	MPI_Status status;
	MPI_Comm_rank(comm, &myRank);
	MPI_Cart_coords(comm, myRank, DIMS, coords);

	if (direction == COL_DIR) { // sort rows

		MPI_Cart_shift(comm, ROW_DIR, DISP, &source, &dest);

		for (int i = 0; i < rowSize; i++) {

			exchangeInRow(coords, i, source, dest, order, myCuboid, cuboidType,
					comm, status);

		}
	} else { // sort columns

		MPI_Cart_shift(comm, COL_DIR, DISP, &source, &dest);

		for (int i = 0; i < rowSize; i++) {
			exchangeInColumn(coords, i, source, dest, order, myCuboid,
					cuboidType, comm, status);
		}

	}

}
/**
 * Function that performs the exchange between two neighboring processes in the same row
 */
void exchangeInRow(int *coords, int i, int source, int dest, int order,
		Cuboid *myCuboid, MPI_Datatype cuboidType, MPI_Comm comm,
		MPI_Status status) {

	Cuboid recveivedCuboid;

	if (i % 2 == coords[ROW_DIR] % 2) { // sending side

		if (dest != -1) {
			MPI_Send(myCuboid, 1, cuboidType, dest, 0, comm);

			MPI_Recv(myCuboid, 1, cuboidType, dest, 0, comm, &status);
		}

	} else { // Receiving side
		if (source != -1) {
			MPI_Recv(&recveivedCuboid, 1, cuboidType, source, 0, comm, &status);

			if (coords[COL_DIR] % 2 == 0) { // ascending

				swapByOrder(&recveivedCuboid, myCuboid, order, ROW_DIR, ASC);

			} else { // descending

				swapByOrder(&recveivedCuboid, myCuboid, order, ROW_DIR, DESC);

			}

			MPI_Send(&recveivedCuboid, 1, cuboidType, source, 0, comm);
		}
	}

}

/**
 * Function that performs the exchange between two neighboring processes in the same column
 */
void exchangeInColumn(int *coords, int i, int source, int dest, int order,
		Cuboid *myCuboid, MPI_Datatype cuboidType, MPI_Comm comm,
		MPI_Status status) {

	Cuboid recveivedCuboid;
	if (i % 2 == coords[COL_DIR] % 2) {

		if (dest != -1) {

			MPI_Send(myCuboid, 1, cuboidType, dest, 0, comm);

			MPI_Recv(myCuboid, 1, cuboidType, dest, 0, comm, &status);
		}
	} else {

		if (source != -1) {

			MPI_Recv(&recveivedCuboid, 1, cuboidType, source, 0, comm, &status);

			// descending
			swapByOrder(&recveivedCuboid, myCuboid, order, COL_DIR, ASC);

			MPI_Send(&recveivedCuboid, 1, cuboidType, source, 0, comm);
		}

	}

}


/**
 * Function that swaps between Cuboids in a row or in a column by the order that received.
 * 1 - ascending order
 * 0 - descending order
 *
 * direction argument indicates whether a row or a column
 */
void swapByOrder(Cuboid *recveivedCuboid, Cuboid *myCuboid, int order,
		int direction, int asc) {

	if (direction == ROW_DIR) {

		if (asc == ASC) { // row in ascending order

			if (order == 1) {
				if (getSurfaceArea(recveivedCuboid)
						> getSurfaceArea(myCuboid)) {
					swap(recveivedCuboid, myCuboid);
				} else if (getSurfaceArea(recveivedCuboid)
						== getSurfaceArea(myCuboid)) {
					if (recveivedCuboid->width > myCuboid->width) {
						swap(recveivedCuboid, myCuboid);
					}
				}
			} else { // swap order direction
				if (getSurfaceArea(recveivedCuboid)
						< getSurfaceArea(myCuboid)) {
					swap(recveivedCuboid, myCuboid);
				} else if (getSurfaceArea(recveivedCuboid)
						== getSurfaceArea(myCuboid)) {
					if (recveivedCuboid->width < myCuboid->width) {
						swap(recveivedCuboid, myCuboid);
					}
				}
			}

		} else { // row in deascending order

			if (order == 1) { // noraml order
				if (getSurfaceArea(recveivedCuboid)
						< getSurfaceArea(myCuboid)) {

					swap(recveivedCuboid, myCuboid);

				} else if (getSurfaceArea(recveivedCuboid)
						== getSurfaceArea(myCuboid)) {
					if (recveivedCuboid->width < myCuboid->width) {
						swap(recveivedCuboid, myCuboid);

					}
				}

			} else { // swap order direction
				if (getSurfaceArea(recveivedCuboid)
						> getSurfaceArea(myCuboid)) {

					swap(recveivedCuboid, myCuboid);
				} else if (getSurfaceArea(recveivedCuboid)
						== getSurfaceArea(myCuboid)) {
					if (recveivedCuboid->width > myCuboid->width) {
						swap(recveivedCuboid, myCuboid);

					}
				}

			}

		}

	} else {	// column
		if (order == 1) {
			if (getSurfaceArea(recveivedCuboid) > getSurfaceArea(myCuboid)) {
				swap(recveivedCuboid, myCuboid);
			} else if (getSurfaceArea(recveivedCuboid)
					== getSurfaceArea(myCuboid)) {
				if (recveivedCuboid->width > myCuboid->width) {
					swap(recveivedCuboid, myCuboid);
				}
			}
		} else { // swap order direction
			if (getSurfaceArea(recveivedCuboid) < getSurfaceArea(myCuboid)) {
				swap(recveivedCuboid, myCuboid);
			} else if (getSurfaceArea(recveivedCuboid)
					== getSurfaceArea(myCuboid)) {
				if (recveivedCuboid->width < myCuboid->width) {
					swap(recveivedCuboid, myCuboid);
				}
			}
		}

	}

}


/**
 * Function that swaps between two cuboids
 */
void swap(Cuboid *recveivedCuboid, Cuboid *myCuboid) {
	Cuboid temp;
	temp = *recveivedCuboid;
	*recveivedCuboid = *myCuboid;
	*myCuboid = temp;
}

/**
 * Function to read the data from the input file.
 */
void readFile(Cuboid *matrix, FILE *fp, int size) {

	for (int i = 0; i < size; i++) {

		fscanf(fp, "%d", &matrix[i].id);
		fscanf(fp, "%lf", &matrix[i].length);
		fscanf(fp, "%lf", &matrix[i].width);
		fscanf(fp, "%lf", &matrix[i].height);
	}
}

/**
 * Function to store the result to file/
 */

void storeToFile(Cuboid *matrix, int size, FILE *fp) {

	fp = fopen(RESULT_FILE_PATH, "w");

	for (int i = 0; i < size; i++) {
		fprintf(fp, "%d ", matrix[i].id);
	}
	fclose(fp);

}

/**
 * Function that peforms the calculation of the surface area of a cuboid.
 *  	   ________
 *		  /  	  /|
 * 		 /		 / | height
 * 		---------  |
 * 		|		|  /
 * 		|		| /	length
 * 		|		|/
 * 		---------
 * 			width
 */
double getSurfaceArea(Cuboid *cuboid) {

	double width, height, length;
	width = cuboid->width;
	height = cuboid->height;
	length = cuboid->length;

	return (2 * width * height) + (2 * width * length) + (2 * height * length);

}
