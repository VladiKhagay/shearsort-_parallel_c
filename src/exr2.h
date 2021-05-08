
#define MASTER 0
#define TAG 0
#define ROW_DIR 1
#define COL_DIR 0
#define ASC 1
#define DESC 0
#define DISP 1
#define DIMS 2
#define CUBOID_NUM_ATTR 4
#define DAT_FILE_PATH "../cuboids.dat"
#define RESULT_FILE_PATH "../result.dat"

//---------- Cuboid struct : ----------//

typedef struct {
	int id;
	double length;
	double width;
	double height;
} Cuboid;

//---------- Functions prototypes : ----------//

MPI_Comm createCartesianComm(int rank, int numOfProcesses, int *coords);
void createCuboidType(MPI_Datatype *cubidType);
void shearSort(MPI_Comm comm, const int size, const int order, Cuboid *myCuboid,
		MPI_Datatype cuboidType);
void oddEvenSort(MPI_Comm comm, const int direction, const int rowSize,
		const int order, Cuboid *myCuboid, MPI_Datatype cuboidType);

void exchangeInRow(int *coords, int i, int source, int dest, int order,
		Cuboid *myCuboid, MPI_Datatype cuboidType, MPI_Comm comm,
		MPI_Status status);

void exchangeInColumn(int *coords, int i, int source, int dest, int order,
		Cuboid *myCuboid, MPI_Datatype cuboidType, MPI_Comm comm,
		MPI_Status status);

void swapByOrder(Cuboid *recveivedCuboid, Cuboid *myCuboid, int order,
		int direction, int asc);
void swap(Cuboid *recveivedCuboid, Cuboid *myCuboid);
void readFile(Cuboid *matrix, FILE *fp, int size);
void storeToFile(Cuboid *matrix, int size, FILE *fp);
double getSurfaceArea(Cuboid *cuboid);
