#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

__device__ int *iToxy(int, int);
__device__ int xyToi(int, int, int);
__device__ uchar3 promedio_actual(uchar3*, int, int, int, int);
__global__ void cudaBlur(uchar3*, uchar3*, int, int, int, int);
inline int _ConvertSMVer2Cores(int, int);

int main(int n, char* argv[])
{
	int hilos = 0;
	int bloques = 0;
	int hilos_totales;
	int cuda_err = cudaSuccess;
	int tam_imagen = 0;
	float tam_kernel = -1;
	Mat image;

	//	Verificacion de los parametros para la correcta ejecucion del programa
	if (n != 4) {
		cout << "blur_cuda <ruta img> <Tamano del kernel> <thread>" << endl;
		return 0;
	}

	stringstream ss;
	ss << argv[3];
	ss >> hilos_totales;

	//Se especifican las caracteristicas de la tarjeta
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int hilos_max = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

	//Se hace el calculo de bloques e hilos por bloque
	bloques = (hilos_totales / (hilos_max * 2)) + 1;
	hilos = hilos_totales / bloques;

	//	Determina el tam_kernel del kernel
	ss.clear();
	ss << argv[2];
	ss >> tam_kernel;

	//	Se carga la imagen en host
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//	Determina el tamaño del bloque de memoria para la imagen
	tam_imagen = image.cols*image.rows * sizeof(uchar3);

	//	Reservar la memoria en device para imagen original
	uchar3 *image_d;
	cuda_err = cudaMalloc(&image_d, tam_imagen);

	//	Reservar la memoria en device para imagen de respuesta
	uchar3 *ans_d;
	cuda_err = cudaMalloc(&ans_d, tam_imagen);

	//	Copiar imagen original al puntero image en device
	cuda_err = cudaMemcpy(image_d, image.data, tam_imagen, cudaMemcpyHostToDevice);

	//	llamar proceso de blur paralelo
	cudaBlur <<< bloques, hilos >>> (image_d, ans_d, image.cols, image.rows, hilos*bloques, (int)floor(tam_kernel));

	//	Copia la respuesta del apuntador ans_d a image, desde el device al host
	cuda_err = cudaMemcpy(image.data, ans_d, tam_imagen, cudaMemcpyDeviceToHost);

	//	Liberar memoria en device
	cuda_err = cudaFree(image_d);

	cuda_err = cudaFree(ans_d);

	imwrite( "blur.jpg", image );

	return 0;
}

/**
* Recorre los puntos del vector de datos de la imagen haciendo el blur a cada uno de ellos
*/
__global__ void cudaBlur(uchar3 *image, uchar3 *ans, int cols, int rows, int n_hilos, int tam_kernel) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id; i < cols*rows; i += n_hilos) {
		*(ans + i) = promedio_actual(image, i, rows, cols, tam_kernel);
	}
	return;
}

/**
* Convierte i a una cordenada de la forma (x,y).
* Retorna un apuntador con 2 pociciones reservadas.
* En la primera almacena el valor de x
* En la segunda almacena el valor de y
*/
__device__ int *iToxy(int i, int cols) {
	int *ans;
	ans = (int*)malloc(2 * sizeof(int));
	*ans = i%cols;
	*(ans + 1) = i / cols;
	return ans;
}

/**
* convierte una cordenada (x,y) a un valor i para array
* Retorna un entero con el valor de i
*/
__device__ int xyToi(int x, int y, int cols) {
	return cols*y + x;
}

/**
* Halla la suma promediada de los pixeles vecinos en base a un kernel
* image*			Un apuntador a el vector de datos de la imagen
* pos:			El indice del pixel, el indice en base a un array unidimencional
* rows, cols:	dimenciones de la imagen que se esta procesando
* tam_kernel:		El tamano del kernel para los pixeles vecinos
* Retorna un entero con el valor de i
*/
__device__ uchar3 promedio_actual(uchar3* image, int pos, int rows, int cols, int tam_kernel) {
	float  sum_peso;
	float3 sum = {0,0,0};

	sum_peso = 0;

	int *ptr_aux = iToxy(pos, cols);
	int x = *ptr_aux;
	int y = *(ptr_aux + 1);
	free(ptr_aux);

	for (int k = -tam_kernel; k <= tam_kernel; k++) {
		for (int j = -tam_kernel; j <= tam_kernel; j++) {
			if ((x + k) >= 0 && (x + k) < cols &&
				(y + j) >= 0 && (y + j) < rows) {
				float peso = exp(-(k*k + j*j) / (float)(2 * tam_kernel*tam_kernel)) / (3.141592 * 2 * tam_kernel*tam_kernel);
				sum.x += peso * (*(image + xyToi(x + k, y + j, cols))).x;
				sum.y += peso * (*(image + xyToi(x + k, y + j, cols))).y;
				sum.z += peso * (*(image + xyToi(x + k, y + j, cols))).z;
				sum_peso += peso;
			}
		}
	}

	uchar3 ans;

	ans.x = (uchar)floor(sum.x / sum_peso);
	ans.y = (uchar)floor(sum.y / sum_peso);
	ans.z = (uchar)floor(sum.z / sum_peso);

	return ans;
}

/**
 * Funcion de "cuda_helper.h" localizada en samples/common para determinar el numero de cores por multiprocesador del device
 */
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	return nGpuArchCoresPerSM[index - 1].Cores;
}
