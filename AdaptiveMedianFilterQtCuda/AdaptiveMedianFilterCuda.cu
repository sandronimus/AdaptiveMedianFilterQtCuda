#include <qelapsedtimer.h>
#include "AdaptiveMedianFilterCuda.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_AREA_SIZE 7
#define MEDIAN_BUFFER_SIZE (MAX_AREA_SIZE * MAX_AREA_SIZE + 1)

__device__ void quickSort(unsigned char *arr, int left, int right) {
	int i = left, j = right;
	int tmp;
	int pivot = arr[(left + right) / 2];

	/* partition */
	while (i <= j) {
		while (arr[i] < pivot)
			i++;
		while (arr[j] > pivot)
			j--;
		if (i <= j) {
			tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
			i++;
			j--;
		}
	};

	/* recursion */
	if (left < j)
		quickSort(arr, left, j);
	if (i < right)
		quickSort(arr, i, right);
}

__global__ void filterKernel(unsigned char *imageData, unsigned char *filteredImageData, int bytesPerLine,
	unsigned char *medianBuffer)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int width = gridDim.x;
	int height = gridDim.y;

	bool processed = false;

	int pixelOffset = y * bytesPerLine + x;

	unsigned int pixel = imageData[pixelOffset];

	// текущая окрестность точки
	int n = 3;

	unsigned char *median = medianBuffer + ((y * width) + x) * MEDIAN_BUFFER_SIZE;

	//thrust::device_vector<unsigned char> median(MAX_AREA_SIZE * MAX_AREA_SIZE + 1, 255);

	//std::array<unsigned char, MAX_AREA_SIZE * MAX_AREA_SIZE + 1> median;
	
	while (!processed) {
		// минимальное значение яркости в окрестности
		double zMin = 255;
		// максимальное значение яркости в окрестности
		double zMax = 0;
		// медиана значений яркости
		double zMed = 0;

		// размер окрестности в одну сторону
		int sDelta = (n - 1) / 2;

		int processedPixelCount = 0;

		// проходим по окрестности точки, вычисляя значения параметров
		for (int sx = x - sDelta; sx <= x + sDelta; sx++) {
			for (int sy = y - sDelta; sy <= y + sDelta; sy++) {
				if (sx < 0 || sy < 0 || sx >= width || sy >= height) {
					continue;
				}

				unsigned int currentPixel = imageData[sy * bytesPerLine + sx];

				if (currentPixel < zMin) {
					zMin = currentPixel;
				}

				if (currentPixel > zMax) {
					zMax = currentPixel;
				}

				median[processedPixelCount] = currentPixel;

				processedPixelCount++;
			}
		}

		quickSort(median, 0, processedPixelCount);

		zMed = median[processedPixelCount / 2];

		double a1 = zMed - zMin;
		double a2 = zMed - zMax;

		if (a1 > 0 && a2 < 0) {
			double b1 = pixel - zMin;
			double b2 = pixel - zMax;

			if (b1 > 0 && b2 < 0) {
				filteredImageData[pixelOffset] = pixel;
			}
			else {
				filteredImageData[pixelOffset] = zMed;
			}

			processed = true;
		}
		else {
			n += 2;
			if (n > 7) {
				filteredImageData[pixelOffset] = zMed;
				processed = true;
			}
		}
	}
}

bool AdaptiveMedianFilterCuda::init() {
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
	}

	cudaEnabled = cudaStatus == cudaSuccess;

	return cudaEnabled;
}

bool AdaptiveMedianFilterCuda::enabled() {
	return cudaEnabled;
}

bool AdaptiveMedianFilterCuda::filterImageWithCuda(int width, int height,
						 unsigned char *imageData, unsigned char *filteredImageData,
						 int bytesPerLine, int bytesCount, qint64 *computeOnlyTimeout) {
	unsigned char *dev_imageData = 0;
	unsigned char *dev_filteredImageData = 0;
	unsigned char *dev_medianBuffer = 0;
	cudaError_t cudaStatus;
	dim3 image(width, height);

	cudaStatus = cudaMalloc((void**)&dev_imageData, bytesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for image data failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_filteredImageData, bytesCount);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for filtered image data failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_medianBuffer, MEDIAN_BUFFER_SIZE * width * height *
		sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc for median buffer failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_imageData, imageData, bytesCount, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for image data to device failed!");
		goto Error;
	}

	QElapsedTimer computeOnlyTimer;
	computeOnlyTimer.start();
	filterKernel<<<image, 1>>>(dev_imageData, dev_filteredImageData, bytesPerLine, dev_medianBuffer);
	*computeOnlyTimeout = computeOnlyTimer.elapsed();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "filterKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching filterKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(filteredImageData, dev_filteredImageData, bytesCount, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy for filtered image data from device failed!");
		goto Error;
	}

Error:
	cudaFree(dev_imageData);
	cudaFree(dev_filteredImageData);
	cudaFree(dev_medianBuffer);

	return cudaStatus == cudaSuccess;
}