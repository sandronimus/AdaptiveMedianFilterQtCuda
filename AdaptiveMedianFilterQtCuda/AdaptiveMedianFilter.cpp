#include "AdaptiveMedianFilter.h"
#include "AdaptiveMedianFilterCuda.cuh"
#include <qdebug.h>
#include <QtGui\qimage.h>
#include <qelapsedtimer.h>

#define MAX_AREA_SIZE 7


AdaptiveMedianFilter::AdaptiveMedianFilter(AdaptiveMedianFilterCuda *filterCuda)
	: filterCuda(filterCuda)
{
}


AdaptiveMedianFilter::~AdaptiveMedianFilter()
{
}

void AdaptiveMedianFilter::processFile(QString fileName, QString outputFileName, QString outputFileNameCuda) {
	image = QImage(fileName).convertToFormat(QImage::Format_Grayscale8);

	if (image.isNull()) {
		qDebug() << "image not loaded ";

		return;
	}

	width = image.width();
	height = image.height();

	filteredImage = QImage(width, height, image.format());

	imageData = image.bits();
	filteredImageData = filteredImage.bits();

	bytesPerLine = image.bytesPerLine();

	QElapsedTimer timer;
	timer.start();
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			processPixel(x, y);
		}
	}
	qDebug() << "Filtering " << fileName << " with Qt taken " << timer.elapsed() << " milliseconds";
	filteredImage.save(outputFileName);

	if (filterCuda->enabled()) {
		QElapsedTimer timerCuda;
		qint64 cudaComputeOnlyTimeout;
		timerCuda.start();
		filterCuda->filterImageWithCuda(width, height, imageData, filteredImageData, bytesPerLine, image.byteCount(), &cudaComputeOnlyTimeout);
		qDebug() << "Filtering " << fileName << " with Cuda taken " << timerCuda.elapsed() << " milliseconds";
		qDebug() << "Compute only filtering " << fileName << " with Cuda taken " << timerCuda.elapsed() << " milliseconds";
		filteredImage.save(outputFileNameCuda);
	}
}

void AdaptiveMedianFilter::processPixel(int x, int y)
{
	bool processed = false;

	int pixelOffset = y * bytesPerLine + x;

	unsigned int pixel = imageData[pixelOffset];

	// текущая окрестность точки
	int n = 3;

	unsigned int median[MAX_AREA_SIZE * MAX_AREA_SIZE + 1];

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


void AdaptiveMedianFilter::quickSort(unsigned int *arr, int left, int right) {
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