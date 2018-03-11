#pragma once

#include <qstring.h>
#include <QtGui\qimage.h>
#include "AdaptiveMedianFilterCuda.cuh"

class AdaptiveMedianFilter
{
public:
	AdaptiveMedianFilter(AdaptiveMedianFilterCuda *filterCuda);
	~AdaptiveMedianFilter();

	void processFile(QString fileName, QString outputFileName, QString outputFileNameCuda);

private:
	AdaptiveMedianFilterCuda *filterCuda;

	QImage image;
	QImage filteredImage;

	uchar *imageData;
	uchar *filteredImageData;

	int width;
	int height;
	int bytesPerLine;

	void processPixel(int x, int y);
	void quickSort(unsigned int *arr, int left, int right);
};

