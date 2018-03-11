#pragma once

#include <qelapsedtimer.h>

class AdaptiveMedianFilterCuda {
public:
	bool init();

	bool enabled();

	bool filterImageWithCuda(int width, int height,
		unsigned char *imageData, unsigned char *filteredImageData,
		int bytesPerLine, int bytesCount, qint64 *computeOnlyTimeout);

private:
	bool cudaEnabled;
};