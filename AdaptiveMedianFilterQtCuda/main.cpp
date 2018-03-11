#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QDebug>
#include "AdaptiveMedianFilter.h"
#include "AdaptiveMedianFilterCuda.cuh"

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	QDir dir;
	QDir filteredDir;

	dir.cd("images");
	filteredDir.cd("images_filtered");

	qDebug() << "image dir: " << dir.absolutePath();

	QStringList fileList = dir.entryList();
	qDebug() << "image files: " << fileList;

	AdaptiveMedianFilterCuda filterCuda;
	AdaptiveMedianFilter filter(&filterCuda);

	filterCuda.init();

	for (int i = 0; i < fileList.length(); i++)
	{
		QString fileName = dir.absoluteFilePath(fileList.at(i));
		QString outputFileName = filteredDir.absoluteFilePath(fileList.at(i));
		QString outputFileNameCuda = filteredDir.absoluteFilePath(fileList.at(i));
		outputFileNameCuda = outputFileNameCuda.insert(outputFileNameCuda.lastIndexOf(".") - 1, "_cuda");

		if (dir.exists(fileList.at(i))) {
			filter.processFile(fileName, outputFileName, outputFileNameCuda);
		}
	}

	return a.exec();
}
