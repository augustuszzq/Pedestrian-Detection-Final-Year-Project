#include <iostream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#define INRIANegativeImageList "INRIANegativeImageList.txt"

using namespace std;
using namespace cv;

int CropImageCount = 0;

int main()
{
	Mat src;
	string ImgName;

	char saveName[256];
	ifstream fin(INRIANegativeImageList);


	while(getline(fin,ImgName))
	{
		cout<<"处理："<<ImgName<<endl;
		ImgName = "INRIAPerson/Train/neg/" + ImgName;

		src = imread(ImgName,1);

		if(src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));
			for(int i=0; i<10; i++)
			{
				int x = ( rand() % (src.cols-64) );
				int y = ( rand() % (src.rows-128) );
				//cout<<x<<","<<y<<endl;
				Mat imgROI = src(Rect(x,y,64,128));
				sprintf(saveName,"dataset/neg/noperson%06d.jpg",++CropImageCount);
				imwrite(saveName, imgROI);
			}
		}
	}

  cout<<"总共裁剪出"<<CropImageCount<<"张图片"<<endl;

}
