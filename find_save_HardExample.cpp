#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "dataset.h"
#include "my_svm.h"

using namespace std;
using namespace cv;

int HardExampleCount = 0;

int main(int argc, char** argv)
{
  Mat src;
	string ImgName;

	char saveName[256];
	ifstream fin("INRIANegativeImageList.txt");//


	int DescriptorDim;//
	MySVM svm;//
  svm.load("SVM_HOG.xml");


	DescriptorDim = svm.get_var_count();//
	int supportVectorNum = svm.get_support_vector_count();//
	cout<<"支持向量个数："<<supportVectorNum<<endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//

	//
	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//
		for(int j=0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i,j) = pSVData[j];
		}
	}

	//
	double * pAlphaData = svm.get_alpha_vector();//
	for(int i=0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0,i) = pAlphaData[i];
	}


	resultMat = -1 * alphaMat * supportVectorMat;

	vector<float> myDetector;
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	myDetector.push_back(svm.get_rho());
	cout<<"检测子维数："<<myDetector.size()<<endl;
	//
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);


  while(getline(fin,ImgName))
  {
    cout<<"处理："<<ImgName<<endl;
    ImgName = "INRIAPerson/Train/neg/" + ImgName;
    src = imread(ImgName,1);//

      vector<Rect> found, found_filtered;

      myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);
      //t = (double)getTickCount() - t;
      //printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());
      size_t i, j;
      for( i = 0; i < found.size(); i++ )
      {
          Rect r = found[i];
          for( j = 0; j < found.size(); j++ )
              if( j != i && (r & found[j]) == r)
                  break;
          if( j == found.size() )
              found_filtered.push_back(r);
      }

      for( i = 0; i < found_filtered.size(); i++ )
      {
          Rect r = found_filtered[i];
          // the HOG detector returns slightly larger rectangles than the real objects.
          // so we slightly shrink the rectangles to get a nicer output.
          //r.x += cvRound(r.width*0.1);
          //r.width = cvRound(r.width*0.8);
          //r.y += cvRound(r.height*0.07);
          //r.height = cvRound(r.height*0.8);
          if(r.x < 0)
            r.x = 0;
          if(r.y < 0)
            r.y = 0;
          if(r.x + r.width > src.cols)
            r.width = src.cols - r.x;
          if(r.y + r.height > src.rows)
            r.height = src.rows - r.y;
          Mat imgROI = src(Rect(r.x, r.y, r.width, r.height));
          resize(imgROI,imgROI,Size(64,128));
          sprintf(saveName,"dataset/HardExample/hardexample%06d.jpg",++HardExampleCount);
          imwrite(saveName,imgROI);
          //rectangle(src, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
      }
      //imshow("people detector", src);
      //waitKey(0);
  }

  cout<<"HardExampleCount: "<<HardExampleCount<<endl;

  return 0;
}
