#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "dataset.h" // 定义一些数据
#include "my_svm.h" // MySVM继承自CvSVM的类

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//
	MySVM svm;//

  //
	if(TRAIN)
	{
		string ImgName;//
		ifstream finPos(PosSamListFile);//
		//ifstream finPos("PersonFromVOC2012List.txt");//正样本图片的文件名列表
		ifstream finNeg(NegSamListFile);//

		Mat sampleFeatureMat;//
		Mat sampleLabelMat;//


		//
		for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
		{
			cout<<"处理："<<ImgName<<endl;
			ImgName = "dataset/pos/trueimg" + ImgName;//
			Mat src = imread(ImgName);//
			if(CENTRAL_CROP)
        if(src.cols >= 96 && src.rows >= 160)
				    src = src(Rect(16,16,64,128));//

			vector<float> descriptors;//
			hog.compute(src,descriptors,Size(8,8));//(8,8)
			//

			//
			if( 0 == num )
			{
				DescriptorDim = descriptors.size();//
				sampleFeatureMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, DescriptorDim, CV_32FC1);
				sampleLabelMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, 1, CV_32FC1);
			}

			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num,i) = descriptors[i];//
			sampleLabelMat.at<float>(num,0) = 1;//
		}

		//
		for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
		{
			cout<<"处理："<<ImgName<<endl;
      ImgName = "dataset/neg/" + ImgName;//
			Mat src = imread(ImgName);//
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//
			hog.compute(src,descriptors,Size(8,8));//(8,8)
			//cout<<""<<descriptors.size()<<endl;

			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];//+
			sampleLabelMat.at<float>(num+PosSamNO,0) = -1;//

		}

		//
		if(HardExampleNO > 0)
		{
			ifstream finHardExample(HardExampleListFile);//
			//
			for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
			{
				cout<<"处理："<<ImgName<<endl;
        ImgName = "dataset/HardExample/" + ImgName;//
				Mat src = imread(ImgName);//


				vector<float> descriptors;//
				hog.compute(src,descriptors,Size(8,8));//

				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for(int i=0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
				sampleLabelMat.at<float>(num+PosSamNO+NegSamNO,0) = -1;//负样本类别为-1，无人
			}
		}


	/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

			}
			fout<<endl;
		}*/

		//
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout<<"开始训练SVM分类器"<<endl;
		svm.train(sampleFeatureMat,sampleLabelMat, Mat(), Mat(), param);//训练分类器
		cout<<"训练完成"<<endl;
		svm.save("SVM_HOG.xml");//

	}
	else //
	{
		svm.load("SVM_HOG.xml");//
	}


	DescriptorDim = svm.get_var_count();//
	int supportVectorNum = svm.get_support_vector_count();//
	cout<<"numbers of supportVector"<<supportVectorNum<<endl;

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
	//
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	//
	myDetector.push_back(svm.get_rho());
	cout<<"number of testing dimensions"<<myDetector.size()<<endl;
	//
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//
	ofstream fout("HOGDetectorForOpenCV.txt");
	for(int i=0; i<myDetector.size(); i++)
	{
		fout<<myDetector[i]<<endl;
	}

	Mat src = imread(TestImageFileName);
	vector<Rect> found, found_filtered;//
	cout<<"HOG detection"<<endl;
	myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//


	for(int i=0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j=0;
		for(; j < found.size(); j++)
			if(j != i && (r & found[j]) == r)
				break;
		if( j == found.size())
			found_filtered.push_back(r);
	}
  cout<<"numbers of bounding box"<<found_filtered.size()<<endl;

	//
	for(int i=0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
	}

	imwrite("ImgProcessed.jpg",src);
	namedWindow("src",0);
	imshow("src",src);
	waitKey();

  return 0;
}
