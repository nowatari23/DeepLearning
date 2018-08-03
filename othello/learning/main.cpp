/* -*- mode:c++; coding:utf-8-ws-dos; tab-width:4 -*- ==================== */
/* -----------------------------------------------------------------------
 * $Id: main.cpp 2720 2017-12-30 21:04:35+09:00 nowatari $
 * ======================================================================= */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <iostream>
#include <vector>

#include "../NeuralNet.h"
#include "../teacherData.h"

/*======================================================================
 *
 *======================================================================*/
int main(int argc, char **argv)
{
	NeuralNet	othelloNet;

#if 0
#if 1
#if 0
	// ニューラルネット作成
	// 8x8x2 -> 7x7x64
	othelloNet.AddConvolutionLayer(8, 8, 2, 2, 32, 1, 0);
	othelloNet.AddRReLULayer(7 * 7 * 32);
	// 7x7x64 -> 7x7x64
	//othelloNet.AddMaxPoolingLayer( 7, 7, 8, 3, 1, 1);

	// 7x7x64 -> 6x6x64
	othelloNet.AddConvolutionLayer(7, 7, 32, 2, 32, 1, 0);
	othelloNet.AddRReLULayer(6 * 6 * 32);
	/*
	// 6x6x64 -> 6x6x64
	//othelloNet.AddMaxPoolingLayer( 6, 6, 8, 3, 1, 1);
	
	// 6x6x64 -> 5x5x64
	othelloNet.AddConvolutionLayer(6, 6, 32, 2, 32, 1, 0);
	othelloNet.AddRReLULayer(5 * 5 * 32);
	// 5x5x64 -> 5x5x64
	//othelloNet.AddMaxPoolingLayer( 5, 5, 8, 3, 1, 1);
	
	// 5x5x64 -> 4x4x64
	othelloNet.AddConvolutionLayer(5, 5, 32, 2, 64, 1, 0);
	othelloNet.AddRReLULayer(4 * 4 * 64);
	// 4x4x64 -> 4x4x64
	//othelloNet.AddMaxPoolingLayer( 4, 4, 64, 3, 1, 1);

	othelloNet.AddAffineLayer(4 * 4 * 64, 6 * 6 * 8);
	othelloNet.AddRReLULayer(6 * 6 * 8);
	*/

	othelloNet.AddAffineLayer(6 * 6 * 32, 6 * 6 * 8);
	othelloNet.AddRReLULayer(6 * 6 * 8);

	othelloNet.AddAffineLayer(6 * 6 * 8, 3 * 3 *8);
	othelloNet.AddRReLULayer(     3 * 3 * 8);

	othelloNet.AddAffineLayer(3 * 3 * 8, 64);
	othelloNet.AddRReLULayer(64);

	othelloNet.AddSoftMaxLayer(   64);
#endif
	othelloNet.AddConvolutionLayer(8, 8, 2, 3, 16, 1, 0);
	othelloNet.AddRReLULayer(6 * 6 * 16);

	othelloNet.AddConvolutionLayer(6, 6, 16, 3, 32, 1, 0);
	othelloNet.AddRReLULayer(4 * 4 * 32);

	othelloNet.AddAffineLayer(4 * 4 * 32, 64);
	othelloNet.AddRReLULayer(64);

	othelloNet.AddSoftMaxLayer(64);

#else
	othelloNet.AddAffineLayer(8 * 8 * 2, 8 * 8 * 16);
	othelloNet.AddRReLULayer(8 * 8 * 16);
	
	othelloNet.AddAffineLayer(8 * 8 * 16, 8 * 8 * 16);
	othelloNet.AddRReLULayer(8 * 8 * 16);
	
	othelloNet.AddAffineLayer(8 * 8 * 16, 8 * 8 * 16);
	othelloNet.AddRReLULayer(8 * 8 * 16);

	othelloNet.AddAffineLayer(8 * 8 * 16, 8 * 8 * 16);
	othelloNet.AddRReLULayer(8 * 8 * 16);
	
	othelloNet.AddAffineLayer(8 * 8 * 16, 64);
	othelloNet.AddRReLULayer(64);

	othelloNet.AddSoftMaxLayer(64);
#endif

#else
	// ニューラルネット読み込み
	{
		FILE	*pFile;
		char	buf[4];
		std::vector<char> data;

		pFile	= fopen("../othello.net", "rb");

		while (fread(buf, 1, sizeof(buf), pFile) > 0)
		{
			for (int i = 0; i < sizeof(buf); ++i)
				data.push_back(buf[i]);
		}

		fclose(pFile);

		othelloNet.Load(data);
	}
	// 教師データ読み込み
	teacherData	log(othelloNet.GetInputNum(), othelloNet.GetOutputNum());

	log.Load("teacher.log");
	
	unsigned int learnEnd = log.GetDataCount();// 1;
	int learnCount = 0;
	double learnRatio = 0.001;
	double threshold = 1.0e-3;
	while (learnCount < 1000000)
	{
		double	totalError = 0.0;
		std::vector<double> output;

		output.resize(othelloNet.GetOutputNum());

		++learnCount;

		std::cout << "learn count = " << learnCount << " learnEnd = (" << learnEnd;
		std::cout << "/" << log.GetDataCount() << ")";
		std::cout << " learn ratio = " << learnRatio;

		for (unsigned int i = 0; i < learnEnd; ++i)
		{
			othelloNet.SetInput(log.GetInput(i));
			othelloNet.Forward();
			othelloNet.GetOutput(output);

#if 0
			std::cout << "teacher data " << i << std::endl;

			for (unsigned int j = 0; j < output.size(); ++j)
				std::cout << output[j] << " " << log.GetTeacher(i)[j] << std::endl;
#endif
			totalError += othelloNet.CalcCrossEntropyLoss(log.GetTeacher(i));

			othelloNet.Backward();
		}

		std::cout << " error = " << totalError << std::endl;

		if (totalError < threshold)
		{
			othelloNet.LearnAdamReset();

			// ニューラルネット保存
			{
				FILE	*pFile;

				std::vector<char> data;
				othelloNet.Save(data);

				pFile = fopen("../othello.net", "wb");

				fwrite(&data[0], 1, data.size(), pFile);

				fclose(pFile);
			}

			if (++learnEnd >= log.GetDataCount())
			{
				std::cout << "learn end" << std::endl;
				break;
			}
#if 0
			if ((learnEnd % 80) == 0)
			{
				learnRatio *= 0.1;
				othelloNet.SetLearnRatio(learnRatio);

			}
#endif
		}
		//othelloNet.DeltaNormalize();
		//othelloNet.Learn(learnRatio / learnEnd);
		othelloNet.LearnAdam();// learnRatio / learnEnd);
	}
#endif
	// ニューラルネット保存
	{
		FILE	*pFile;

		std::vector<char> data;
		othelloNet.Save(data);
		
		pFile	= fopen("../othello.net", "wb");
		
		fwrite(&data[0], 1, data.size(), pFile);
		
		fclose(pFile);
	}
	
	return (0);
}

