/* -*- mode:c++; coding:utf-8-ws-dos; tab-width:4 -*- ==================== */
/* -----------------------------------------------------------------------
 * $Id: main.cpp 1 2018-01-05 14:19:12+09:00 nowatari $
 * ======================================================================= */

#include <iostream>
#include <vector>

#include "./NeuralNet.h"


/*----------------------------------------------------------------------
 *
 *----------------------------------------------------------------------*/
int main(void)
{
	NeuralNet	xorNet;
	std::vector<double> input[2] = { 
		{1.0, 0.0, 0.0,
		 0.0, 0.0, 0.0,
		 0.0, 0.0, 0.0 },
	   { 0.0, 0.0, 0.0, 
		 0.0, 0.0, 0.0,
		 0.0, 0.0, 1.0 },
	};
	std::vector<double> teacher[2] = { {1.0}, {0.0}, };

#if 0
	// 4x4x2 -> 3x3x1
	//xorNet.AddConvolutionLayer(4, 4, 2, 2, 2, 1, 0);
	//xorNet.AddReLULayer(3*3*2);
	//xorNet.AddAffineLayer(3 * 3 * 2, 1);
	//xorNet.AddSigmoidLayer(1);
	//xorNet.AddAffineLayer(3*3*1, 2);
	//xorNet.AddSoftMaxLayer(2);
	// 4x4x2 -> 1x1x1
	xorNet.AddConvolutionLayer(3, 3, 1, 2, 2, 1, 0);
	xorNet.AddReLULayer(2*2*2);
	xorNet.AddMaxPoolingLayer(2, 2, 2, 2, 1, 0);
	//xorNet.AddAffineLayer(2*2*1, 1);
	xorNet.AddAffineLayer(2, 1);
	xorNet.AddSigmoidLayer(1);
#else
	xorNet.AddAffineLayer(3 * 3 * 1, 2);
	xorNet.AddReLULayer(2);
	xorNet.AddAffineLayer(2, 1);
	xorNet.AddSigmoidLayer(1);
#endif

	for (auto count = 0; count < 100000; ++count)
	{
		std::vector<double>	output[4] = { {1.0}, {0.0}, {0.0}, {0.0} };

		std::cout << "step  = " << count << std::endl;

		double totalError = 0.0;

		for (auto i = 0; i < 2/*4*/; ++i)
		{
			xorNet.SetInput(input[i]);
			xorNet.Forward();
			xorNet.GetOutput(output[i]);
			
			double error	= xorNet.CalcLoss(teacher[i]);

			std::cout << "{";
			for (auto v : input[i])
				std::cout << v << ",";

			std::cout << "} = ";

			for (auto v : output[i])
				std::cout << v << "," ;
			
			std::cout << "(";
			for (auto v : teacher[i])
				std::cout << v << ",";

			std::cout << ")" << std::endl;

			if (error <= (0.01*0.01))
				continue;

			xorNet.Backward();
			xorNet.Learn(0.1);

			totalError += error;
		}

		if (totalError == 0.0)
		{
			std::cout << "learn end" << std::endl;
			break;
		}
	}
	return (0);
}

