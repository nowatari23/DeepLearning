/* -*- mode:c++; coding:utf-8-ws-dos; tab-width:4 -*- ==================== */
/* -----------------------------------------------------------------------
 * $Id: NeuralNet.cpp 2720 2018-01-02 21:21:06+09:00 nowatari $
 * ======================================================================= */

#include <random>

#include "NeuralNet.h"

//----------------------------------------------------------------------
/**
 * コンストラクタ
 *
 * @param inputNum  入力の要素数
 * @param outputNum 出力の要素数
 */
//----------------------------------------------------------------------
NeuralNet::AffineLayer::AffineLayer(unsigned int inputNum,
									unsigned int outputNum) :
Layer(inputNum, outputNum, LayerType::Affine)
{
	std::random_device			rd;
	std::mt19937				mt(rd());
	std::normal_distribution<>	dist(0.0, 1.0);

	m_Input.resize( inputNum);
	
	m_Weight.resize(outputNum * inputNum);
	m_Bias.resize(  outputNum);

	m_MomentWeight.resize(  outputNum * inputNum);
	m_VelocityWeight.resize(outputNum * inputNum);

	m_MomentBias.resize(  outputNum);
	m_VelocityBias.resize(outputNum);
	
	m_DeltaWeight.resize(outputNum * inputNum);
	m_DeltaBias.resize(  outputNum);
	
	for (unsigned int o = 0; o < outputNum; ++o)
	{
		for (unsigned int i = 0; i < inputNum; ++i)
		{
			WeightAt(        i, o)	= dist(mt);
			MomentWeightAt(  i, o)	= 0.0;
			VelocityWeightAt(i, o)	= 0.0;
			DeltaWeightAt(   i, o)	= 0.0;
		}

		BiasAt(        o)	= dist(mt);
		MomentBiasAt(  o)	= 0.0;
		VelocityBiasAt(o)	= 0.0;
		DeltaBiasAt(   o)	= 0.0;
	}
}

//----------------------------------------------------------------------
/**
 * 前方出力
 *
 * @param  input  入力値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::AffineLayer::Forward(const std::vector<double> &input,
									 std::vector<double>       &output)
{
	if (input.size() != m_InputNum)
		return;
	
	output.resize(m_OutputNum);

	// 学習用に入力値の値を保存.
	for (unsigned int i = 0; i < m_InputNum; ++i)
		InputAt(i)	= input[i];
	
	for (unsigned int o = 0; o < m_OutputNum; ++o)
	{
		output[o]	= 0.0;
		
		for (unsigned int i = 0; i < m_Input.size(); ++i)
			output[o]	+= input[i] * WeightAt(i, o);

		output[o]	= output[o] + BiasAt(o);
	}
}

//----------------------------------------------------------------------
/**
 * 後方出力
 *
 * @param  delta  出力差分値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::AffineLayer::Backward(const std::vector<double> &delta,
									  std::vector<double>       &output)
{
	if (delta.size() != m_OutputNum)
		return;
	
	output.resize(m_InputNum);
	
	for (unsigned int i = 0; i < m_InputNum; ++i)
	{
		output[i]	= 0.0;
		
		for (unsigned int o = 0; o < m_OutputNum; ++o)
		{
			output[i]	+= delta[o] * WeightAt(i, o);

			DeltaWeightAt(i, o)	+= delta[o] * InputAt(i);
		}
	}

	for (unsigned int o = 0; o < m_OutputNum; ++o)
		DeltaBiasAt(o)	+= delta[o];
}

//----------------------------------------------------------------------
/**
 * 学習
 *
 * @param learnRatio 学習率
 */
//----------------------------------------------------------------------
void NeuralNet::AffineLayer::Learn(double learnRatio)
{
	for (unsigned int o = 0; o < m_OutputNum; ++o)
	{
		for (unsigned int i = 0; i < m_InputNum; ++i)
		{
			WeightAt(i, o)		+= learnRatio * DeltaWeightAt(i, o);
			DeltaWeightAt(i, o)	=  0.0;
		}
		
		BiasAt(o)		+= learnRatio * DeltaBiasAt(o);
		DeltaBiasAt(o)	=  0.0;
	}
}

//----------------------------------------------------------------------
/**
 * Adam学習
 *
 * @param alpha    係数更新率
 * @param beta1    モーメント更新率
 * @param beta2    速度更新率
 * @param epsilon  誤差値
 */
//----------------------------------------------------------------------
void NeuralNet::AffineLayer::LearnAdam(double alpha,
									   double beta1,
									   double beta2,
									   double epsilon)
{
	for (unsigned int o = 0; o < m_OutputNum; ++o)
	{
		for (unsigned int i = 0; i < m_InputNum; ++i)
		{
			MomentWeightAt(i, o)	= beta1 * MomentWeightAt(i, o)
									+ (1.0-beta1) * DeltaWeightAt(i, o);
			VelocityWeightAt(i, o)	= beta2 * VelocityWeightAt(i, o)
									+ (1.0-beta2)
									* pow(DeltaWeightAt(i, o), 2);
			
			const double	m	= MomentWeightAt(  i, o) / (1.0-beta1);
			const double	v	= VelocityWeightAt(i, o) / (1.0-beta2);
			
			WeightAt(i, o)		+= alpha * m / (sqrt(v) + epsilon);
			DeltaWeightAt(i, o)	=  0.0;
		}

		MomentBiasAt(o)		= beta1 * MomentBiasAt(o)
							+ (1.0-beta1) * DeltaBiasAt(o);
		VelocityBiasAt(o)	= beta2 * VelocityBiasAt(o)
							+ (1.0-beta2)
							* pow(DeltaBiasAt(o), 2);
		
		const double	m	= MomentBiasAt(  o) / (1.0-beta1);
		const double	v	= VelocityBiasAt(o) / (1.0-beta2);

		BiasAt(o)		+= alpha * m / (sqrt(v) + epsilon);
		DeltaBiasAt(o)	=  0.0;
	}
}

//----------------------------------------------------------------------
/**
 * コンストラクタ
 *
 * @param inputNum  入力の要素数
 * @param type      レイヤーの種類
 */
//----------------------------------------------------------------------
NeuralNet::ActivateLayer::ActivateLayer(unsigned int inputNum,
										unsigned int type) :
Layer(inputNum, inputNum, type)
{
}

//----------------------------------------------------------------------
/**
 * 前方出力
 *
 * @param  input  入力値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::ActivateLayer::Forward(const std::vector<double> &input,
									   std::vector<double>       &output)
{
	if (input.size() != m_InputNum)
		return;
	
	output.resize(m_OutputNum);
	
	for (unsigned int o = 0; o < m_OutputNum; ++o)
		output[o]	= ForwardFunc(input[o], o);
}

//----------------------------------------------------------------------
/**
 * 後方出力
 *
 * @param  delta  出力差分値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::ActivateLayer::Backward(const std::vector<double> &delta,
										std::vector<double>       &output)
{
	if (delta.size() != m_OutputNum)
		return;

	output.resize(m_InputNum);
	
	for (unsigned int o = 0; o < m_OutputNum; ++o)
		output[o]	= BackwardFunc(delta[o], o);
}

//----------------------------------------------------------------------
/**
 * コンストラクタ
 *
 * @param inputNum 入力数
 */
//----------------------------------------------------------------------
NeuralNet::RReLULayer::RReLULayer(unsigned int inputNum) :
ReLULayer(inputNum, LayerType::RReLU)
{
	m_Alpha.resize(inputNum);

	for (unsigned int i = 0; i < inputNum; ++i)
		SetAlpha(i, GetRandomAlpha());
}

//----------------------------------------------------------------------
/**
 * 学習
 *
 * @param learnRatio 学習率
 */
//----------------------------------------------------------------------
void NeuralNet::RReLULayer::Learn(double learnRatio)
{
	for (unsigned int i = 0; i < m_InputNum; ++i)
		SetAlpha(i, 0.5 * (GetAlpha(i) + GetRandomAlpha()));
}

//----------------------------------------------------------------------
/**
 * アルファ値を乱数で取得.
 *
 */
//----------------------------------------------------------------------
double NeuralNet::RReLULayer::GetRandomAlpha(void)
{
	std::random_device					rd;
	std::mt19937						mt(rd());
	std::uniform_real_distribution<>	dist(0.0, 0.1);

	return (dist(mt));
}

//----------------------------------------------------------------------
/**
 * 前方出力
 * -出力の総和を１に調整する
 * @param  input  入力値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::SoftMaxLayer::Forward(const std::vector<double> &input,
									  std::vector<double>       &output)
{
	double	total		= 0.0;
	double	maxValue;

	if (input.size() != m_InputNum)
		return;

	output.resize(m_OutputNum);

	// オーバーフロー対策に最大値で引く.
	maxValue	= input[0];
	for (unsigned int i = 1; i < m_InputNum; ++i)
	{
		if (input[i] > maxValue)
			maxValue	= input[i];
	}

	for (unsigned int o = 0; o < m_OutputNum; ++o)
	{
		output[o]	=  exp(input[o]-maxValue);
		total		+= output[o];
	}

	// 総和で割る
	for (unsigned int o = 0; o < m_OutputNum; ++o)
		output[o]	/= total;
}

//----------------------------------------------------------------------
/**
 * 後方出力
 *
 * @param  delta  出力差分値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::SoftMaxLayer::Backward(const std::vector<double> &delta,
									   std::vector<double>       &output)
{
	if (delta.size() != m_OutputNum)
		return;

	output.resize(m_InputNum);
	
	for (unsigned int o = 0; o < m_OutputNum; ++o)
		output[o]	= delta[o];
}

//----------------------------------------------------------------------
/**
 * コンストラクタ
 *
 * @param width      入力の横幅
 * @param height     入力の高さ
 * @param channel    入力のチャンネル数
 * @param filterSize フィルタのサイズ
 * @param filterNum  フィルタの数
 * @param stride     フィルタのストライド
 * @param padding    フィルタのパディング
 * @param type       レイヤータイプ
 */
//----------------------------------------------------------------------
NeuralNet::ConvolutionLayer::ConvolutionLayer(unsigned int width,
											  unsigned int height,
											  unsigned int channel,
											  unsigned int filterSize,
											  unsigned int filterNum,
											  unsigned int stride,
											  unsigned int padding) :
FilterLayer(width,
			height,
			channel,
			filterSize,
			filterNum,
			stride,
			padding,
			LayerType::Convolution)
{
	std::random_device			rd;
	std::mt19937				mt(rd());
	std::normal_distribution<>	dist(0.0, 1.0);

	m_Input.resize(m_InputNum);
	
	// フィルタバッファ
	m_Filter.resize(channel*filterNum*filterSize*filterSize);
	m_Bias.resize(  channel*filterNum);

	m_MomentFilter.resize(  channel*filterNum*filterSize*filterSize);
	m_VelocityFilter.resize(channel*filterNum*filterSize*filterSize);

	m_MomentBias.resize(  channel*filterNum);
	m_VelocityBias.resize(channel*filterNum);

	m_DeltaFilter.resize(channel*filterNum*filterSize*filterSize);
	m_DeltaBias.resize(  channel*filterNum);

	for (unsigned int c = 0; c < channel; ++c)
	{
		for (unsigned int f = 0; f < filterNum; ++f)
		{
			for (unsigned int x = 0; x < filterSize; ++x)
			{
				for (unsigned int y = 0; y < filterSize; ++y)
				{
					FilterAt(        x, y, f, c)	= pow(dist(mt), 2);
					MomentFilterAt(  x, y, f, c)	= 0.0;
					VelocityFilterAt(x, y, f, c)	= 0.0;
					DeltaFilterAt(   x, y, f, c)	= 0.0;
				}
			}
			BiasAt(f, c)			= pow(dist(mt), 2);
			MomentBiasAt(f, c)		= 0.0;
			VelocityBiasAt(f, c)	= 0.0;
			DeltaBiasAt(f, c)		= 0.0;
		}
	}
}

//----------------------------------------------------------------------
/**
 * 前方出力
 *
 * @param  input  入力値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::ConvolutionLayer::Forward(const std::vector<double> &input,
										  std::vector<double>       &output)
{
	if (input.size() != m_InputNum)
		return;
	
	output.resize(m_OutputNum);

	// 横幅.
	for (unsigned int w = 0; w < m_WMax; ++w)
	{
		// 高さ
		for (unsigned int h = 0; h < m_HMax; ++h)
		{
			// フィルタ数ループ
			for (unsigned int f = 0; f < m_FilterNum; ++f)
			{
				int	outputIndex	= OutputIndex(w, h, f);
				
				output[outputIndex]	= 0.0;
			}
		}
	}
	
	// チャンネル数ループ
	for (unsigned int c = 0; c < m_Channel; ++c)
	{
		// 横幅.
		for (unsigned int w = 0; w < m_WMax; ++w)
		{
			// 高さ
			for (unsigned int h = 0; h < m_HMax; ++h)
			{
				// フィルタ数ループ
				for (unsigned int f = 0; f < m_FilterNum; ++f)
				{
					int	outputIndex	= OutputIndex(w, h, f);
					
					// フィルタ計算.
					for (unsigned int i = 0; i < m_FilterSize; ++i)
					{
						int	iW	= w*m_Stride+i-m_Padding;

						if ((iW < 0) || (iW >= (int)m_Width))
							continue;
						
						for (unsigned int j = 0; j < m_FilterSize; ++j)
						{
							int	iH	= h*m_Stride+j-m_Padding;

							if ((iH < 0) || (iH >= (int)m_Height))
								continue;
							
							int	inputIndex	= InputIndex(iW, iH, c);

							output[outputIndex]	+=
								input[inputIndex]
								* FilterAt(i, j, f, c);
						}
					}
					output[outputIndex]	+= GetBias(f, c);
				}
			}
		}
		
		// 横幅.
		for (unsigned int w = 0; w < m_Width; ++w)
		{
			// 高さ
			for (unsigned int h = 0; h < m_Height; ++h)
			{
				unsigned int inputIndex	= InputIndex(w, h, c);
				
				InputAt(inputIndex)	= input[inputIndex];
			}
		}
	}
}

//----------------------------------------------------------------------
/**
 * 後方出力
 *
 * @param  delta  出力差分値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::ConvolutionLayer::Backward(const std::vector<double> &delta,
										   std::vector<double>       &output)
{
	if (delta.size() != m_OutputNum)
		return;

	output.resize(m_InputNum);

	// チャンネル数ループ
	for (unsigned int c = 0; c < m_Channel; ++c)
	{
		// 横幅.
		for (unsigned int w = 0; w < m_Width; ++w)
		{
			// 高さ
			for (unsigned int h = 0; h < m_Height; ++h)
			{
				int	inputIndex	= InputIndex(w, h, c);

				output[inputIndex]	= 0.0;
			}
		}
	}
	
	// チャンネル数ループ
	for (unsigned int c = 0; c < m_Channel; ++c)
	{
		// フィルタ数ループ
		for (unsigned int f = 0; f < m_FilterNum; ++f)
		{
			// 横幅.
			for (unsigned int w = 0; w < m_WMax; ++w)
			{
				// 高さ
				for (unsigned int h = 0; h < m_HMax; ++h)
				{
					int outputIndex	= OutputIndex(w, h, f);

					// フィルタ計算.
					for (unsigned int i = 0; i < m_FilterSize; ++i)
					{
						int	iW	= w*m_Stride-i+m_Padding;

						if ((iW < 0) || (iW >= (int)m_Width))
							continue;
						
						for (unsigned int j = 0; j < m_FilterSize; ++j)
						{
							int	iH	= h*m_Stride-j+m_Padding;

							if ((iH < 0) || (iH >= (int)m_Height))
								continue;

							int	inputIndex	= InputIndex(iW, iH, c);
							
							output[inputIndex]	+= delta[outputIndex]
												 * FilterBackAt(i, j, f, c);

							DeltaFilterAt(i, j, f, c)	+=
								delta[outputIndex]
							   * InputAt(inputIndex);
						}
					}

					DeltaBiasAt(f, c)	+= delta[outputIndex];
				}
			}
		}
	}
}

//----------------------------------------------------------------------
/**
 * 学習
 *
 * @param learnRatio 学習率
 */
//----------------------------------------------------------------------
void NeuralNet::ConvolutionLayer::Learn(double learnRatio)
{
	// フィルタ数ループ
	for (unsigned int f = 0; f < m_FilterNum; ++f)
	{
		// チャンネル数ループ
		for (unsigned int c = 0; c < m_Channel; ++c)
		{
			// フィルタ計算.
			for (unsigned int i = 0; i < m_FilterSize; ++i)
			{
				for (unsigned int j = 0; j < m_FilterSize; ++j)
				{
					FilterAt(i, j, f, c) += DeltaFilterAt(i, j, f, c)
										  * learnRatio;
					DeltaFilterAt(i, j, f, c)	= 0.0;
				}
			}
			
			BiasAt(f, c)		+= DeltaBiasAt(f, c)
								 * learnRatio;
			DeltaBiasAt(f, c)	=  0.0;
		}
	}
}


//----------------------------------------------------------------------
/**
 * Adam学習
 *
 * @param alpha    係数更新率
 * @param beta1    モーメント更新率
 * @param beta2    速度更新率
 * @param epsilon  誤差値
 */
//----------------------------------------------------------------------
void NeuralNet::ConvolutionLayer::LearnAdam(double alpha,
											double beta1,
											double beta2,
											double epsilon)
{
	// フィルタ数ループ
	for (unsigned int f = 0; f < m_FilterNum; ++f)
	{
		// チャンネル数ループ
		for (unsigned int c = 0; c < m_Channel; ++c)
		{
			// フィルタ計算.
			for (unsigned int i = 0; i < m_FilterSize; ++i)
			{
				for (unsigned int j = 0; j < m_FilterSize; ++j)
				{
					MomentFilterAt(i, j, f, c)	=
						beta1
					   *MomentFilterAt(i, j, f, c)
					   +(1.0-beta1)
					   *DeltaFilterAt(i, j, f, c);
					VelocityFilterAt(i, j, f, c)	=
						beta2
					   *VelocityFilterAt(i, j, f, c)
					   +(1.0-beta2)
					   *pow(DeltaFilterAt(i, j, f, c), 2);

					const double	m	= MomentFilterAt(  i, j, f, c)
										/ (1.0-beta1);
					const double	v	= VelocityFilterAt(i, j, f, c)
										/ (1.0-beta2);

					FilterAt(i, j, f, c)	+= alpha * m
											/ (sqrt(v) + epsilon);

					DeltaFilterAt(i, j, f, c)	= 0.0;
				}
			}

			MomentBiasAt(f, c)	=
						beta1
					   *MomentBiasAt(f, c)
					   +(1.0-beta1)
					   *DeltaBiasAt( f, c);
			VelocityBiasAt(f, c)	=
						beta2
					   *VelocityBiasAt(f, c)
					   +(1.0-beta2)
					   *pow(DeltaBiasAt(f, c), 2);

			const double	m	= MomentBiasAt(  f, c)
								/ (1.0-beta1);
			const double	v	= VelocityBiasAt(f, c)
								/ (1.0-beta2);

			BiasAt(f, c)		+= alpha * m
								/ (sqrt(v) + epsilon);
			DeltaBiasAt(f, c)	=  0.0;
		}
	}
}

//----------------------------------------------------------------------
/**
 * 前方出力
 *
 * @param  input  入力値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::MaxPoolingLayer::Forward(const std::vector<double> &input,
										 std::vector<double>       &output)
{
	if (input.size() != m_InputNum)
		return;
	
	output.resize(m_OutputNum);

	// チャンネル数ループ
	for (unsigned int c = 0; c < m_Channel; ++c)
	{
		// 横幅.
		for (unsigned int w = 0; w < m_WMax; ++w)
		{
			// 高さ
			for (unsigned int h = 0; h < m_HMax; ++h)
			{
				int	outputIndex	= OutputIndex(w, h, c);

				m_Mask[outputIndex]	= InputIndex(w*m_Stride,
												 h*m_Stride,
												 c);
				if (m_Mask[outputIndex] < m_InputNum)
				{
					output[outputIndex] =
						input[m_Mask[outputIndex]];
				}
				else {
					output[outputIndex] = 0.0;
				}

				// フィルタ計算.
				for (unsigned int i = 0; i < m_FilterSize; ++i)
				{
					int	iW	= w*m_Stride+i-m_Padding;

					if ((iW < 0) || (iW >= (int)m_Width))
						continue;

					for (unsigned int j = 0; j < m_FilterSize; ++j)
					{
						int	iH	= h*m_Stride+j-m_Padding;

						if ((iH < 0) || (iH >= (int)m_Height))
							continue;

						int	inputIndex	= InputIndex(iW, iH, c);

						if (input[inputIndex] > output[outputIndex])
						{
							output[outputIndex]	= input[inputIndex];
							m_Mask[outputIndex]	= inputIndex;
						}
					}
				}
			}
		}
	}
}

//----------------------------------------------------------------------
/**
 * 後方出力
 *
 * @param  delta  出力差分値配列
 * @param  output 出力値受取配列
 */
//----------------------------------------------------------------------
void NeuralNet::MaxPoolingLayer::Backward(const std::vector<double> &delta,
										  std::vector<double>       &output)
{
	if (delta.size() != m_OutputNum)
		return;

	output.resize(m_InputNum);

	// チャンネル数ループ
	for (unsigned int c = 0; c < m_Channel; ++c)
	{
		// 横幅.
		for (unsigned int w = 0; w < m_Width; ++w)
		{
			// 高さ
			for (unsigned int h = 0; h < m_Height; ++h)
			{
				int	inputIndex	= InputIndex(w, h, c);

				output[inputIndex]	= 0.0;
			}
		}
	}
	
	// チャンネル数ループ
	for (unsigned int c = 0; c < m_Channel; ++c)
	{
		// 横幅.
		for (unsigned int w = 0; w < m_WMax; ++w)
		{
			// 高さ
			for (unsigned int h = 0; h < m_HMax; ++h)
			{
				int outputIndex	= OutputIndex(w, h, c);

				output[m_Mask[outputIndex]]	+= delta[outputIndex];
			}
		}
	}
}

//----------------------------------------------------------------------
/**
 * コンストラクタ
 */
//----------------------------------------------------------------------
NeuralNet::NeuralNet()
{
}

//----------------------------------------------------------------------
/**
 * デストラクタ
 */
//----------------------------------------------------------------------
NeuralNet::~NeuralNet()
{
}

//----------------------------------------------------------------------
/**
 * 全結合層の追加
 *
 * @param inputNum  入力の数
 * @param outputNum 出力の数
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddAffineLayer(unsigned int inputNum,
							   unsigned int outputNum)
{
	m_Layer.push_back(std::shared_ptr<AffineLayer>
					  (new AffineLayer(inputNum, outputNum)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * ReLU層の追加
 *
 * @param inputNum  入力の数
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddReLULayer(unsigned int inputNum)
{
	m_Layer.push_back(std::shared_ptr<ReLULayer>
					  (new ReLULayer(inputNum)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * Randomized ReLU層の追加
 *
 * @param inputNum  入力の数
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddRReLULayer(unsigned int inputNum)
{
	m_Layer.push_back(std::shared_ptr<RReLULayer>
					  (new RReLULayer(inputNum)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * Leaky ReLU層の追加
 *
 * @param inputNum  入力の数
 * @param alpha     x<0 の傾き
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddLReLULayer(unsigned int inputNum, double alpha)
{
	m_Layer.push_back(std::shared_ptr<LReLULayer>
					  (new LReLULayer(inputNum, alpha)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * Sigmoid層の追加
 *
 * @param inputNum  入力の数
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddSigmoidLayer(unsigned int inputNum)
{
	m_Layer.push_back(std::shared_ptr<SigmoidLayer>
					  (new SigmoidLayer(inputNum)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * Soft-Max層の追加
 *
 * @param inputNum  入力の数
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddSoftMaxLayer(unsigned int inputNum)
{
	m_Layer.push_back(std::shared_ptr<SoftMaxLayer>
					  (new SoftMaxLayer(inputNum)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * 畳み込み層の追加
 * -width x height x chanel が入力になる
 * -((height+2*padding-filterSize)/stride+1)
 *  x((width+2*padding-filterSize)/stride+1)
 *  x filterNum が出力になる
 *
 * @param width      入力の横幅
 * @param height     入力の高さ
 * @param channel    入力のチャンネル数
 * @param filterSize フィルタのサイズ
 * @param filterNum  フィルタの数
 * @param stride     フィルタの移動幅
 * @param padding    パディングの幅
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddConvolutionLayer(unsigned int width,
									unsigned int height,
									unsigned int channel,
									unsigned int filterSize,
									unsigned int filterNum,
									unsigned int stride,
									unsigned int padding)
{
	m_Layer.push_back(std::shared_ptr<ConvolutionLayer>
					  (new ConvolutionLayer(width,
											height,
											channel,
											filterSize,
											filterNum,
											stride,
											padding)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * プーリング(最大値選出)層の追加
 * -width x height x chanel が入力になる
 * -((height+2*padding-filterSize)/stride+1)
 *  x((width+2*padding-filterSize)/stride+1)
 *  x channel が出力になる
 *
 * @param width      入力の横幅
 * @param height     入力の高さ
 * @param channel    入力のチャンネル数
 * @param filterSize フィルタのサイズ
 * @param stride     フィルタの移動幅
 * @param padding    パディングの幅
 *
 * @return           成否
 */
//----------------------------------------------------------------------
bool NeuralNet::AddMaxPoolingLayer(unsigned int width,
								   unsigned int height,
								   unsigned int channel,
								   unsigned int filterSize,
								   unsigned int stride,
								   unsigned int padding)
{
	m_Layer.push_back(std::shared_ptr<MaxPoolingLayer>
					  (new MaxPoolingLayer(width,
										   height,
										   channel,
										   filterSize,
										   stride,
										   padding)));

	return (CheckAddLayerConnect());
}

//----------------------------------------------------------------------
/**
 * 入力値の設定
 *
 * @param input      入力値の配列
 */
//----------------------------------------------------------------------
void NeuralNet::SetInput(const std::vector<double> &input)
{
	m_Input	= input;
}

//----------------------------------------------------------------------
/**
 * 出力値の取得
 *
 * @param output   出力値の配列
 */
//----------------------------------------------------------------------
void NeuralNet::GetOutput(std::vector<double> &output) const
{
	if (m_Layer.size() == 0)
		return;

	output	= m_Output;
}

//----------------------------------------------------------------------
/**
 * 損失値の計算
 *
 * @param teacher   教師信号の配列
 *
 * @return          二乗誤差の総和
 */
//----------------------------------------------------------------------
double NeuralNet::CalcSquareLoss(const std::vector<double> &teacher)
{
	double	lossSum	= 0.0;
	
	m_Loss.resize(teacher.size());
	
	for (unsigned int i = 0; i < teacher.size(); ++i)
	{
		m_Loss[i]	= teacher[i] - m_Output[i];

		lossSum		+= m_Loss[i] * m_Loss[i];
		//lossSum		+= -teacher[i] * log(m_Output[i] + 1.0e-7);
	}

	return (lossSum);
}

//----------------------------------------------------------------------
/**
* 損失値の計算
*
* @param teacher   教師信号の配列
*
* @return          クロスエントロピー誤差の総和
*/
//----------------------------------------------------------------------
double NeuralNet::CalcCrossEntropyLoss(const std::vector<double> &teacher)
{
	double	lossSum = 0.0;

	m_Loss.resize(teacher.size());

	for (unsigned int i = 0; i < teacher.size(); ++i)
	{
		m_Loss[i] = teacher[i] - m_Output[i];

		lossSum		+= -teacher[i] * log(m_Output[i] + 1.0e-7);
	}

	return (lossSum);
}

//----------------------------------------------------------------------
/**
 * 前方出力
 */
//----------------------------------------------------------------------
void NeuralNet::Forward(void)
{
	std::vector<double>	tmp[2];

	tmp[0]	= m_Input;
	for (unsigned int i = 0; i < m_Layer.size(); ++i)
	{ 
		m_Layer[i]->Forward(tmp[i&1], tmp[(i+1)&1]);
		for (auto v : tmp[(i+1)&1])
		{ 
			if (isnan(v))
				v = 0.0;
		}
	}
	m_Output	= tmp[m_Layer.size()&1];
}

//----------------------------------------------------------------------
/**
 * 後方出力
 */
//----------------------------------------------------------------------
void NeuralNet::Backward(void)
{
	std::vector<double>	tmp[2];
	
	tmp[0]	= m_Loss;
	for(unsigned int i = 0, index = m_Layer.size()-1;
		i < m_Layer.size();
		++i, --index)
		m_Layer[index]->Backward(tmp[i&1], tmp[(i+1)&1]);

	m_Input	= tmp[m_Layer.size()&1];
}

//----------------------------------------------------------------------
/**
 * 学習
 *
 * @param learnRatio  学習率
 */
//----------------------------------------------------------------------
void NeuralNet::Learn(double learnRatio)
{
	if (m_Layer.size() == 0)
		return ;

	for (unsigned int i = 0; i < m_Layer.size(); ++i)
		m_Layer[i]->Learn(learnRatio);
}

//----------------------------------------------------------------------
/**
 * Adam学習
 *
 * @param alpha    係数更新率
 * @param beta1    モーメント更新率
 * @param beta2    速度更新率
 * @param epsilon  誤差値
 */
//----------------------------------------------------------------------
void NeuralNet::LearnAdam(double alpha,
						  double beta1,
						  double beta2,
						  double epsilon)
{
	if (m_Layer.size() == 0)
		return ;

	for (unsigned int i = 0; i < m_Layer.size(); ++i)
		m_Layer[i]->LearnAdam(alpha, beta1, beta2, epsilon);
}

//----------------------------------------------------------------------
/**
 * Adamパラメータリセット
 */
//----------------------------------------------------------------------
void NeuralNet::LearnAdamReset(void)
{
	if (m_Layer.size() == 0)
		return ;

	for (unsigned int i = 0; i < m_Layer.size(); ++i)
		m_Layer[i]->LearnAdamReset();
}

//----------------------------------------------------------------------
/**
 * Deltaの正規化
 */
//----------------------------------------------------------------------
void NeuralNet::DeltaNormalize(void)
{
	if (m_Layer.size() == 0)
		return ;

	for (unsigned int i = 0; i < m_Layer.size(); ++i)
		m_Layer[i]->DeltaNormalize();
}

//----------------------------------------------------------------------
/**
 * 保存
 *
 * @param  data  バイナリ配列
 */
//----------------------------------------------------------------------
void NeuralNet::Save(std::vector<char> &data)
{
	for (unsigned int i = 0; i < m_Layer.size(); ++i)
	{
		unsigned int	type	= m_Layer[i]->GetType();

		WriteIntData(data, type);

		switch (type)
		{
		  case Affine:
			{
				const std::shared_ptr<AffineLayer>	pAffineLayer	=
					std::dynamic_pointer_cast<AffineLayer>(m_Layer[i]);

				WriteIntData(data, pAffineLayer->GetInputNum());
				WriteIntData(data, pAffineLayer->GetOutputNum());

				for (unsigned int o = 0;
					 o < pAffineLayer->GetOutputNum();
					 ++o)
				{
					for (unsigned int i = 0;
						 i < pAffineLayer->GetInputNum();
						 ++i)
						WriteDoubleData(data, pAffineLayer->GetWeight(i, o));

					WriteDoubleData(data, pAffineLayer->GetBias(o));
				}
			}
			break;
			
		  case ReLU:
		  case Sigmoid:
		  case SoftMax:
			{
				const std::shared_ptr<Layer>	pLayer	=
					std::dynamic_pointer_cast<Layer>(m_Layer[i]);

				WriteIntData(data, pLayer->GetInputNum());
			}
			break;
			 
		  case RReLU:
			{
				const std::shared_ptr<RReLULayer>	pRReLULayer	=
					std::dynamic_pointer_cast<RReLULayer>(m_Layer[i]);

				WriteIntData(data, pRReLULayer->GetInputNum());

				for (unsigned int i = 0;
					 i < pRReLULayer->GetInputNum();
					 ++i)
				{
					WriteDoubleData(data, pRReLULayer->GetAlpha(i));
				}
			}
			break;
			
		  case LReLU:
			{
				const std::shared_ptr<LReLULayer>	pLReLULayer	=
					std::dynamic_pointer_cast<LReLULayer>(m_Layer[i]);

				WriteIntData(data, pLReLULayer->GetInputNum());
				WriteDoubleData(data, pLReLULayer->GetAlpha());
			}
			break;
			
		  case Convolution:
			{
				const std::shared_ptr<ConvolutionLayer>	pConvLayer	=
					std::dynamic_pointer_cast<ConvolutionLayer>(m_Layer[i]);

				WriteIntData(data, pConvLayer->GetWidth());
				WriteIntData(data, pConvLayer->GetHeight());
				WriteIntData(data, pConvLayer->GetChannel());
				WriteIntData(data, pConvLayer->GetFilterSize());
				WriteIntData(data, pConvLayer->GetFilterNum());
				WriteIntData(data, pConvLayer->GetStride());
				WriteIntData(data, pConvLayer->GetPadding());

				// filter & bias
				for (unsigned int c = 0;
					 c < pConvLayer->GetChannel();
					 ++c)
				{
					for (unsigned int f = 0;
						 f < pConvLayer->GetFilterNum();
						 ++f)
					{
						for (unsigned int x = 0;
							 x < pConvLayer->GetFilterSize();
							 ++x)
						{
							for (unsigned int y = 0;
								 y < pConvLayer->GetFilterSize();
								 ++y)
							{
								WriteDoubleData(data,
												pConvLayer->GetFilter(x, y, f, c));
							}
						}
						
						WriteDoubleData(data, pConvLayer->GetBias(f, c));
					}
				}
			}
			break;

		  case MaxPooling:
			{
				const std::shared_ptr<PoolingLayer>	pPoolLayer	=
					std::dynamic_pointer_cast<PoolingLayer>(m_Layer[i]);

				WriteIntData(data, pPoolLayer->GetWidth());
				WriteIntData(data, pPoolLayer->GetHeight());
				WriteIntData(data, pPoolLayer->GetChannel());
				WriteIntData(data, pPoolLayer->GetFilterSize());
				WriteIntData(data, pPoolLayer->GetStride());
				WriteIntData(data, pPoolLayer->GetPadding());
			}
			break;
		}
	}
	// ターミネータ.
	WriteIntData(data, LayerType::Blank);
}

//----------------------------------------------------------------------
/**
 * 読み込み
 *
 * @param  data  バイナリ配列
 */
//----------------------------------------------------------------------
void NeuralNet::Load(const std::vector<char> &data)
{
	unsigned int	type;
	unsigned int	index = 0;

	while ((type = ReadIntData(data, index)) != LayerType::Blank)
	{
		switch (type)
		{
		  case Affine:
			{
				unsigned int inputNum	= ReadIntData(data, index);
				unsigned int outputNum	= ReadIntData(data, index);

				AddAffineLayer(inputNum, outputNum);
				std::shared_ptr<AffineLayer>	pAffineLayer	=
					std::dynamic_pointer_cast<AffineLayer>
						(m_Layer[m_Layer.size()-1]);

				// weight & bias
				for (unsigned int o = 0;
					 o < pAffineLayer->GetOutputNum();
					 ++o)
				{
					for (unsigned int i = 0;
						 i < pAffineLayer->GetInputNum();
						 ++i)
					{
						pAffineLayer->SetWeight(i,
												o,
												ReadDoubleData(data, index));
					}
					
					pAffineLayer->SetBias(o, ReadDoubleData(data, index));
				}
			}
			break;
			
		  case ReLU:
			{
				unsigned int inputNum	= ReadIntData(data, index);

				AddReLULayer(inputNum);
			}
			break;

		  case RReLU:
			{
				unsigned int inputNum	= ReadIntData(data, index);

				AddRReLULayer(inputNum);

				std::shared_ptr<RReLULayer>	pRReLULayer	=
					std::dynamic_pointer_cast<RReLULayer>
						(m_Layer[m_Layer.size()-1]);

				for (unsigned int i = 0;
					 i < pRReLULayer->GetInputNum();
					 ++i)
				{
					pRReLULayer->SetAlpha(i, ReadDoubleData(data, index));
				}
			}
			break;
			
		  case Sigmoid:
			{
				unsigned int inputNum	= ReadIntData(data, index);

				AddSigmoidLayer(inputNum);
			}
			break;

		  case LReLU:
			{
				unsigned int inputNum	= ReadIntData(   data, index);
				double       alpha		= ReadDoubleData(data, index);

				AddLReLULayer(inputNum, alpha);
			}
			break;

		  case SoftMax:
			{
				unsigned int inputNum	= ReadIntData(data, index);

				AddSoftMaxLayer(inputNum);
			}
			break;

		  case Convolution:
			{
				unsigned int width		= ReadIntData(data, index);
				unsigned int height		= ReadIntData(data, index);
				unsigned int channel	= ReadIntData(data, index);
				unsigned int filterSize	= ReadIntData(data, index);
				unsigned int filterNum	= ReadIntData(data, index);
				unsigned int stride		= ReadIntData(data, index);
				unsigned int padding	= ReadIntData(data, index);
				
				AddConvolutionLayer(width,
									height,
									channel,
									filterSize,
									filterNum,
									stride,
									padding);
				
				std::shared_ptr<ConvolutionLayer>	pConvLayer	=
					std::dynamic_pointer_cast<ConvolutionLayer>(m_Layer[m_Layer.size()-1]);

				// filter & bias
				for (unsigned int c = 0;
					 c < pConvLayer->GetChannel();
					 ++c)
				{
					for (unsigned int f = 0;
						 f < pConvLayer->GetFilterNum();
						 ++f)
					{
						for (unsigned int x = 0;
							 x < pConvLayer->GetFilterSize();
							 ++x)
						{
							for (unsigned int y = 0;
								 y < pConvLayer->GetFilterSize();
								 ++y)
							{
								pConvLayer->SetFilter(
									x,
									y,
									f,
									c,
									ReadDoubleData(data, index));
							}
						}
						
						pConvLayer->SetBias(f, c, ReadDoubleData(data, index));
					}
				}
			}
			break;

		  case MaxPooling:
			{
				unsigned int width		= ReadIntData(data, index);
				unsigned int height		= ReadIntData(data, index);
				unsigned int channel	= ReadIntData(data, index);
				unsigned int filterSize	= ReadIntData(data, index);
				unsigned int stride		= ReadIntData(data, index);
				unsigned int padding	= ReadIntData(data, index);

				AddMaxPoolingLayer(width,
								   height,
								   channel,
								   filterSize,
								   stride,
								   padding);
			}
			break;
		}
	}
}

