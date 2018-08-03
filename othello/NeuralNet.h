﻿/* -*- mode:c++; coding:utf-8-ws-dos; tab-width:4 -*- ==================== */
/* -----------------------------------------------------------------------
 * $Id: NeuralNet.h 2720 2018-01-02 21:20:54+09:00 nowatari $
 * ======================================================================= */

#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <math.h>
#include <memory>
#include <vector>

class NeuralNet
{
  private:
	typedef enum LayerType
	{
		Blank		= 0x00000000UL,

		Affine		= 0x10000000UL,

		ReLU		= 0x20000000UL,
		RReLU		= 0x20000001UL,
		LReLU		= 0x20000002UL,
		
		Sigmoid		= 0x30000000UL,

		SoftMax		= 0x40000000UL,

		Convolution	= 0x50000000UL,

		MaxPooling	= 0x60000000UL,
	} LayerType;
	//----------------------------------------------------------------------
	/// レイヤー基底クラス
	class Layer
	{
	  public:
		Layer(unsigned int inputNum,
			  unsigned int outputNum,
			  unsigned int type) :
		m_InputNum(inputNum),
		m_OutputNum(outputNum),
		m_Type(type)
		{}
		virtual ~Layer() {}

		virtual void Forward(const std::vector<double> &input,
							 std::vector<double>       &output) = 0;
		virtual void Backward(const std::vector<double> &delta,
							  std::vector<double>       &output) = 0;
		virtual void Learn(double learnRatio) {}
		virtual void LearnAdam(double alpha,
							   double beta1,
							   double beta2,
							   double epsilon) {}
		virtual void LearnAdamReset(void) {}
		virtual void DeltaNormalize(void) {}
		
		void         SetType(unsigned int type) {m_Type	= type;}
		unsigned int GetType(void) const        {return (m_Type);}

		unsigned int GetInputNum( void) const   {return (m_InputNum);}
		unsigned int GetOutputNum(void) const   {return (m_OutputNum);}
		
	  protected:
		unsigned int	m_Type;
		unsigned int	m_InputNum;
		unsigned int	m_OutputNum;
	};

	//----------------------------------------------------------------------
	// 全結合層.
	class AffineLayer : public Layer
	{
	  public:
		AffineLayer(unsigned int inputNum,
					unsigned int outputNum);
		~AffineLayer() {}

		void Forward(const std::vector<double> &input,
					 std::vector<double>       &output);
		void Backward(const std::vector<double> &delta,
					  std::vector<double>       &output);
		void Learn(double learnRatio);
		void LearnAdam(double alpha,
					   double beta1,
					   double beta2,
					   double epsilon);
		void LearnAdamReset(void)
		{
			for (unsigned int o = 0; o < m_OutputNum; ++o)
			{
				for (unsigned int i = 0; i < m_InputNum; ++i)
				{
					MomentWeightAt(  i, o)	= 0.0;
					VelocityWeightAt(i, o)	= 0.0;
				}
				MomentBiasAt(  o)	= 0.0;
				VelocityBiasAt(o)	= 0.0;
			}
		}
		void DeltaNormalize(void)
		{
			double	total;

			total	= 0.0;
			for (unsigned int i = 0; i < m_DeltaWeight.size(); ++i)
				total	+= m_DeltaWeight[i] * m_DeltaWeight[i];

			if (total > 0.0)
			{
				for (unsigned int i = 0; i < m_DeltaWeight.size(); ++i)
					m_DeltaWeight[i]	/= total;
			}

			total	= 0.0;
			for (unsigned int i = 0; i < m_DeltaBias.size(); ++i)
				total	+= m_DeltaBias[i] * m_DeltaBias[i];

			if (total > 0.0)
			{
				for (unsigned int i = 0; i < m_DeltaBias.size(); ++i)
					m_DeltaBias[i]	/= total;
			}
		}
		
		double GetWeight(unsigned int i, unsigned int o) const
		{
			return (WeightAt(i, o));
		}
		void   SetWeight(unsigned int i, unsigned int o, double w)
		{
			WeightAt(i, o)	= w;
		}
		double GetBias(unsigned int o) const
		{
			return (BiasAt(o));
		}
		void   SetBias(unsigned int o, double b)
		{
			BiasAt(o)	= b;
		}

	  private:
		std::vector<double>	m_Input;
		std::vector<double>	m_Weight;
		std::vector<double>	m_Bias;
		std::vector<double>	m_MomentWeight;
		std::vector<double>	m_MomentBias;
		std::vector<double>	m_VelocityWeight;
		std::vector<double>	m_VelocityBias;
		std::vector<double>	m_DeltaWeight;
		std::vector<double>	m_DeltaBias;

		unsigned int WeightIndex(unsigned int i,
								 unsigned int o) const
		{
			return (o*m_InputNum+i);
		}
		double &InputAt(unsigned int i)
		{
			return (m_Input[i]);
		}
		const double &InputAt(unsigned int i) const
		{
			return (m_Input[i]);
		}
		double &WeightAt(unsigned int i, unsigned int o)
		{
			return (m_Weight[WeightIndex(i, o)]);
		}
		const double &WeightAt(unsigned int i, unsigned int o) const
		{
			return (m_Weight[WeightIndex(i, o)]);
		}
		double &BiasAt(unsigned int o)
		{
			return (m_Bias[o]);
		}
		const double &BiasAt(unsigned int o) const
		{
			return (m_Bias[o]);
		}
		double &MomentWeightAt(unsigned int i, unsigned int o)
		{
			return (m_MomentWeight[WeightIndex(i, o)]);
		}
		const double &MomentWeightAt(unsigned int i, unsigned int o) const
		{
			return (m_MomentWeight[WeightIndex(i, o)]);
		}
		double &VelocityWeightAt(unsigned int i, unsigned int o)
		{
			return (m_VelocityWeight[WeightIndex(i, o)]);
		}
		const double &VelocityWeightAt(unsigned int i, unsigned int o) const
		{
			return (m_VelocityWeight[WeightIndex(i, o)]);
		}
		double &MomentBiasAt(unsigned int o)
		{
			return (m_MomentBias[o]);
		}
		const double &MomentBiasAt(unsigned int o) const
		{
			return (m_MomentBias[o]);
		}
		double &VelocityBiasAt(unsigned int o)
		{
			return (m_VelocityBias[o]);
		}
		const double &VelocityBiasAt(unsigned int o) const
		{
			return (m_VelocityBias[o]);
		}
		double &DeltaWeightAt(unsigned i, unsigned o)
		{
			return (m_DeltaWeight[WeightIndex(i, o)]);
		}
		const double &DeltaWeightAt(unsigned i, unsigned o) const
		{
			return (m_DeltaWeight[WeightIndex(i, o)]);
		}
		double &DeltaBiasAt(unsigned int o)
		{
			return (m_DeltaBias[o]);
		}
		const double &DeltaBiasAt(unsigned int o) const
		{
			return (m_DeltaBias[o]);
		}
	};
	//----------------------------------------------------------------------
	/// 活性化層
	class ActivateLayer : public Layer
	{
	  public:
		ActivateLayer(unsigned int inputNum,
					  unsigned int type);
		virtual ~ActivateLayer(void)
		{}
		
		void Forward(const std::vector<double> &input,
					 std::vector<double>       &output);
		void Backward(const std::vector<double> &delta,
					  std::vector<double>       &output);

	  public:
		virtual double ForwardFunc( double x, unsigned int index) = 0;
		virtual double BackwardFunc(double x, unsigned int index) = 0;
	};
	//----------------------------------------------------------------------
	/// ReLU層
	class ReLULayer : public ActivateLayer
	{
	  public:
		ReLULayer(unsigned int inputNum) :
		ActivateLayer(inputNum, LayerType::ReLU)
		{
			m_Mask.resize(inputNum);
		}
		ReLULayer(unsigned int inputNum, LayerType type) :
		ActivateLayer(inputNum, type)
		{
			m_Mask.resize(inputNum);
		}
		virtual ~ReLULayer() {}

	  protected:
		double ForwardFunc(double x, unsigned int index)
		{
			if (x < 0.0)
			{
				m_Mask[index]	= false;
				return (0.0);
			}

			m_Mask[index]	= true;
			return (x);
		}
		double BackwardFunc(double x, unsigned int index)
		{
			return (m_Mask[index] ? x : 0.0);
		}
		std::vector<bool>	m_Mask;
	};
	//----------------------------------------------------------------------
	/// Randomized ReLU層
	class RReLULayer : public ReLULayer
	{
	  public:
		RReLULayer(unsigned int inputNum);
		~RReLULayer() {}

		void Learn(double learnRatio);
		void LearnAdam(double alpha,
					   double beta1,
					   double beta2,
					   double epsilon)
		{
			Learn(alpha);
		}
		double GetAlpha(unsigned int index) const {return (m_Alpha[index]);}
		void   SetAlpha(unsigned int index, double alpha)
		{
			m_Alpha[index]	= alpha;
		}
	  private:
		double	GetRandomAlpha(void);
		
	  protected:
		double ForwardFunc(double x, unsigned int index)
		{
			if (x < 0.0)
			{
				m_Mask[index]	= false;
				return (x * m_Alpha[index]);
			}

			m_Mask[index]	= true;
			return (x);
		}
		double BackwardFunc(double x, unsigned int index)
		{
			return (m_Mask[index] ? x : x * m_Alpha[index]);
		}
		std::vector<double>	m_Alpha;
	};
	//----------------------------------------------------------------------
	/// Leaky ReLU層
	class LReLULayer : public ReLULayer
	{
	  public:
		LReLULayer(unsigned int inputNum, double alpha) :
		ReLULayer(inputNum, LayerType::LReLU),
		m_Alpha(alpha)
		{}
		~LReLULayer() {}

		double GetAlpha(void) const   {return (m_Alpha);}
		void   SetAlpha(double alpha) {m_Alpha	= alpha;}
	
	  protected:
		double ForwardFunc(double x, unsigned int index)
		{
			if (x < 0.0)
			{
				m_Mask[index]	= false;
				return (x * m_Alpha);
			}

			m_Mask[index]	= true;
			return (x);
		}
		double BackwardFunc(double x, unsigned int index)
		{
			return (m_Mask[index] ? x : x * m_Alpha);
		}
		double	m_Alpha;
	};
	
	//----------------------------------------------------------------------
	/// Sigmoid層
	class SigmoidLayer : public ActivateLayer
	{
	  public:
		SigmoidLayer(unsigned int inputNum) :
		ActivateLayer(inputNum, LayerType::Sigmoid)
		{
			m_Mask.resize(inputNum);
		}
		~SigmoidLayer() {}

	  protected:
		double ForwardFunc(double x, unsigned int index)
		{
			m_Mask[index]	= 1.0 / (1.0 + exp(-x));
			return (m_Mask[index]);
		}
		double BackwardFunc(double x, unsigned int index)
		{
			return (x * (1.0 - m_Mask[index]) * m_Mask[index]);
		}
	  private:
		std::vector<double>	m_Mask;
	};

	//----------------------------------------------------------------------
	/// Soft-Max層
	class SoftMaxLayer : public Layer
	{
	  public:
		SoftMaxLayer(unsigned int inputNum) :
		Layer(inputNum, inputNum, LayerType::SoftMax)
		{}
		~SoftMaxLayer() {}

		void Forward(const std::vector<double> &input,
					 std::vector<double>       &output);
		void Backward(const std::vector<double> &delta,
					  std::vector<double>       &output);
	};

	//----------------------------------------------------------------------
	/// フィルタ層基底クラス
	class FilterLayer : public Layer
	{
	  private:
		const double	m_PaddingValue	= 0.0;

	  public:
		FilterLayer(unsigned int width,
					unsigned int height,
					unsigned int channel,
					unsigned int filterSize,
					unsigned int filterNum,
					unsigned int stride,
					unsigned int padding,
					unsigned int type) :
		m_Width(     width),
		m_Height(    height),
		m_Channel(   channel),
		m_FilterSize(filterSize),
		m_FilterNum( filterNum),
		m_Stride(    stride),
		m_Padding(   padding),
		m_WMax(( width+2*padding-filterSize)/stride + 1),
		m_HMax((height+2*padding-filterSize)/stride + 1),
		Layer(channel * width * height, 
			  filterNum
			 *(( width+2*padding-filterSize)/stride + 1)
			 *((height+2*padding-filterSize)/stride + 1),
			  type)
		{}
		virtual ~FilterLayer() {}

		unsigned int GetWidth(     void) const {return (m_Width);}
		unsigned int GetHeight(    void) const {return (m_Height);}
		unsigned int GetChannel(   void) const {return (m_Channel);}
		unsigned int GetFilterSize(void) const {return (m_FilterSize);}
		unsigned int GetFilterNum( void) const {return (m_FilterNum);}
		unsigned int GetStride(    void) const {return (m_Stride);}
		unsigned int GetPadding(   void) const {return (m_Padding);}
		
	  protected:
		unsigned int	m_Width;
		unsigned int	m_Height;
		unsigned int	m_Channel;
		unsigned int	m_FilterSize;
		unsigned int	m_FilterNum;
		unsigned int	m_Stride;
		unsigned int	m_Padding;
		unsigned int	m_WMax;
		unsigned int	m_HMax;

		int InputIndex(unsigned int w,
					   unsigned int h,
					   unsigned int c) const
		{
			return (c*(m_Height*m_Width)
				   +w*(m_Height)
				   +h);
		}
		int OutputIndex(unsigned int w,
						unsigned int h,
						unsigned int f) const
		{
			return (f*(m_HMax*m_WMax)
				   +w*(m_HMax)
				   +h);
		}
		int FilterIndex(unsigned int x,
						unsigned int y,
						unsigned int f,
						unsigned int c) const
		{
			return (c*(m_FilterNum*m_FilterSize*m_FilterSize)
				   +f*(m_FilterSize*m_FilterSize)
				   +x*(m_FilterSize)
				   +y);
		}
		int FilterBackIndex(unsigned int x,
							unsigned int y,
							unsigned int f,
							unsigned int c) const
		{
			return (FilterIndex(m_FilterSize-x-1,
								m_FilterSize-y-1,
								f,
								c));
		}
	};

	//----------------------------------------------------------------------
	/// 畳み込み層
	class ConvolutionLayer : public FilterLayer
	{
	  public:
		ConvolutionLayer(unsigned int width,
						 unsigned int height,
						 unsigned int channel,
						 unsigned int filterSize,
						 unsigned int filterNum,
						 unsigned int stride,
						 unsigned int padding);
		~ConvolutionLayer() {}

		void Forward(const std::vector<double> &input,
					 std::vector<double>       &output);
		void Backward(const std::vector<double> &delta,
					  std::vector<double>       &output);
		void Learn(double learnRatio);
		void LearnAdam(double alpha,
					   double beta1,
					   double beta2,
					   double epsilon);
		void LearnAdamReset(void)
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
							MomentFilterAt(  i, j, f, c)	= 0.0;
							VelocityFilterAt(i, j, f, c)	= 0.0;
						}
					}
					MomentBiasAt(  f, c)	= 0.0;
					VelocityBiasAt(f, c)	= 0.0;
				}
			}
		}
		void DeltaNormalize(void)
		{
			double	total;

			total	= 0.0;
			for (unsigned int i = 0; i < m_DeltaFilter.size(); ++i)
				total	+= m_DeltaFilter[i] * m_DeltaFilter[i];

			if (total > 0.0)
			{
				for (unsigned int i = 0; i < m_DeltaFilter.size(); ++i)
					m_DeltaFilter[i]	/= total;
			}

			total	= 0.0;
			for (unsigned int i = 0; i < m_DeltaBias.size(); ++i)
				total	+= m_DeltaBias[i] * m_DeltaBias[i];

			if (total > 0.0)
			{
				for (unsigned int i = 0; i < m_DeltaBias.size(); ++i)
					m_DeltaBias[i]	/= total;
			}
		}

		double GetFilter(unsigned int x,
						 unsigned int y,
						 unsigned int f,
						 unsigned int c) const
		{
			return (FilterAt(x, y, f, c));
		}
		void   SetFilter(unsigned int x,
						 unsigned int y,
						 unsigned int f,
						 unsigned int c,
						 double       v)
		{
			FilterAt(x, y, f, c)	= v;
		}
		double GetBias(unsigned int f, unsigned int c) const
		{
			return (BiasAt(f, c));
		}
		void   SetBias(unsigned int f, unsigned int c, double b)
		{
			BiasAt(f, c)	= b;
		}
		
	  protected:
		std::vector<double>	m_Input;
		std::vector<double>	m_Filter;
		std::vector<double>	m_Bias;
		std::vector<double>	m_MomentFilter;
		std::vector<double>	m_VelocityFilter;
		std::vector<double>	m_MomentBias;
		std::vector<double>	m_VelocityBias;
		std::vector<double>	m_DeltaFilter;
		std::vector<double>	m_DeltaBias;

		unsigned int BiasIndex(unsigned int f,
							   unsigned int c) const
		{
			return (f*m_Channel+c);
		}
		double &InputAt(unsigned int i)
		{
			return (m_Input[i]);
		}
		const double &InputAt(unsigned int i) const
		{
			return (m_Input[i]);
		}
		double &InputAt(unsigned int w,
						unsigned int h,
						unsigned int c)
		{
			return (m_Input[InputIndex(w, h, c)]);
		}
		const double &InputAt(unsigned int w,
							  unsigned int h,
							  unsigned int c) const
		{
			return (m_Input[InputIndex(w, h, c)]);
		}
		double &FilterAt(unsigned int x,
						 unsigned int y,
						 unsigned int f,
						 unsigned int c)
		{
			return (m_Filter[FilterIndex(x, y, f, c)]);
		}
		const double &FilterAt(unsigned int x,
							   unsigned int y,
							   unsigned int f,
							   unsigned int c) const
		{
			return (m_Filter[FilterIndex(x, y, f, c)]);
		}
		double &FilterBackAt(unsigned int x,
							 unsigned int y,
							 unsigned int f,
							 unsigned int c)
		{
			return (m_Filter[FilterBackIndex(x, y, f, c)]);
		}
		const double &FilterBackAt(unsigned int x,
								   unsigned int y,
								   unsigned int f,
								   unsigned int c) const
		{
			return (m_Filter[FilterBackIndex(x, y, f, c)]);
		}
		double &BiasAt(unsigned int f,
					   unsigned int c)
		{
			return (m_Bias[BiasIndex(f, c)]);
		}
		const double &BiasAt(unsigned int f,
							 unsigned int c) const
		{
			return (m_Bias[BiasIndex(f, c)]);
		}
		double &MomentFilterAt(unsigned int x,
							   unsigned int y,
							   unsigned int f,
							   unsigned int c)
		{
			return (m_MomentFilter[FilterIndex(x, y, f, c)]);
		}
		const double &MomentFilterAt(unsigned int x,
									 unsigned int y,
									 unsigned int f,
									 unsigned int c) const
		{
			return (m_MomentFilter[FilterIndex(x, y, f, c)]);
		}
		double &VelocityFilterAt(unsigned int x,
								 unsigned int y,
								 unsigned int f,
								 unsigned int c)
		{
			return (m_VelocityFilter[FilterIndex(x, y, f, c)]);
		}
		const double &VelocityFilterAt(unsigned int x,
									   unsigned int y,
									   unsigned int f,
									   unsigned int c) const
		{
			return (m_VelocityFilter[FilterIndex(x, y, f, c)]);
		}
		double &MomentBiasAt(unsigned int f,
							unsigned int c)
		{
			return (m_MomentBias[BiasIndex(f, c)]);
		}
		const double &MomentBiasAt(unsigned int f,
								  unsigned int c) const
		{
			return (m_MomentBias[BiasIndex(f, c)]);
		}
		double &VelocityBiasAt(unsigned int f,
							   unsigned int c)
		{
			return (m_VelocityBias[BiasIndex(f, c)]);
		}
		const double &VelocityBiasAt(unsigned int f,
									 unsigned int c) const
		{
			return (m_VelocityBias[BiasIndex(f, c)]);
		}
		double &DeltaFilterAt(unsigned int x,
							  unsigned int y,
							  unsigned int f,
							  unsigned int c)
		{
			return (m_DeltaFilter[FilterIndex(x, y, f, c)]);
		}
		const double &DeltaFilterAt(unsigned int x,
									unsigned int y,
									unsigned int f,
									unsigned int c) const
		{
			return (m_DeltaFilter[FilterIndex(x, y, f, c)]);
		}
		double &DeltaFilterBackAt(unsigned int x,
								  unsigned int y,
								  unsigned int f,
								  unsigned int c)
		{
			return (m_DeltaFilter[FilterBackIndex(x, y, f, c)]);
		}
		const double &DeltaFilterBackAt(unsigned int x,
										unsigned int y,
										unsigned int f,
										unsigned int c) const
		{
			return (m_DeltaFilter[FilterBackIndex(x, y, f, c)]);
		}
		double &DeltaBiasAt(unsigned int f,
							unsigned int c)
		{
			return (m_DeltaBias[BiasIndex(f, c)]);
		}
		const double &DeltaBiasAt(unsigned int f,
								  unsigned int c) const
		{
			return (m_DeltaBias[BiasIndex(f, c)]);
		}
	};
	//----------------------------------------------------------------------
	/// プーリング層基底クラス
	class PoolingLayer : public FilterLayer
	{
	  public:
		PoolingLayer(unsigned int width,
					 unsigned int height,
					 unsigned int channel,
					 unsigned int filterSize,
					 unsigned int stride,
					 unsigned int padding,
					 unsigned int type) :
		FilterLayer(width,
					height,
					channel,
					filterSize,
					channel,
					stride,
					padding,
					type)
		{}
		virtual ~PoolingLayer() {};
	};
	//----------------------------------------------------------------------
	/// 最大値プーリング層
	class MaxPoolingLayer : public PoolingLayer
	{
	  public:
		MaxPoolingLayer(unsigned int width,
						unsigned int height,
						unsigned int channel,
						unsigned int filterSize,
						unsigned int stride,
						unsigned int padding) :
		PoolingLayer(width,
					 height,
					 channel,
					 filterSize,
					 stride,
					 padding,
					 LayerType::MaxPooling)
		{
			m_Mask.resize(m_OutputNum);
		}
		~MaxPoolingLayer() {}

		void Forward(const std::vector<double> &input,
					 std::vector<double>       &output);
		void Backward(const std::vector<double> &delta,
					  std::vector<double>       &output);

	  private:
		std::vector<unsigned int>	m_Mask;
	};

	//----------------------------------------------------------------------
	
	// レイヤー配列
	std::vector<std::shared_ptr<Layer>>	m_Layer;

	// 入力値.
	std::vector<double>	m_Input;

	// 出力値.
	std::vector<double>	m_Output;

	// 損失値.
	std::vector<double>	m_Loss;
	
	static void WriteIntData(std::vector<char> &data,
							 unsigned int      value)
	{
		char *pChar	= (char *)&value;

		for (int i = 0; i < sizeof(value); ++i)
			data.push_back(*pChar++);
	}
	static void WriteDoubleData(std::vector<char> &data,
								double            value)
	{
		char *pChar	= (char *)&value;

		for (int i = 0; i < sizeof(value); ++i)
			data.push_back(*pChar++);
	}
	static unsigned int ReadIntData(const std::vector<char> &data,
									unsigned int            &index)
	{
		unsigned int	value	= 0;
		char	*pChar	= (char *)&value;

		for (int i = 0; i < sizeof(value); ++i)
			pChar[i]	=  data[index+i];

		index	+= sizeof(value);

		return (value);
	}
	static double ReadDoubleData(const std::vector<char> &data,
								 unsigned int            &index)
	{
		double	value;
		char	*pChar	= (char *)&value;

		for (int i = 0; i < sizeof(value); ++i)
			pChar[i]	=  data[index+i];

		index	+= sizeof(value);

		return (value);
	}
	bool CheckAddLayerConnect(void)
	{
		if (m_Layer.size() < 2)
			return (true);

		if (m_Layer[m_Layer.size()-2]->GetOutputNum()
		!=  m_Layer[m_Layer.size()-1]->GetInputNum())
		{
			m_Layer.pop_back();
			
			return (false);
		}
		
		return (true);
	}
	
  public:
	NeuralNet();
	~NeuralNet();

	bool   AddAffineLayer(unsigned int inputNum,
						  unsigned int outputNum);

	bool   AddReLULayer( unsigned int inputNum);
	bool   AddRReLULayer(unsigned int inputNum);
	bool   AddLReLULayer(unsigned int inputNum, double alpha);
	
	bool   AddSigmoidLayer(unsigned int inputNum);
	
	bool   AddSoftMaxLayer(unsigned int inputNum);

	bool   AddConvolutionLayer(unsigned int width,
							   unsigned int height,
							   unsigned int channel,
							   unsigned int filterSize,
							   unsigned int filterNum,
							   unsigned int stride,
							   unsigned int padding);

	bool   AddMaxPoolingLayer(unsigned int width,
							  unsigned int height,
							  unsigned int channel,
							  unsigned int filterSize,
							  unsigned int stride,
							  unsigned int padding);

	void   SetInput( const std::vector<double> &input);
	void   GetOutput(std::vector<double> &output) const;

	double CalcSquareLoss(      const std::vector<double> &teacher);
	double CalcCrossEntropyLoss(const std::vector<double> &teacher);

	void   Forward( void);
	void   Backward(void);

	void   Learn(    double learnRatio = 0.001);
	void   LearnAdam(double alpha   = 0.001,
					 double beta1   = 0.9,
					 double beta2   = 0.999,
					 double epsilon = 1.0e-8);
	void   LearnAdamReset(void);
	void   DeltaNormalize(void);
	
	unsigned int GetInputNum(void) const
	{
		if (m_Layer.size() == 0)
			return (0);

		return (m_Layer[0]->GetInputNum());
	}
	
	unsigned int GetOutputNum(void) const
	{
		if (m_Layer.size() == 0)
			return (0);

		return (m_Layer[m_Layer.size()-1]->GetOutputNum());
	}
	
	void    Save(std::vector<char>       &data);
	void    Load(const std::vector<char> &data);
};

#endif /* NEURAL_NET_H_ */
