#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <vector>

class teacherData
{
private:
	typedef struct data_tag
	{
		std::vector<double>	input;
		std::vector<double>	teacher;
		std::vector<int>	count;
	} data;

	std::vector<data>	m_Data;
	unsigned int		m_Input;
	unsigned int		m_Output;

	void SetTeacher(data &setData) const
	{
		unsigned int max= 0;

		for (unsigned int i = 0; i < setData.count.size(); ++i)
		{
			if (setData.count[i] > setData.count[max])
				max	= i;
		}
		
		setData.teacher.resize(setData.count.size());

		for (unsigned int i = 0; i < setData.count.size(); ++i)
			setData.teacher[i] = (i == max) ? 1.0 : 0.0;
	}

public:
	teacherData(unsigned int input, unsigned int output) :
		m_Input(input),
		m_Output(output)
	{}

	void Load(const char *fileName)
	{
		FILE	*pFile;

		pFile = fopen(fileName, "rb");

		if (pFile == NULL)
			return;

		while (!feof(pFile))
		{
			std::vector<double>	input;

			for (unsigned int i = 0; i < m_Input; ++i)
			{
				double	temp;

				if (fread(&temp, 1, sizeof(temp), pFile) == 0)
				{
					fclose(pFile);
					return;
				}

				input.push_back(temp);
			}

			std::vector<int> count;

			for (unsigned int i = 0; i < m_Output; ++i)
			{
				int	temp;

				if (fread(&temp, 1, sizeof(temp), pFile) == 0)
				{
					fclose(pFile);
					return;
				}
				count.push_back(temp);
			}

			Add(input, count);
		}
		fclose(pFile);
	}

	void Save(const char *fileName) const
	{
		FILE	*pFile;

		pFile = fopen(fileName, "wb");

		for (unsigned int i = 0; i < m_Data.size(); ++i)
		{
			for (unsigned int j = 0; j < m_Input; ++j)
				fwrite(&m_Data[i].input[j], 1, sizeof(double), pFile);

			for (unsigned int j = 0; j < m_Output; ++j)
				fwrite(&m_Data[i].count[j], 1, sizeof(int), pFile);
		}

		fclose(pFile);
	}

	void Add(std::vector<double> &input, int id)
	{
		for (unsigned int i = 0; i < m_Data.size(); ++i)
		{
			if (std::equal(input.cbegin(), input.cend(), m_Data[i].input.cbegin()))
			{
				++m_Data[i].count[id];

				SetTeacher(m_Data[i]);
				return;
			}
		}

		std::vector<int>	count;
		count.resize(m_Output);

		for (unsigned int i = 0; i < m_Output; ++i)
		{
			if (id == i)
			{
				count[i] = 1;
			}
			else {
				count[i] = 0;
			}
		}

		Add(input, count);
	}

	void Add(std::vector<double> &input, std::vector<int> &count)
	{
		data	temp;
		temp.input = input;
		temp.count = count;

		SetTeacher(temp);

		m_Data.push_back(temp);
	}

	unsigned int	GetDataCount(void) const { return (m_Data.size()); }
	const std::vector<double> &GetInput(  int index)	const { return (m_Data[index].input); }
	const std::vector<double> &GetTeacher(int index)	const { return (m_Data[index].teacher); }
};