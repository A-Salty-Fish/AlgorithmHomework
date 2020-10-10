#include <iostream>
#include <vector>
using namespace std;
const int N = 6;
bool result[N];//�������Ž�
bool IsFirstResult = true;//�Ƿ�Ϊ��һ���ҵ���
bool tmpResult[N];//������ʱ��
int Nums(bool array[])//���ؽ���Ҫ����Ŀ
{
	int sum = 0;
	for (int i = 0; i < N; i++) sum += array[i] ? 1 : 0;
	return sum;
}
//�ݹ����input���飬depthΪ��ȣ�CurrentSumΪ��ǰ�ͣ�
//RestSumΪʣ���ܺͣ�targetΪĿ��ͣ�numsΪ���鳤��
void dfs(vector<int>& input, int depth, int CurrentSum,
	int RestSum, int target, int nums)
{
	if (depth == nums)//��Ҷ�ӽڵ�
		if (CurrentSum == target)//�ж��Ƿ��ҵ���
		{
			if (IsFirstResult)//����ǵ�һ���ҵ��⣬��ֱ�ӱ���
			{
				int i = 0;
				for (int x : tmpResult)
				{
					result[i++] = x;
				}
				IsFirstResult = false;
			}
			else//���ǵ�һ���ҵ��⣬��Ƚ������Ƿ�����
			{
				if (Nums(result) > Nums(tmpResult))
				{
					int i = 0;
					for (int x : tmpResult)
					{
						result[i++] = x;
					}
				}
			}
		}
		else return;
	else
	{
		if (CurrentSum + input[depth] <= target)//����Ŀ��ֵ�����ֵ
		{
			tmpResult[depth] = true;//����ֵ������ʱ��
			dfs(input, depth + 1, CurrentSum + input[depth],
				RestSum - input[depth], target, nums);
			tmpResult[depth] = false;//����
		}
		if (CurrentSum + RestSum - input[depth] >= target)//С��Ŀ��ֵ���Ҽ�ֵ
		{
			dfs(input, depth + 1, CurrentSum,
				RestSum - input[depth], target, nums);
		}
	}
}
void GetSolution(vector<int>& input, int target)
{
	int RestSum = 0;
	for (int x : input)
		RestSum += x;
	dfs(input, 0, 0, RestSum, target, input.size());
}

