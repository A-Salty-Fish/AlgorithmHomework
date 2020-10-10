#include <iostream>
#include <vector>
using namespace std;
const int N = 6;
bool result[N];//保存最优解
bool IsFirstResult = true;//是否为第一次找到解
bool tmpResult[N];//保存临时解
int Nums(bool array[])//返回解需要的数目
{
	int sum = 0;
	for (int i = 0; i < N; i++) sum += array[i] ? 1 : 0;
	return sum;
}
//递归遍历input数组，depth为深度，CurrentSum为当前和，
//RestSum为剩下总和，target为目标和，nums为数组长度
void dfs(vector<int>& input, int depth, int CurrentSum,
	int RestSum, int target, int nums)
{
	if (depth == nums)//到叶子节点
		if (CurrentSum == target)//判断是否找到解
		{
			if (IsFirstResult)//如果是第一次找到解，则直接保存
			{
				int i = 0;
				for (int x : tmpResult)
				{
					result[i++] = x;
				}
				IsFirstResult = false;
			}
			else//不是第一次找到解，则比较数量是否最优
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
		if (CurrentSum + input[depth] <= target)//大于目标值则左剪值
		{
			tmpResult[depth] = true;//将该值存入临时解
			dfs(input, depth + 1, CurrentSum + input[depth],
				RestSum - input[depth], target, nums);
			tmpResult[depth] = false;//回溯
		}
		if (CurrentSum + RestSum - input[depth] >= target)//小于目标值则右剪值
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

