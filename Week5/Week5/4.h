#pragma once
#include <iostream>
using namespace std;
const int N = 3;
int Array[N] = { 1,1,3};
//判断当前第i个元素有没有在k-i之间出现
bool IsRepeat(int k, int i)
{
	for (int j = k; j <= i - 1; j++)
	{
		if (Array[j] == Array[i])
		{
			return true;
		}
	}
	return false;
}
void dfs(int n)
{
	if (n == N)
	{
		for (int x : Array)
		{
			cout << x << "";
		}
		cout << endl;
	}
	else
	{
		for (int i=n;i<N;i++)
		{
			if (IsRepeat(n, i)) continue;//如果出现重复则跳过这次交换
			swap(Array[n], Array[i]);
			dfs(n + 1);
			swap(Array[n], Array[i]);
		}
	}
}
