#pragma once
#include <iostream>
using namespace std;
const int N = 3;
int Array[N] = { 1,1,3};
//�жϵ�ǰ��i��Ԫ����û����k-i֮�����
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
			if (IsRepeat(n, i)) continue;//��������ظ���������ν���
			swap(Array[n], Array[i]);
			dfs(n + 1);
			swap(Array[n], Array[i]);
		}
	}
}
