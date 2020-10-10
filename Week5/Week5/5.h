#pragma once
#include <iostream>
using namespace std;
const int n = 4;
const int m = 2;
int Array[n] = { 1,2,3,4 };
int Arrange[m] = { 1,2 };//记录排列

void GetArrange(int i)//获得m个元素的排列
{
	if (i==m)
	{
		for (int x : Arrange)
		{
			cout << x;
		}
		cout << endl;
	}
	else
	{
		for (int j = i; j < m; j++)
		{
			swap(Arrange[i], Arrange[j]);
			GetArrange(i + 1);
			swap(Arrange[i], Arrange[j]);
		}
	}
}
int num = 0;//当前子集元素个数
bool Choice[n] = { false };
void GetArray(int i)//获得含m个元素的子集
{
	if (num == m)//找到叶子节点
	{
		int j = 0;
		for (int i = 0; i < n; i++)//获得一个子集
			if (Choice[i])
				Arrange[j++] = Array[i];
		GetArrange(0);//打印子集排列
	}
	else
	{
		if (num < m)//左剪枝
		{
			Choice[i] = true;
			num++;
			GetArray(i + 1);
		}
		//回溯
		Choice[i] = false;
		num--;
		if (n - i  + num > m)//右剪枝
		{
			GetArray(i + 1);
		}
	}
}