#pragma once
#include <iostream>
using namespace std;
const int n = 4;
const int m = 2;
int Array[n] = { 1,2,3,4 };
int Arrange[m] = { 1,2 };//��¼����

void GetArrange(int i)//���m��Ԫ�ص�����
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
int num = 0;//��ǰ�Ӽ�Ԫ�ظ���
bool Choice[n] = { false };
void GetArray(int i)//��ú�m��Ԫ�ص��Ӽ�
{
	if (num == m)//�ҵ�Ҷ�ӽڵ�
	{
		int j = 0;
		for (int i = 0; i < n; i++)//���һ���Ӽ�
			if (Choice[i])
				Arrange[j++] = Array[i];
		GetArrange(0);//��ӡ�Ӽ�����
	}
	else
	{
		if (num < m)//���֦
		{
			Choice[i] = true;
			num++;
			GetArray(i + 1);
		}
		//����
		Choice[i] = false;
		num--;
		if (n - i  + num > m)//�Ҽ�֦
		{
			GetArray(i + 1);
		}
	}
}