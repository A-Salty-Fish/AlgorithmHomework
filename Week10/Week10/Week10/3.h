#pragma once
#include<string>
#include<vector>
using namespace std;
// �ݹ��������������
string MaxSub(string fruit1, string fruit2)
{
	if (fruit1.size() == 0 || fruit2.size() == 0)
		return string("");
	string sub1 = fruit1.substr(1, fruit1.size() - 1);
	string sub2 = fruit2.substr(1, fruit2.size() - 1);
	if (fruit1[0] == fruit2[0])
		return fruit1[0] + MaxSub(sub1, sub2);
	else
	{
		string subRtn1 = MaxSub(fruit1, sub2);
		string subRtn2 = MaxSub(sub1, fruit2);
		return subRtn1.size() > subRtn2.size() ? subRtn1 : subRtn2;
	}
}
string NewFruit(string fruit1, string fruit2)
{
	string sub = MaxSub(fruit1, fruit2);
	string result;
	int fruit1Index = 0, fruit2Index = 0, subIndex = 0;
	while (fruit1Index != fruit1.size() && fruit2Index != fruit2.size())
	{
		// �������ڹ����������е��ַ�������
		if (fruit1[fruit1Index] != sub[subIndex])
		{
			result.push_back(fruit1[fruit1Index]);
			fruit1Index++;
			continue;
		}
		if (fruit2[fruit2Index] != sub[subIndex])
		{
			result.push_back(fruit2[fruit2Index]);
			fruit2Index++;
			continue;
		}
		// �����ڹ����������е��ַ�������
		result.push_back(sub[subIndex]);
		subIndex++;
		fruit1Index++;
		fruit2Index++;
	}
	// ��ʣ�µ��ַ�������
	while (fruit1Index != fruit1.size())
		result.push_back(fruit1[fruit1Index++]);
	while (fruit2Index != fruit2.size())
		result.push_back(fruit2[fruit2Index++]);
	return result;
}