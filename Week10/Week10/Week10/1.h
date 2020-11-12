#pragma once
#include<vector>
#include<algorithm>
using namespace std;
int MaxConferences(vector<int> people, vector<int> room)
{
	int count = 0;
	//��С��������
	sort(people.begin(), people.end());
	sort(room.begin(), room.end());
	auto peopleIndex = people.begin();
	auto roomIndex = room.begin();
	while (peopleIndex != people.end() && roomIndex != room.end())
	{
		//̰�� �����ܰ�С�����������ٵĻ���
		if (*peopleIndex <= *roomIndex)
		{
			count++;
			peopleIndex++;
			roomIndex++;
		}
		else
		{
			roomIndex++;
		}
	}
	return count;
}