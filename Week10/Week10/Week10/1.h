#pragma once
#include<vector>
#include<algorithm>
using namespace std;
int MaxConferences(vector<int> people, vector<int> room)
{
	int count = 0;
	//从小到大排序
	sort(people.begin(), people.end());
	sort(room.begin(), room.end());
	auto peopleIndex = people.begin();
	auto roomIndex = room.begin();
	while (peopleIndex != people.end() && roomIndex != room.end())
	{
		//贪心 尽可能把小房间分配给人少的会议
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