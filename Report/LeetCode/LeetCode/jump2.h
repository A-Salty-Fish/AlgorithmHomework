#pragma once
#include <vector>
using namespace std;
class Solution {
public:
    int jump(vector<int>& nums) {
        if (nums.size() == 1) return 0;
        int currentIndex = 0;//当前下标
        int count = 0;//跳跃次数
        while (true)
        {
            //下一跳就能到终点
            if (currentIndex + nums[currentIndex] >= nums.size() - 1)
                return count + 1;
            //贪心，寻找两跳能覆盖最远的中间点
            int max = currentIndex + nums[currentIndex];
            int nextStep = 1;
            for (int i = 1; i <= nums[currentIndex]; i++)
            {
                int currentLen = currentIndex + i + nums[currentIndex + i];
                if (currentLen >= max)
                {
                    max = currentLen;
                    nextStep = i;
                }
            }
            currentIndex += nextStep;
            count++;
        }
        return count;
    }
};