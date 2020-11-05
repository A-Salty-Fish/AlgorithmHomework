#pragma once
#include <vector>
using namespace std;
class Solution {
public:
    int jump(vector<int>& nums) {
        if (nums.size() == 1) return 0;
        int currentIndex = 0;//��ǰ�±�
        int count = 0;//��Ծ����
        while (true)
        {
            //��һ�����ܵ��յ�
            if (currentIndex + nums[currentIndex] >= nums.size() - 1)
                return count + 1;
            //̰�ģ�Ѱ�������ܸ�����Զ���м��
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