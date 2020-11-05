#pragma once
#include<queue>
#include<vector>
#include<numeric>
#include<unordered_set>
using namespace std;
class Solution {
public:
    struct Node {
        int depth;//��ŵ�ǰ����
        int remain;//���ʣ���
        Node(int d, int r) :depth(d), remain(r) {}
        bool operator<(const Node& a) const
        {
            return remain > a.remain; //С����
        }
    };
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2 == 1)//�����Ϊ�������޷�����
            return false;
        vector<unordered_set<int>> IsRepeat;//���ĳ������������Ƿ���ظ�
        IsRepeat.resize(nums.size());
        priority_queue<Node> p;//���ȶ��У�С����
        //����ʼ�ڵ����
        p.push(Node(0, sum / 2));
        p.push(Node(0, sum / 2 - nums[0]));
        while (!p.empty())
        {
            Node current = p.top(); p.pop();//����ͷ���
            int remain = current.remain, depth = current.depth;
            //��֦����
            if (remain == 0) return true;
            if (remain < 0) continue;
            if (depth >= nums.size() - 1) continue;
            if (IsRepeat[depth].find(remain) !=
                IsRepeat[depth].end()) continue;
            //�������״̬
            IsRepeat[depth].insert(remain);
            p.push(Node(depth + 1, remain));
            p.push(Node(depth + 1, remain - nums[depth + 1]));
        }
        return false;
    }
};