#pragma once
#include<queue>
#include<vector>
#include<numeric>
#include<unordered_set>
using namespace std;
class Solution {
public:
    struct Node {
        int depth;//存放当前层数
        int remain;//存放剩余和
        Node(int d, int r) :depth(d), remain(r) {}
        bool operator<(const Node& a) const
        {
            return remain > a.remain; //小顶堆
        }
    };
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2 == 1)//数组和为奇数则无法二分
            return false;
        vector<unordered_set<int>> IsRepeat;//存放某层遍历的子树是否会重复
        IsRepeat.resize(nums.size());
        priority_queue<Node> p;//优先队列，小顶堆
        //将初始节点入队
        p.push(Node(0, sum / 2));
        p.push(Node(0, sum / 2 - nums[0]));
        while (!p.empty())
        {
            Node current = p.top(); p.pop();//出队头结点
            int remain = current.remain, depth = current.depth;
            //剪枝操作
            if (remain == 0) return true;
            if (remain < 0) continue;
            if (depth >= nums.size() - 1) continue;
            if (IsRepeat[depth].find(remain) !=
                IsRepeat[depth].end()) continue;
            //存放子树状态
            IsRepeat[depth].insert(remain);
            p.push(Node(depth + 1, remain));
            p.push(Node(depth + 1, remain - nums[depth + 1]));
        }
        return false;
    }
};