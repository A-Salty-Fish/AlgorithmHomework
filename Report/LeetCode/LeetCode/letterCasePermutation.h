#pragma once
#include<string>
#include<vector>
using namespace std;
class Solution {
public:
    //深度优先
    void dfs(string S, vector<string>& Result, int n)
    {
        {
            //递归出口
            if (n == S.length())
            {
                Result.push_back(S);
                return;
            }
            if ((S[n] >= 'a' && S[n] <= 'z'))
            {
                S[n] -= 32;
                dfs(S, Result, n + 1);
                //回溯
                S[n] += 32;
            }
            else if ((S[n] >= 'A' && S[n] <= 'Z'))
            {
                S[n] += 32;
                dfs(S, Result, n + 1);
                //回溯
                S[n] -= 32;
            }
            dfs(S, Result, n + 1);
        }
    }
    vector<string> letterCasePermutation(string S) {
        int length = S.length();
        vector<string> Result;
        if (length == 0)
            return Result;
        dfs(S, Result, 0);
        return Result;
    }
};