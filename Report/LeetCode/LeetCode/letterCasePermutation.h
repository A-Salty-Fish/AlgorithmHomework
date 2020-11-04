#pragma once
#include<string>
#include<vector>
using namespace std;
class Solution {
public:
    void dfs(string S, vector<string>& Result, int n)
    {
        if (n == S.length())
        {
            Result.push_back(S);
            return;
        }
        dfs(S, Result, n + 1);
        if ((S[n] >= 'a' && S[n] <= 'z'))
        {
            S[n] -= 32;
            dfs(S, Result, n + 1);
        }
        else if ((S[n] >= 'A' && S[n] <= 'Z'))
        {
            S[n] += 32;
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