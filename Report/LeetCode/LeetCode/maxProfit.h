#pragma once
#include<vector>
using namespace std;
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int num = prices.size();
        if (num <= 1) return 0;
        int(*dp)[3] = new int[num][3];
        //dp[i][x]第i天进入(处于)x状态（0.不持股，1.持股，2.冷冻期）
        //不持股
        dp[0][0] = 0;
        //持股
        dp[0][1] = -prices[0];
        //冷冻期
        dp[0][2] = 0;
        for (int i = 1; i < num; i++) {
            //第i天不持股可以从两种状态转移而来，1.第i-1天不持股，今天仍不买股票，保持不持              股状态。2.冷冻期结束了，但是今天不买股票。
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2]);
            //第i天持股可从两种状态转移而来，1.第i-1天不持股(包含昨天是冷冻期，冷冻期结束后转为不持股状态和昨天本身就不持股这两种情况)，今天买股票。2.第i-1天持股，今天不卖出，保持持股状态。
            dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
            //只有第i-1天卖出了股票，第i天才处于冷冻期。
            dp[i][2] = dp[i - 1][1] + prices[i];
        }
        //只有最后一天不持股（不持股状态）或者前一天已经卖掉了（今天为冷冻期）这两种情况手里是拿着钱的，最大值在二者中产生。
        return max(dp[num - 1][0], dp[num - 1][2]);
    }
};