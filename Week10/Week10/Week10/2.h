#pragma once
const int maxNum = 100;
static int num[maxNum][maxNum];
//动态规划求解 递推式：f(m,n)=f(m-1,n)+f(m,n-1)
int pathNum(int m, int n)
{
	if ((m == 0 && n == 0) || m < 0 || n < 0) return 0;
	if (m == 0 && n == 1) return 1;
	if (m == 1 && n == 0) return 1;
	else
	{
		if (num[m][n] != 0) return num[m][n];
		else
		{
			num[m][n] = pathNum(m - 1, n) + pathNum(m, n - 1);
			return num[m][n];
		}
	}
}