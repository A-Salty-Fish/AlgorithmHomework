#pragma once
#include <math.h>
int SepaExpNum(int num)
{
	if (num <= 2) return 1;
	else
	{
		int sum = 0;
		for (int i = 2;i <=num ;i++)
		{
			if (num % i == 0)
			{
				sum += SepaExpNum(num / i);
			}
		}
		return sum;
	}
}