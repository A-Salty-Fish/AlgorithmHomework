#pragma once
int FindSubscript(int* array, int length)
{
	if (array[0] >= length || array[length - 1] < 0) return -1;
	int mid = length / 2;
	int bias = 0;
	while (length != 0)
	{
		if (array[mid] == mid) return mid;
		else if (array[mid] > mid)
		{
			mid = (mid + bias) / 2;
		}
		else
		{
			int tmpBias = mid;
			mid = mid + std::max(1, (mid - bias) / 2);
			bias = tmpBias;
		}
		length /= 2;
	}
	return -1;
}