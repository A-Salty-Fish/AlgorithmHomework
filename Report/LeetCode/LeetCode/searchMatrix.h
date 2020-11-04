#pragma once
#include<vector>
using namespace std;
class Solution {
public:
    bool search_submatrix(vector<vector<int>>& matrix, int target,
        int left, int right, int up, int down)
    {
        if (left > right || up > down)
            return false;
        else if (target < matrix[up][left] || target > matrix[down][right])
            return false;
        else
        {
            int row = up;
            int mid = (left + right);// 2
            while (row <= down and target >= matrix[row][mid])
            {
                if (target == matrix[row][mid])
                    return true;
                row = row + 1;
            }
            return search_submatrix(matrix, target, left, mid - 1, row, down) || search_submatrix(matrix, target, mid + 1, right, up, row - 1);
        }
    }
    bool searchMatrix(vector<vector<int>>& matrix, int target)
    {
        if (matrix.size() == 0 || matrix[0].size() == 0)
            return false;
        int left = 0;
        int right = matrix[0].size() - 1;
        int up = 0;
        int down = matrix.size() - 1;
        return search_submatrix(matrix, target, left, right, up, down);
    }
};
//https://blog.csdn.net/qq_30841655/article/details/104512991