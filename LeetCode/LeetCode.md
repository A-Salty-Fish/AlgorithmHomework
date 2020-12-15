[toc]
# 分治法
## 240.搜索二维矩阵II
```
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
```
```
思路：
利用每一行有序的信息，我们可以关于将二维矩阵分成四个部分，其中target只可能落在两个小矩阵里面，然后再递归对这两个小矩阵查找。
步骤1：当前矩阵下，检验矩阵是否为空，以及target是否大于最大值或者小于最小值，如果任何一个满足，则返回false；如都不满足，则进行步骤2
步骤2：mid=中间列的索引，计算row使得matrix[row-1][mid]<=target<=matrix[row][mid]，如果target=matrix[row-1][mid]或者target=matrix[row][mid]，则返回true；否则，进行步骤3
步骤3：通过步骤2可知，matrix[row-1][mid]<target<matrix[row][mid]严格成立，我们据此将矩阵划分成四个小矩阵。左上角和右下角矩阵是不可能的，target只可能落在左下角和右上角矩阵内。
步骤4：对左下角和右上角矩阵做递归
```
```cpp
class Solution {
public:
    bool search_submatrix(vector<vector<int>>& matrix, int target,
    int left,int right,int up,int down)
    {
        if (left > right || up > down)
            return false;
        else if (target < matrix[up][left] || target > matrix[down][right])
            return false;
        else
        {
            int row = up;
            int mid = (left + right) / 2;
            while (row <= down and target >= matrix[row][mid])
            {
                if (target == matrix[row][mid])
                    return true;
                row = row + 1;
            }
            return search_submatrix(matrix, target, left, mid-1, row, down) ||                          search_submatrix(matrix, target, mid+1, right, up, row-1);
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
```
## 315.计算右侧小于当前元素的个数
```
给定一个整数数组 nums，按要求返回一个新数组 counts。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。
```
```
思路：归并排序逆序数，使用索引数组。
```
```cpp
class Solution {
public:
    vector<int> counter;
    void merge(vector<int>& nums,vector<int>& indexes,int start,int mid, int end) {
        int left = start;
        int right = mid+1;
        vector<int> tmpMerge(end-start+1);
        vector<int> tmpIndex(end-start+1);
        int p=0;
        for (;left<=mid&&right<=end;){
            if (nums[right]>=nums[left]){
                tmpMerge[p]=nums[left];
                tmpIndex[p++]=indexes[left];
                counter[indexes[left++]]+=(right-mid-1);
            } else{//nums[lef]>nums[right]
                tmpMerge[p]=nums[right];
                tmpIndex[p++]=indexes[right++];
            }
        }
        while (left<=mid) {
            tmpMerge[p]=nums[left];
            tmpIndex[p++]=indexes[left];
            counter[indexes[left++]]+=(right-mid-1);
        }
        while (right<=end) {
            tmpMerge[p]=nums[right];
            tmpIndex[p++]=indexes[right++];
        }
        //复制回原数组
        p=0;
        for (int i=start;i<=end;++i) {
            nums[i]=tmpMerge[p];
            indexes[i]=tmpIndex[p++];
        }
    }
    void helper(vector<int>& nums,vector<int>& indexes,int start,int end) {
        if (start>=end) return;
        int mid=start+(end-start)/2;
        helper(nums,indexes,start,mid);
        helper(nums,indexes,mid+1,end);
        if (nums[mid]>nums[mid+1])
            merge(nums,indexes,start,mid,end);
    }
    vector<int> countSmaller(vector<int>& nums) {
        vector<int> res;
        if (nums.size()<1) return res;
        // 初始化索引数组
        vector<int> indexes(nums.size());
        counter.resize(nums.size());
        for (int i=0;i<nums.size();i++)
            indexes[i]=i;
        helper(nums,indexes,0,nums.size()-1);
        for (int item:counter)
            res.push_back(item);
        return res;
    }
};
```
## 327.区间和计数
```
给定一个整数数组 nums，返回区间和在 [lower, upper] 之间的个数，包含 lower 和 upper。
区间和 S(i, j) 表示在 nums 中，位置从 i 到 j 的元素之和，包含 i 和 j (i ≤ j)。
```
```cpp
class Solution {
public:
    int countRangeSum(vector<int>& nums, int lower, int upper) {
        int size = nums.size();
        if( size == 0 ) return 0;
        vector<long long > presum;//前缀和数组， [0,i] 区间和
        long long pre = 0;
        for( auto & num : nums)
        {
            pre += num;
            presum.emplace_back(pre);
        }
        int result = 0;
        vector<long long > temp(size);//用于合并两个有序数组的临时数组
        mergesort(presum,lower,upper,temp,0,size-1,result);
        return result;
    }
    void mergesort(vector<long long>& presum, int lower, int upper,vector<long long >& temp,int left,int right,int &result)
    {
        if( left == right)//分到只剩一个元素
        {
            if( presum[left] >= lower && presum[left] <= upper)
            {
                result++;
            }
            return;
        }
        int mid = left+(right-left)/2;
        mergesort(presum,lower,upper,temp,left,mid,result);//使 [left,mid] 有序
        mergesort(presum,lower,upper,temp,mid+1,right,result);//使 [mid+1,right] 有序
        //合并之前先统计
        int i = left ;// i 指向左区间
        int j_left = mid+1;
        int j_right = mid+1;// j_left、j_right 指向右区间，i < j，相减得到区间和
        while( i < mid+1 )// i 固定时，j 越大差越大；j  固定时，i 越大差越小
        {
            while(j_left <= right && presum[j_left] - presum[i] < lower )//找到下限位置
            {
                j_left++;
            }
            j_right = j_left;
            while( j_right <= right && presum[j_right] - presum[i] <= upper) //找到上限位置
            {
                j_right++;
                result++;//找到一对
            }
            i++;   
        }
        //合并
        i = left;
        int j = mid+1;
        int t = 0;
        while( i <= mid && j <= right)
        {
            if( presum[i] <= presum[j])
            {
                temp[t++] = presum[i++];
            }
            else
            {
                temp[t++] = presum[j++];
            }
        }
        while( i <= mid )
        {
            temp[t++] = presum[i++];
        }
        while( j <= right )
        {
            temp[t++] = presum[j++];
        }
        t = 0;
        i = left;
        while( i <= right)
        {
            presum[i++] = temp[t++];
        }
    }
};
```
## 658. 找到 K 个最接近的元素(二分查找)
```
给定一个排序好的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）的 k 个数。返回的结果必须要是按升序排好的。
整数 a 比整数 b 更接近 x 需要满足：
|a - x| < |b - x| 或者
|a - x| == |b - x| 且 a < b
```
```cpp
class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x) {
    int lower = 0, high = arr.size() - k;
    while (lower < high) {
        int mid = lower + ((high - lower) / 2);
        if (x - arr[mid] > arr[mid + k] - x ) {
            lower = mid + 1;
        } else {
            high = mid;
        }
    }
    return vector<int>(arr.begin() + lower, arr.begin() + lower + k);
}
};
```
## 95. 不同的二叉搜索树 II
```
给定一个整数 n，生成所有由 1 ... n 为节点所组成的 二叉搜索树 。
```
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> generateTrees(int start, int end) {
        if (start > end) {
            return { nullptr };
        }
        vector<TreeNode*> allTrees;
        // 枚举可行根节点
        for (int i = start; i <= end; i++) {
            // 获得所有可行的左子树集合
            vector<TreeNode*> leftTrees = generateTrees(start, i - 1);
            // 获得所有可行的右子树集合
            vector<TreeNode*> rightTrees = generateTrees(i + 1, end);
            // 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
            for (auto& left : leftTrees) {
                for (auto& right : rightTrees) {
                    TreeNode* currTree = new TreeNode(i);
                    currTree->left = left;
                    currTree->right = right;
                    allTrees.emplace_back(currTree);
                }
            }
        }
        return allTrees;
    }

    vector<TreeNode*> generateTrees(int n) {
        if (!n) {
            return {};
        }
        return generateTrees(1, n);
    }
};
```
# 回溯法
## 131. 分割回文串
```
给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
返回 s 所有可能的分割方案。
示例:
输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
```
```cpp
class Solution {
private:
    vector<vector<string>> result;
    vector<string> path; // 放已经回文的子串
    void backtracking (const string& s, int startIndex) {
        // 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
        if (startIndex >= s.size()) {
            result.push_back(path);
            return;
        }
        // 寻找下一个截取位置
        for (int i = startIndex; i < s.size(); i++) {
            if (isPalindrome(s, startIndex, i)) {   // 是回文子串
                // 获取[startIndex,i]在s中的子串
                string str = s.substr(startIndex, i - startIndex + 1);
                path.push_back(str);
            } else {                                // 不是回文，跳过
                continue;
            }
            backtracking(s, i + 1); // 寻找i+1为起始位置的子串
            path.pop_back(); // 回溯过程，弹出本次已经填在的子串
        }
    }
    // 判断是否是回文串
    bool isPalindrome(const string& s, int start, int end) {
        for (int i = start, j = end; i < j; i++, j--) {
            if (s[i] != s[j]) {
                return false;
            }
        }
        return true;
    }
public:
    vector<vector<string>> partition(string s) {
        result.clear();
        path.clear();
        backtracking(s, 0);
        return result;
    }
};
```
## 78. 子集
```
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。
```
```cpp
class Solution {
public:
    vector<vector<int>> result;// 保存子集
    vector<int> currentSubset;// 存放临时解
    vector<vector<int>> subsets(vector<int>& nums) {
        subRecursive(nums, 0);
        return result;
    }
    void subRecursive(vector<int>& nums,int i) {
        // 递归出口
        if (i == nums.size())
        {
            result.push_back(currentSubset);
            return;
        }
        // 将当前字符加入子集
        currentSubset.push_back(nums[i]);
        subRecursive(nums, i+1);
        // 弹出当前字符 回溯
        currentSubset.pop_back();
        subRecursive(nums, i+1);
    }
};
```
## 90. 子集 II
```
给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。
示例:
输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```
```cpp
class Solution {
public:
    vector<vector<int>> result;
    vector<int> currentSubset;
    void dfs(vector<int> nums, int index) {
        result.push_back(currentSubset);
        if(index == nums.size()) {
            return;
        }
        for(int i=index; i<nums.size(); ++i) {
            // 去除重复子树
            if(i > index && nums[i] == nums[i-1]) 
                continue;
            currentSubset.push_back(nums[i]);
            dfs(nums,i+1);
            // 回溯
            currentSubset.pop_back();
        }
    }
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        dfs(nums,0);
        return result;
    }
};
```
## 46. 全排列
```
给定一个 没有重复 数字的序列，返回其所有可能的全排列。
```
```cpp
class Solution {
public:
    void backtrack(vector<vector<int>>& result, vector<int>& output, int current, int len){
        // 递归出口
        if (current == len) {
            result.push_back(output);
            return;
        }
        for (int i = current; i < len; ++i) {
            // 交换
            swap(output[i], output[current]);
            // 继续递归填下一个数
            backtrack(result, output, current + 1, len);
            // 回溯
            swap(output[i], output[current]);
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> result;
        backtrack(result, nums, 0, (int)nums.size());
        return result;
    }
};
```

## 47. 全排列 II
```
给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
```
```cpp
class Solution {
private:
    vector<vector<int>> result;
    vector<int> path;
    void backtracking (vector<int>& nums, vector<bool>& used) {
        // 此时说明找到了一组
        if (path.size() == nums.size()) {
            result.push_back(path);
            return;
        }
        for (int i = 0; i < nums.size(); i++) {
            // used[i - 1] == true，说明同一树支nums[i - 1]使用过
            // used[i - 1] == false，说明同一树层nums[i - 1]使用过
            // 如果同一树层nums[i - 1]使用过则直接跳过
            if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) {
                continue;
            }
            if (used[i] == false) {
                used[i] = true;
                path.push_back(nums[i]);
                backtracking(nums, used);
                path.pop_back();
                used[i] = false;
            }
        }
    }
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end()); // 排序
        vector<bool> used(nums.size(), false);
        backtracking(nums, used);
        return result;
    }
};
```
## 980. 不同路径 III
```
在二维网格 grid 上，有 4 种类型的方格：
1 表示起始方格。且只有一个起始方格。
2 表示结束方格，且只有一个结束方格。
0 表示我们可以走过的空方格。
-1 表示我们无法跨越的障碍。
返回在四个方向（上、下、左、右）上行走时，从起始方格到结束方格的不同路径的数目。
每一个无障碍方格都要通过一次，但是一条路径中不能重复通过同一个方格。
```
```cpp
class Solution {
public:
    int zeroNum=0;// 可走的格子数
    int pathNum = 0;// 路径数
    int uniquePathsIII(vector<vector<int>>& grid) {
        int startX=0;
        int startY=0;
        Init(grid, startX, startY);
        dfs(grid, 0, startX, startY);
        return pathNum;
    }
    // 初始化可走的格子数和起点坐标
    void Init(vector<vector<int>>& grid, int & startX, int &startY) {
        for(int i=0; i<grid.size(); i++)
            for(int j=0; j<grid[i].size(); j++){
                if(grid[i][j]==1){
                    startX=i;
                    startY=j;
                }
                else if(grid[i][j]==0)
                    zeroNum++;
            }
    }
    // 深度优先遍历
    void dfs(vector<vector<int>>& grid, int tmpZeroNum, int currentX, int currentY){
        // 越界
        if(currentX<0||currentX>=grid.size()||currentY<0||currentY>=grid[0].size())
            return;
        // 障碍物
        if (grid[currentX][currentY]==-1)
            return;
        // 走到终点
        if(grid[currentX][currentY]==2){
            if(tmpZeroNum>zeroNum)// 走过所有可走格
                pathNum++;
            return;
        }
        grid[currentX][currentY]=-1;// 标记当前格
        // 在选择列表中进行递归循环
        dfs(grid, tmpZeroNum+1, currentX-1, currentY); // 左
        dfs(grid, tmpZeroNum+1, currentX, currentY+1); // 下
        dfs(grid, tmpZeroNum+1, currentX+1, currentY); // 右
        dfs(grid, tmpZeroNum+1, currentX, currentY-1); // 上
        // 回溯
        grid[currentX][currentY]=0;
        
    }
};
```
## 784. 字母大小写全排列
```
给定一个字符串S，通过将字符串S中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。
```
```cpp
class Solution {
public:
    void dfs(string S, vector<string>& Result, int n)
    {
        if (n == S.length()) // 到边界则存入一个解
        {
            Result.push_back(S);
            return;
        }
        if ((S[n] >= 'a' && S[n] <= 'z')) // 如果是小写字母
        {
            S[n] -= 32;// 变成大写字母
            dfs(S,Result,n+1);
            S[n] += 32;// 回溯
        }
        else if ((S[n] >= 'A' && S[n] <= 'Z'))// 如果是大写字母
        {
            S[n] += 32;// 变成小写字母
            dfs(S,Result,n+1);
            S[n] -= 32;// 回溯
        }
        dfs(S,Result,n+1);
    }
    vector<string> letterCasePermutation(string S) {
        int length = S.length();
        vector<string> Result;
        if (length == 0) 
            return Result;
        dfs(S,Result,0);
        return Result;
    }
};
```
## 77. 组合
```
给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
```
```cpp
class Solution {
public:
    vector<vector<int>> results;// 所有解
    vector<int> result;// 一个解
    vector<vector<int>> combine(int n, int k) {
        dfs(n, 1, k);
        return results;
    }
    void dfs(int n, int current, int remain) {
        if (current > n + 1) return;
        if (remain == 0) { // 得到一个解
            results.push_back(result);
            return;
        }
        result.push_back(current);
        dfs(n, current+1, remain-1);
        // 回溯
        result.pop_back();
        dfs(n, current+1, remain);
    }
};
```
## 112. 路径总和
```
给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
```
```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if (root==NULL) return false;
        return dfs(root,sum);
    }
    bool dfs(TreeNode* currentNode, int remain)
    {
        if (currentNode == NULL)
            return false;
        // 到叶节点
        if (currentNode->left==NULL&&currentNode->right==NULL)
            return currentNode->val == remain;
        remain -= currentNode->val;
        bool left = dfs(currentNode->left,remain);
        bool right = dfs(currentNode->right,remain);
        return left || right;
    }
};
```
## 1293. 网格中的最短路径
PS：回溯+剪枝只能跑通一半的答案，后面就超时了。
```
给你一个m * n 的网格，其中每个单元格不是 0（空）就是 1（障碍物）。每一步，您都可以在空白单元格中上、下、左、右移动。
如果您 最多 可以消除 k 个障碍物，请找出从左上角 (0, 0) 到右下角 (m-1, n-1) 的最短路径，并返回通过该路径所需的步数。如果找不到这样的路径，则返回 -1。
```
```cpp
class Solution {
public:
    int minStep = INT32_MAX;
    int shortestPath(vector<vector<int>>& grid, int k) {
        dfs(grid,0,0,grid.size()-1,grid[0].size()-1,k,0);
        return minStep == INT32_MAX ? -1 : minStep;
    }
    // 剪枝函数 返回步数下界
    int lowBound(int currentX, int currentY,int targetX, int targetY) {
        return (targetY - currentY) + (targetX - currentX);
    }
    void dfs(vector<vector<int>>& grid,int currentX, int currentY,
     int targetX, int targetY, int k, int step) {
         // 到达终点
        if (currentX==targetX&&currentY==targetY)
            if (step < minStep)
                {
                    minStep = step;
                    return;
                }
        // 越界
        if (currentX<0||currentX>targetX||currentY<0||currentY>targetY)
            return;
        int tmp = grid[currentX][currentY]; // 记录该格状态
        if (tmp == -1) // 该格走过
            return;
        // 下界函数剪枝
        if (lowBound(currentX, currentY, targetX, targetY) + step > minStep)
            return;
        // 走到障碍物
        if (tmp==1)
        {
            // 有消除障碍物的机会
            if (k>0)
            {
                grid[currentX][currentY] = 0;// 将其标记为可走
                dfs(grid,currentX,currentY,targetX,targetY,k-1,step);
                grid[currentX][currentY] = tmp;// 回溯
            }
            return;
        }
        // 当前格可走 拓展四个方向 优先拓展右下
        if (tmp == 0)
        {
            grid[currentX][currentY] = -1;// 标记为走过
            dfs(grid,currentX+1,currentY,targetX,targetY,k,step+1);
            dfs(grid,currentX,currentY+1,targetX,targetY,k,step+1);
            dfs(grid,currentX-1,currentY,targetX,targetY,k,step+1);
            dfs(grid,currentX,currentY-1,targetX,targetY,k,step+1);
            grid[currentX][currentY] = tmp;// 回溯
        }
    }
};
```
# 分支限界法
## 698. 划分为k个相等的子集
```
给定一个整数数组  nums 和一个正整数 k，找出是否有可能把这个数组分成 k 个非空子集，其总和都相等。
示例 1：
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。
```
```cpp
class Solution {
public:
    int n, len;
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        n=nums.size();
        vector<bool> vis(n, false);
        sort(nums.begin(), nums.end());
        for(int i=0; i<n; i++) len += nums[i];
        if(len%k!=0) return false;
        len=len/k;
        return dfs(n-1, len, n, nums, vis);
    }
    
    //当前从id位置开始枚举,还有sz个子集未合法是否能够使所有子集合法
    bool dfs(int id, int cur_len, int sz, vector<int>& arr, vector<bool>& vis) {
        if(sz == 0) return true; //n个数全部装下
        bool isok;
        for(int i=id; i>=0; i--) {
            if(!vis[i] && cur_len>=arr[i]) { //当前的数能够装入当前子集中
                vis[i]=true;
                isok = false;
                if(cur_len == arr[i]) isok = dfs(n-1, len, sz-1, arr, vis); //当前子集已满则要重新找一个len子集
                else isok = dfs(i-1, cur_len-arr[i], sz-1, arr, vis); //当前子集将目前最大的数取了,使得后面的子集更容易凑
                if(isok) return isok; //找到一组合法的拼凑序列
                if(cur_len==0) return isok; //最大的数不能装到任何子集使其形成合法序列
                //否则该数属于其他子集,当前子集不应该取它,进行判重剪枝
                vis[i] = false; //回溯
                while(i>0 && arr[i-1]==arr[i]) i--;
            }
        }
        return false;
    }
};
```

## 416.相同子集和分割
```
给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
```
```cpp
class Solution {
public:
    struct Node{
        int depth;
        int remain;
        Node(int d,int r):depth(d),remain(r){}
        bool operator<(const Node& a) const
        {
        return remain > a.remain; //小顶堆
        }
    };
    bool canPartition(vector<int>& nums) {
        int sum = accumulate(nums.begin(),nums.end(),0);
        if (sum % 2 == 1)
            return false;
        vector<unordered_set<int>> IsRepeat;
        IsRepeat.resize(nums.size());
        priority_queue<Node> p;// 广度优先搜索
        p.push(Node(0,sum/2));
        p.push(Node(0,sum/2-nums[0]));
        while (!p.empty())
        {
            Node current = p.top();p.pop();
            int remain = current.remain, depth = current.depth;
            if (remain == 0) return true;
            // 剪枝
            if (remain<0) continue;
            if (depth >= nums.size() - 1) continue;
            // 去除重复子树
            if (IsRepeat[depth].find(remain) !=
             IsRepeat[depth].end()) continue;
            IsRepeat[depth].insert(remain);
            p.push(Node(depth+1,remain));
            p.push(Node(depth+1,remain-nums[depth+1]));
        }
        return false;
    }
};
```
## 1210. 穿过迷宫的最少移动次数
```
我们在一个 n*n 的网格上构建了新的迷宫地图，蛇的长度为 2，也就是说它会占去两个单元格。蛇会从左上角（(0, 0) 和 (0, 1)）开始移动。我们用 0 表示空单元格，用 1 表示障碍物。蛇需要移动到迷宫的右下角（(n-1, n-2) 和 (n-1, n-1)）。
每次移动，蛇可以这样走：
如果没有障碍，则向右移动一个单元格。并仍然保持身体的水平／竖直状态。
如果没有障碍，则向下移动一个单元格。并仍然保持身体的水平／竖直状态。
如果它处于水平状态并且其下面的两个单元都是空的，就顺时针旋转 90 度。蛇从（(r, c)、(r, c+1)）移动到 （(r, c)、(r+1, c)）。
如果它处于竖直状态并且其右面的两个单元都是空的，就逆时针旋转 90 度。蛇从（(r, c)、(r+1, c)）移动到（(r, c)、(r, c+1)）。
返回蛇抵达目的地所需的最少移动次数。
如果无法到达目的地，请返回 -1。
```
```cpp
class Solution {
public:
    int minimumMoves(vector<vector<int>>& grid) {
        int n = grid.size();
        queue<pair<pair<int,int>,bool>> que;//蛇头的位置和是否水平
        int retCnt = 0;
		int size = 0;
        //防止重复加入数据。相当于剪枝
        vector<vector<int>> hCnt;
        vector<vector<int>> vCnt;
        for (int i = 0; i < n; i++) {
             vector<int> tmp(n, -1);
             hCnt.push_back(tmp);
             vCnt.push_back(tmp);
        }
        que.push(make_pair(make_pair(0,1), true));
        while (!que.empty()) {            
            size = que.size();
			int r;
			int c;
			bool isPing;
			while (size > 0) {
			    r = que.front().first.first;
				c = que.front().first.second;
				isPing = que.front().second;
				que.pop();
                size--;

                if ((isPing && hCnt[r][c] >= 0 && hCnt[r][c] <= retCnt) || ((!isPing) && vCnt[r][c] >= 0 && vCnt[r][c] <= retCnt)) {
                    continue;
                }
                //存储数据，剪枝用。空间换时间。不然会超时
                if (isPing)     hCnt[r][c] = retCnt;
                if (!isPing)    vCnt[r][c] = retCnt;

                if ((r == n-1) && (c == n-1) && isPing) {   return retCnt;  }
				//水平向右
				if ((isPing) && (c+1<n) && (grid[r][c+1] == 0))  {que.push(make_pair(make_pair(r, c+1), true));}
				//水平向下
				if ((isPing) && (r+1<n) && (grid[r+1][c] == 0) && (grid[r+1][c-1] == 0))  {que.push(make_pair(make_pair(r+1, c), true));}
				//竖直向下
				if ((!isPing) && (r+1<n) && (grid[r+1][c] == 0))  {que.push(make_pair(make_pair(r+1, c), false));}
				//竖直向右
				if ((!isPing) && (c+1<n) && (grid[r][c+1] == 0) && (grid[r-1][c+1] == 0))  {que.push(make_pair(make_pair(r, c+1), false));}
				//水平旋转
				if ((isPing) && (r+1<n) && (grid[r+1][c] == 0) && (grid[r+1][c-1] == 0))	{que.push(make_pair(make_pair(r+1, c-1), false));}        
				//竖直旋转
				if ((!isPing) && (c+1<n) && (grid[r][c+1] == 0) && (grid[r-1][c+1] == 0)) 	{que.push(make_pair(make_pair(r-1, c+1), true));}
			}
            retCnt++;
        }
        return -1;
    }
};
class Solution2 {
public:
    int minimumMoves(vector<vector<int>>& grid) {
        int n = grid.size() ;
        for (int i = 0; i < n; i++) {
             vector<int> tmp(n, -1);
             hCnt.push_back(tmp);
             vCnt.push_back(tmp);
        }
        dfs(grid, hCnt, vCnt, 0, 1, 1, 0);
        return hCnt[n-1][n-1];
        //if (retCnt == INT_MAX)  return -1;
        //return retCnt;
    }
    void dfs(vector<vector<int>>& grid, vector<vector<int>>& hCnt, vector<vector<int>>& vCnt, int r, int c, bool isPing, int step) {
        int n = grid.size();
        //剪枝是dfs的性能关键
        if ((isPing && hCnt[r][c] >= 0 && hCnt[r][c] <= step) || ((!isPing) && vCnt[r][c] >= 0 && vCnt[r][c] <= step)) {
            return;
        }
        //存储数据，剪枝用。空间换时间。不然会栈溢出
        if (isPing)     hCnt[r][c] = step;
        if (!isPing)    vCnt[r][c] = step;

        if ((r >= n-1) && (c >= n-1)) {
            if (isPing)   retCnt = min(retCnt, step);
            return;
        } 
        //水平向右
        if ((isPing) && (c+1<n) && (grid[r][c+1] == 0))  {dfs(grid, hCnt, vCnt, r, c+1, 1, step+1);}
        //水平向下
        if ((isPing) && (r+1<n) && (grid[r+1][c] == 0) && (grid[r+1][c-1] == 0))  {dfs(grid,  hCnt, vCnt, r+1, c, 1, step+1);}
        //竖直向下
        if ((!isPing) && (r+1<n) && (grid[r+1][c] == 0))  {dfs(grid,  hCnt, vCnt, r+1, c, 0, step+1);}
        //竖直向右
        if ((!isPing) && (c+1<n) && (grid[r][c+1] == 0) && (grid[r-1][c+1] == 0))  {dfs(grid,  hCnt, vCnt, r, c+1, 0, step+1);}
        //水平旋转        
        if ((isPing) && (r+1<n) && (grid[r+1][c] == 0) && (grid[r+1][c-1] == 0))   { dfs(grid,  hCnt, vCnt, r+1, c-1, 0, step+1);}        
        //竖直旋转
        if ((!isPing) && (c+1<n) && (grid[r][c+1] == 0) && (grid[r-1][c+1] == 0))  {dfs(grid,  hCnt, vCnt, r-1, c+1, 1, step+1);}
    }
private:
    int retCnt = INT_MAX;
    vector<vector<int>> hCnt;
    vector<vector<int>> vCnt;
};
```
# 贪心法
## 1497. 检查数组对是否可以被 k 整除
```
给你一个整数数组 arr 和一个整数 k ，其中数组长度是偶数，值为 n 。
现在需要把数组恰好分成 n / 2 对，以使每对数字的和都能够被 k 整除。
如果存在这样的分法，请返回 True ；否则，返回 False 。
```
```cpp
class Solution {
public:
    bool canArrange(vector<int>& arr, int k) {
        // 存放余数的数组
        vector<int> mod(k);
        // 根据余数来存到对应位置
        for (int num: arr) mod[(num%k+k)%k]++;
        // 贪心，匹配互补的余数
        for (int i = 1; i <= k/2; ++i)
            if (mod[i]!= mod[k-i])
                return false;
        return mod[0] % 2 == 0;
    }
};
```
## 134. 加油站
```
在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。。
```
```cpp
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int curSum = 0;
        int totalSum = 0;
        int start = 0;
        for (int i = 0; i < gas.size(); i++) {
            // 尝试找一个局部最优解
            curSum += gas[i] - cost[i];
            totalSum += gas[i] - cost[i];
            if (curSum < 0) {   // 当前累加rest[i]和 curSum一旦小于0
                start = i + 1;  // 起始位置更新为i+1
                curSum = 0;     // curSum从0开始
            }
        }
        if (totalSum < 0) return -1; // 说明怎么走都不可能跑一圈了
        return start;
    }
};
```
## 45. 跳跃游戏 II
```
给定一个非负整数数组，你最初位于数组的第一个位置。
数组中的每个元素代表你在该位置可以跳跃的最大长度。
你的目标是使用最少的跳跃次数到达数组的最后一个位置。
```
```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        if (nums.size() == 1) return 0;
        int currentIndex = 0;
        int count = 0;
        while (true)
        {
            // 下一跳到达终点
            if (currentIndex + nums[currentIndex] >= nums.size() - 1) 
                return count + 1;
            int max = currentIndex + nums[currentIndex];
            int nextStep = 1;
            for (int i = 1;i<=nums[currentIndex];i++)
            {
                int currentLen = currentIndex + i + nums[currentIndex+i];
                // 选取可跳范围最大的点
                if (currentLen>=max)
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
```
## 1288. 删除被覆盖区间
```
给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。
在完成所有删除操作后，请你返回列表中剩余区间的数目。
```
```cpp
class Solution {
  public:
  // 把区间按起始点排序，同起点则再按终点排序
  static bool cmp(const vector<int> &o1, const vector<int> &o2) {
      return o1[0] == o2[0] ? o2[1] < o1[1] : o1[0] < o2[0];
  }
  int removeCoveredIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(),cmp);
    int count = 0;
    int end, prev_end = 0;
    for (auto curr : intervals) {
      end = curr[1];
      if (prev_end < end) {
        ++count;
        prev_end = end;
      }
    }
    return count;
  }
};
```
## 1386. 安排电影院座位
```
如上图所示，电影院的观影厅中有 n 行座位，行编号从 1 到 n ，且每一行内总共有 10 个座位，列编号从 1 到 10 。
给你数组 reservedSeats ，包含所有已经被预约了的座位。比如说，researvedSeats[i]=[3,8] ，它表示第 3 行第 8 个座位被预约了。
请你返回 最多能安排多少个 4 人家庭 。4 人家庭要占据 同一行内连续 的 4 个座位。隔着过道的座位（比方说 [3,3] 和 [3,4]）不是连续的座位，但是如果你可以将 4 人家庭拆成过道两边各坐 2 人，这样子是允许的。
```
```cpp
class Solution {
public:
    // 按行排序
    static bool cmp(vector<int>& seats1,vector<int>& seats2) {
        return seats1[0] < seats2[0];
    }
    int maxNumberOfFamilies(int n, vector<vector<int>>& reservedSeats) {
        sort(reservedSeats.begin(),reservedSeats.end(),cmp);
        int index = 0;// 预定编号
        int result = 0;// 结果
        for (int i=1;i<=n;i++) // 遍历每一行
        {
            bool f1=false,f2=false,f3=false,f4=false;//表示每一个行的23 45 67 89位置有没有被预订
            //检查该行被预订的位置
            while (index<reservedSeats.size()&&reservedSeats[index][0]==i)
            {          
               switch (reservedSeats[index][1])
               {
                   case 2:
                   case 3:
                       f1=true;break;
                   case 4:
                   case 5:
                       f2=true;break;
                   case 6:
                   case 7:
                       f3=true;break;
                   case 8:
                   case 9:
                       f4=true;break;
               }
               index++;
            }
            if(!f1&&!f2&&!f3&&!f4)//可以坐两个人家庭
            {
               result+=2;
            }
            else if(!f1&&!f2||!f3&&!f4||!f2&&!f3) result++;//只能坐一个家庭
            if(index==reservedSeats.size())//已经全部遍历了订票的位置，退出
            {
             result+=(n-i)*2;//没有订票的行最多可以坐两个家庭
             break;
            }
        }
        return result;
    }
};
```
## 122. 买卖股票的最佳时机 II
```
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
```
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {   
        int ans = 0;
        int n = prices.size();
        for (int i = 1; i < n; ++i) {
            // 贪心，只要第二天涨就计入利润
            ans += max(0, prices[i] - prices[i - 1]);
        }
        return ans;
    }
};
```
## 1558. 得到目标数组的最少函数调用次数
![alt 属性文本](https://assets.leetcode.com/uploads/2020/07/10/sample_2_1887.png)
```
给你一个与 nums 大小相同且初始值全为 0 的数组 arr ，请你调用以上函数得到整数数组 nums 。
请你返回将 arr 变成 nums 的最少函数调用次数。
答案保证在 32 位有符号整数以内。
```
```cpp
class Solution {
public:
    // 贪心
    int minOperations(vector<int>& nums) {
        int ret = 0, maxn = 0;
        for (auto num : nums) {
            maxn = max(maxn, num);
            while (num) {// 统计二进制中1的数目
                if (num & 1) {
                    ret++;
                }
                num >>= 1;
            }
        }
        if (maxn) { // 统计最大的那个数所需的操作数
            while (maxn) {
                ret++;
                maxn >>= 1;
            }
            ret--;
        }
        return ret;
    }
};
```
## 1029. 两地调度
```
公司计划面试 2N 人。第 i 人飞往 A 市的费用为 costs[i][0]，飞往 B 市的费用为 costs[i][1]。
返回将每个人都飞到某座城市的最低费用，要求每个城市都有 N 人抵达。
```
```cpp
class Solution {
public:
    // 贪心排序 
    static bool cmp(vector<int> &cost1,vector<int>&cost2)
    {
        return cost1[0]-cost1[1] > cost2[0] - cost2[1];
    }
    int twoCitySchedCost(vector<vector<int>>& costs) {
        sort(costs.begin(),costs.end(),cmp);
        int cost = 0;
        // 前一半是去B更划得来的
        for (int i = 0; i < costs.size() / 2;i++)
        {
            cost += costs[i][1];
        }
        // 后一半是去A更划得来的
        for (int i = costs.size() / 2; i < costs.size();i++)
        {
            cost += costs[i][0];
        }
        return cost;
    }
};
```
## 1276. 不浪费原料的汉堡制作方案
```
圣诞活动预热开始啦，汉堡店推出了全新的汉堡套餐。为了避免浪费原料，请你帮他们制定合适的制作计划。
给你两个整数 tomatoSlices 和 cheeseSlices，分别表示番茄片和奶酪片的数目。不同汉堡的原料搭配如下：
巨无霸汉堡：4 片番茄和 1 片奶酪
小皇堡：2 片番茄和 1 片奶酪
请你以 [total_jumbo, total_small]（[巨无霸汉堡总数，小皇堡总数]）的格式返回恰当的制作方案，使得剩下的番茄片 tomatoSlices 和奶酪片 cheeseSlices 的数量都是 0。
如果无法使剩下的番茄片 tomatoSlices 和奶酪片 cheeseSlices 的数量为 0，就请返回 []。
```
```cpp
class Solution {
public:
    vector<int> numOfBurgers(int tomatoSlices, int cheeseSlices) {
        vector<int> result;
        // 直接解方程 4x+2y=tomato; x+y=cheese;
        int big = (tomatoSlices - 2 * cheeseSlices)/2;
        int small = cheeseSlices - big;
        // 检验解的合法性
        if (big < 0 || small < 0)
            return result;
        if (4*big+2*small!=tomatoSlices||big+small!=cheeseSlices)
            return result;
        result.push_back(big);
        result.push_back(small);
        return result;
    }
};
```
# 动态规划
## 300. 最长上升子序列
```
给定一个无序的整数数组，找到其中最长上升子序列的长度。
示例:
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```
```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n=nums.size();
        if (n == 0) return 0;
        // 记录最长长度
        int maxL = 1;
        vector<int> dp(n, 0);
        for (int i = 0; i < n; ++i) {
            dp[i] = 1;//初始状态
            for (int j = 0; j < i; ++j) {
                if (nums[j] < nums[i]) {
                    // 状态转移方程
                    dp[i] = max(dp[i], dp[j] + 1);
                    if (dp[i]>maxL) maxL = dp[i];
                }
            }
        }
        // 返回最大的那个元素
        return maxL;
    }
};
```
## 64. 最小路径和
```
给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
说明：每次只能向下或者向右移动一步。
```
```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        for (int i = 0;i<grid.size();i++)
        {
            if (i > 0)
                grid[i][0] += grid[i-1][0];
        }
        for (int j = 0;j<grid[0].size();j++)
        {
            if (j > 0)
                grid[0][j] += grid[0][j-1];
        }
        for (int i = 1;i<grid.size();i++)
        {
            for (int j = 1;j<grid[0].size();j++)
            {
                grid[i][j] += min(grid[i-1][j],grid[i][j-1]);
            }
        }
        return grid[grid.size()-1][grid[0].size()-1];
    }
};
```
## 1314. 矩阵区域和
```
给你一个 m * n 的矩阵 mat 和一个整数 K ，请你返回一个矩阵 answer ，其中每个 answer[i][j] 是所有满足下述条件的元素 mat[r][c] 的和： 
i - K <= r <= i + K, j - K <= c <= j + K 
(r, c) 在矩阵内。
```
```cpp
class Solution {
public:
    // 规范矩阵前缀和范围
    int get(const vector<vector<int>>& pre, int m, int n, int x, int y) {
        x = max(min(x, m), 0);
        y = max(min(y, n), 0);
        return pre[x][y];
    }
    vector<vector<int>> matrixBlockSum(vector<vector<int>>& mat, int K) {
        int m = mat.size(), n = mat[0].size();
        vector<vector<int>> P(m + 1, vector<int>(n + 1)); // 矩阵前缀和
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                P[i][j] = P[i - 1][j] + P[i][j - 1] - P[i - 1][j - 1] + mat[i - 1][j - 1];
            }
        }
        vector<vector<int>> ans(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ans[i][j] = get(P, m, n, i + K + 1, j + K + 1) - get(P, m, n, i - K, j + K + 1) - get(P, m, n, i + K + 1, j - K) + get(P, m, n, i - K, j - K);
            }
        }
        return ans;
    }
};
```
## 321. 拼接最大数
```
给定长度分别为 m 和 n 的两个数组，其元素由 0-9 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 k (k <= m + n) 个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。
求满足该条件的最大数。结果返回一个表示该最大数的长度为 k 的数组。
```
```cpp
class Solution {
public:
    vector<int> maxNumber(vector<int> &nums1, vector<int> &nums2, int k) {
        vector<int> ans;
        const int n1 = nums1.size();
        const int n2 = nums2.size();
        for (int k1 = 0; k1 <= k; ++k1) {
            const int k2 = k - k1;
            if (k1 > n1 || k2 > n2) {
                continue;
            }
            ans = max(ans, merge(getMaxNum(nums1, k1), getMaxNum(nums2, k2)));
        }
        return ans;
    }
    // 得到长度为k的最大数组
    vector<int> getMaxNum(const vector<int> &nums, const int k) {
        if (k == 0) {
            return {};
        }
        vector<int> ans;
        int to_pop = nums.size() - k;
        for (const int num : nums) {
            while (!ans.empty() && num > ans.back() && to_pop-- > 0) {
                ans.pop_back();
            }
            ans.push_back(num);
        }
        ans.resize(k);
        return ans;
    }
    vector<int> merge(const vector<int> &nums1, const vector<int> &nums2) {
        vector<int> ans;
        auto s1 = nums1.begin();
        auto e1 = nums1.end();
        auto s2 = nums2.begin();
        auto e2 = nums2.end();
        while (s1 != e1 || s2 != e2) {
            // 按顺序将更大的数连接在前面
            ans.push_back(lexicographical_compare(s1, e1, s2, e2) ? *s2++ : *s1++);
        }
        return ans;
    }
};
```
## 978. 最长湍流子数组
```
当 A 的子数组 A[i], A[i+1], ..., A[j] 满足下列条件时，我们称其为湍流子数组：
若 i <= k < j，当 k 为奇数时， A[k] > A[k+1]，且当 k 为偶数时，A[k] < A[k+1]；
或 若 i <= k < j，当 k 为偶数时，A[k] > A[k+1] ，且当 k 为奇数时， A[k] < A[k+1]。
也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。
返回 A 的最大湍流子数组的长度。
```
```cpp
class Solution {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        vector<vector<int>> dp(arr.size(), vector<int>(2,1));
        int ans = 1;
        for(int i = 1; i < arr.size(); ++i){
            // 大于小于号交替
            if(arr[i] > arr[i-1]){
                dp[i][0] = dp[i-1][1] + 1;
            }else if(arr[i] < arr[i-1]){
                dp[i][1] = dp[i-1][0] + 1;
            }else{
                continue;
            }
            ans = max(ans, max(dp[i][1], dp[i][0]));
        }
        return ans;
    }
};
```
## 494. 目标和
```
给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。
返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
```
```cpp
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        vector<vector<int>> dp(nums.size(),vector<int>(2001,0));
        // dp[i][j+1000] 代表前i个数和为j的情况
        dp[0][nums[0] + 1000] = 1;
        dp[0][-nums[0] + 1000] += 1;
        for (int i = 1; i < nums.size(); i++) {
            for (int sum = -1000; sum <= 1000; sum++) {
                if (dp[i - 1][sum + 1000] > 0) {// 状态转移
                    dp[i][sum + nums[i] + 1000] += dp[i - 1][sum + 1000];
                    dp[i][sum - nums[i] + 1000] += dp[i - 1][sum + 1000];
                }
            }
        }
        return S > 1000 ? 0 : dp[nums.size() - 1][S + 1000];
    }
};
```
## 931. 下降路径最小和
```
给定一个方形整数数组 A，我们想要得到通过 A 的下降路径的最小和。
下降路径可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列。
```
```cpp
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& A) {
        vector<vector<int>> dp(A.size(),vector<int>(A[0].size(),0));
        for (int i=0;i<dp[0].size();i++)
            dp[0][i] = A[0][i];
        for (int i=1;i<dp.size();i++)
        {
            for (int j=0;j<dp[0].size();j++)
                dp[i][j] = A[i][j] + min(dp[i-1][j],
                            min(dp[i-1][j-1>=0?j-1:0],
                            dp[i-1][j+1<dp[0].size()?j+1:dp[0].size()-1]));
        }
        return *min_element(dp[A.size()-1].begin(),dp[A.size()-1].end());
    }
};
```
## 188. 买卖股票的最佳时机 IV
```
给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
```
```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        vector<vector<vector<int>>> dp(prices.size(),vector<vector<int>>(k+1,vector(2,INT32_MIN/2)));
        if (prices.size() == 0) return 0;
        if (k == 0) return 0;
        if (k > prices.size() / 2) k = prices.size() / 2;
        // dp[i][j][k]含义：第i天 交易了j次 k为1持股0不持股
        dp[0][0][0] = 0;
        dp[0][1][1] = -prices[0];
        for (int i = 1; i < prices.size(); i++) {
            for (int j = 0; j <= k; j++) {
                // 如果当前状态没有持有股票
                // 他可以从昨天的没有持有股票的状态转入
                // 也可以从昨天的持有股票状态卖出转入
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1] + prices[i]);
                if (j > 0) {
                    // 如果当前状态持有了股票
                    // 他可以从昨天的持有股票状态转入（持续持有）
                    // 也可以从昨天的没有股票的状态买入转入
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i]);
                }
            }
        }
        int result = 0;
        for (int i=0;i<=k;i++)
            result = max(result,*max_element(dp[prices.size()-1][i].begin(), dp[prices.size()-1][i].end()));
        return result;
    }
};
```
## 309. 最佳买卖股票时机含冷冻期
```
给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
```
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int num = prices.size();
        if (num <= 1) return 0;
        int (*dp)[3] = new int[num][3];
        //dp[i][x]第i天进入(处于)x状态（0.不持股，1.持股，2.冷冻期）
        //不持股
        dp[0][0] = 0;
        //持股
        dp[0][1] = -prices[0];
        //冷冻期
        dp[0][2] = 0;
        for(int i = 1;i < num;i++){
            //第i天不持股可以从两种状态转移而来，1.第i-1天不持股，今天仍不买股票，保持不持              股状态。2.冷冻期结束了，但是今天不买股票。
            dp[i][0] = max(dp[i-1][0], dp[i-1][2]);
            //第i天持股可从两种状态转移而来，1.第i-1天不持股(包含昨天是冷冻期，冷冻期结束后转为不持股状态和昨天本身就不持股这两种情况)，今天买股票。2.第i-1天持股，今天不卖出，保持持股状态。
            dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
            //只有第i-1天卖出了股票，第i天才处于冷冻期。
            dp[i][2] = dp[i-1][1] + prices[i];
        }
        //只有最后一天不持股（不持股状态）或者前一天已经卖掉了（今天为冷冻期）这两种情况手里是拿着钱的，最大值在二者中产生。
        return max(dp[num - 1][0], dp[num - 1][2]);
    }
};
```
## 1504. 统计全 1 子矩形
```
给你一个只包含 0 和 1 的 rows * columns 矩阵 mat ，请你返回有多少个 子矩形 的元素全部都是 1 。
```
```cpp
class Solution {
public:
    int numSubmat(vector<vector<int>>& mat) {
        int n = mat.size();
        int m = mat[0].size();
        // 存放这行左侧到该位置的连续1个数
        vector<vector<int> > left(n,vector<int>(m));
        int now = 0;
        for(int i=0;i<n;i++){
            now = 0;
            for(int j=0;j<m;j++){
                if(mat[i][j] == 1) now ++;
                else now = 0;
                left[i][j] = now;
            }
        }
        int ans = 0,minx;
        //然后对于每个点(i.j)，我们固定子矩形的右下角为(i.j)
        //利用left从该行i向上寻找子矩阵左上角为第k行的矩阵个数
        //每次将子矩阵个数加到答案中
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                minx = INT32_MAX;
                for(int k=i;k>=0;k--){
                    minx = min(left[k][j],minx);
                    ans += minx;
                }
            }
        }
        return ans;
    }
};
```
## 1477. 找两个和为目标值且不重叠的子数组
```
给你一个整数数组 arr 和一个整数值 target 。
请你在 arr 中找 两个互不重叠的子数组 且它们的和都等于 target 。可能会有多种方案，请你返回满足要求的两个子数组长度和的 最小值 。
请返回满足要求的最小长度和，如果无法找到这样的两个子数组，请返回 -1 。
```
```cpp
class Solution {
public:
    int minSumOfLengths(vector<int>& arr, int target) {
        unordered_map<int, int> fuck;   // 记录前缀和出现的最后一个下标
        int t = 0;
        int ans = INT32_MAX;
        vector<int> m_f;   // 记录到每个位置，和为target的最短子数组的长度
        int mi = 0;    
        fuck[0]  = -1;
        for(int i = 0; i < arr.size(); i ++){
            t += arr[i];
            fuck[t] = i;
            // cout << i << " ";
            if(fuck.count(t - target)){
                int l_i = fuck[t - target];
                int ttt = i - l_i;
                if(! mi) mi = ttt;
                else mi = min(mi, ttt);
                m_f.push_back(mi);
                if(l_i != -1 && m_f[l_i] != 0) ans = min(ans, m_f[l_i] + ttt);
            }else{
                m_f.push_back(mi);
            }
        }
        if(ans == INT32_MAX) return -1;
        else return ans;
    }
};
```
## 1575. 统计所有可行路径
```
给你一个 互不相同 的整数数组，其中 locations[i] 表示第 i 个城市的位置。同时给你 start，finish 和 fuel 分别表示出发城市、目的地城市和你初始拥有的汽油总量
每一步中，如果你在城市 i ，你可以选择任意一个城市 j ，满足  j != i 且 0 <= j < locations.length ，并移动到城市 j 。从城市 i 移动到 j 消耗的汽油量为 |locations[i] - locations[j]|，|x| 表示 x 的绝对值。
请注意， fuel 任何时刻都 不能 为负，且你 可以 经过任意城市超过一次（包括 start 和 finish ）。
请你返回从 start 到 finish 所有可能路径的数目。
由于答案可能很大， 请将它对 10^9 + 7 取余后返回。
```
```cpp
class Solution {
public:
    long long mod = 1e9+7;// 答案取余值
    long long maxFuel = 200;// 输入限制大小
    int countRoutes(vector<int>& loc, int start, int finish, int fuel) {
        // dp[i][k]表示花正好k的油到达i点的路径数。
        // 对于每条路径，我们可以再从i走到j，花费abs(loc[i]-loc[j])。
        // dp[j][k+abs(loc[i]-loc[j])] += dp[i][k]
        long sum = accumulate(loc.begin(),loc.end(),0l);
        vector<vector<long long>> dp(loc.size()+1,vector<long long>(maxFuel+1,0));
        int n = loc.size();
        dp[start][0] = 1;
        for(int k=0;k<fuel;k++){
            for(int i=0;i<n;i++){
                for(int j=0;j<n;j++){
                    if(j==i) continue;
                    if(k+abs(loc[i]-loc[j]) > maxFuel) continue;
                    dp[j][k+abs(loc[i]-loc[j])] += dp[i][k];
                    dp[j][k+abs(loc[i]-loc[j])] %= mod;
                }
            }
        }
        long long ans = 0;
        for(int k=0;k<=fuel;k++){
            ans += dp[finish][k];
            ans %= mod;
        }
        return ans;
        
    }
};
```
## 1423. 可获得的最大点数
PS: 本题使用C++版dp只能通过33/40个测试用例，java正常。
```
几张卡牌 排成一行，每张卡牌都有一个对应的点数。点数由整数数组 cardPoints 给出。
每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 k 张卡牌。
你的点数就是你拿到手中的所有卡牌的点数之和。
给你一个整数数组 cardPoints 和整数 k，请你返回可以获得的最大点数。
```
```cpp
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        //dp[i][j] 代表开头取i张 末尾取j张
        vector<vector<int>> dp(k+1,vector<int>(k+1,0));
        for (int i=1;i<=k;i++)
        {
            dp[0][i]=dp[0][i-1]+cardPoints[cardPoints.size()-i];
            dp[i][0]=dp[i-1][0]+cardPoints[i-1];
        }
        int maxNum = INT32_MIN;
        for (int i=0;i<=k;i++)
        {
            maxNum = max(maxNum,dp[i][0]+dp[0][k-i]);
        }
        return maxNum;
    }
};
```