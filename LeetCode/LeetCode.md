[toc]
# 分治法
## 240.搜索二维矩阵II
```
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
```
```cpp
class Solution {
public:
    bool search_submatrix(vector<vector<int>>& matrix, int target,
    int left,int right,int up,int down)
    {
        if (left > right || up > down)// 搜索完
            return false;
        else if (target < matrix[up][left] || target > matrix[down][right])// 比最小值小或比最大值大
            return false;
        else
        {
            int row = up;
            int mid = (left + right);
            // 尝试将搜索范围缩小到左下角和右上角
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
## 315.计算右侧小于当前元素的个数(归并排序+索引数组)
```
给定一个整数数组 nums，按要求返回一个新数组 counts。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。
```
```java
class Solution {
    int[] counter;
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> res=new ArrayList<>(nums.length);
        if (nums.length<1)
            return res;
        //初始化索引
        int[] indexes=new int[nums.length];
        counter=new int[nums.length];
        for (int i=0;i<nums.length;++i)
            indexes[i]=i;
        helper(nums,indexes,0,nums.length-1);
        for (int item:counter)
            res.add(item);
        return res;
    }

    public void helper(int[] nums,int[] indexes,int start,int end){
        if(start>=end)
            return;
        int mid=start+(end-start)/2;
        helper(nums,indexes,start,mid);
        helper(nums,indexes,mid+1,end);
        if (nums[mid]>nums[mid+1])
            merge(nums,indexes,start,mid,end);
    }

    public void merge(int[] nums,int[] indexes,int start,int mid,int end){
        int left=start,right=mid+1;
        int[] merge=new int[end-start+1];
        int[] tmpIndex=new int[end-start+1];
        int p=0;
        for (;left<=mid&&right<=end;) {
            if (nums[right]>=nums[left]){
                merge[p]=nums[left];
                tmpIndex[p++]=indexes[left];
                counter[indexes[left++]]+=(right-mid-1);
            } else{//nums[lef]>nums[right]
                merge[p]=nums[right];
                tmpIndex[p++]=indexes[right++];
            }
        }
        while (left<=mid) {
            merge[p]=nums[left];
            tmpIndex[p++]=indexes[left];
            counter[indexes[left++]]+=(right-mid-1);
        }
        while (right<=end) {
            merge[p]=nums[right];
            tmpIndex[p++]=indexes[right++];
        }
        //复制回原数组
        p=0;
        for (int i=start;i<=end;++i) {
            nums[i]=merge[p];
            indexes[i]=tmpIndex[p++];
        }
    }
}
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
        vector<vector<int> > result;
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
## 698. 划分为k个相等的子集(写不来)
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
    
    //当前从id位置开始枚举,还有有个sz子集未合法是否能够使所有子集合法
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
            //cout << "****** retCnt = " << retCnt << ", n:" << n << endl;
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
