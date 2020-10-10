#include <iostream>

const int N = 50;//集装箱数量
const int ShipWeitht = 500;//货船载重量
int Weights[N];//集装箱重量
int RestWeight[N];//剩下的集装箱总重

//初始化剩下的集装箱总重
void InitRestWeight()
{
	RestWeight[N - 1] = Weights[N - 1];
	for (int i = N - 2; i >= 0; i--)
		RestWeight[i] = RestWeight[i + 1] + Weights[i];
}
//判断能否右剪枝 i为当前集装箱 CurrentWeight为当前货船载重
bool RightCut(int i, int CurrentWeight)
{
	return CurrentWeight + RestWeight[i] - Weights[i] < ShipWeitht;
}
//判断能否左剪枝 i为当前集装箱 CurrentWeight为当前货船载重
bool LeftCut(int i, int CurrentWeight)
{
	return CurrentWeight + Weights[i] > ShipWeitht;
}
//递归求解，i为层数，cw为当前载重
bool DFS(int i,int cw)
{
	if (cw == ShipWeitht) return true;
	if (i >= N) return false;
	
	bool LeftHasResult = false;//记录左节点递归是否有解
	bool RightHasResult = false;//记录右节点递归是否有解
	if (!LeftCut(i,cw))//不满足左剪枝则扩展左节点
	{
		LeftHasResult = DFS(i + 1, cw + Weights[i]);
		if (LeftHasResult) return true;
	}
	if (!RightCut(i, cw))//不满足右剪枝则扩展右节点
	{
		RightHasResult = LeftHasResult = DFS(i + 1, cw);
		if (RightHasResult) return true;
	}
	return false;//该节点左右递归都无解
}
