#include <iostream>

const int N = 50;//��װ������
const int ShipWeitht = 500;//����������
int Weights[N];//��װ������
int RestWeight[N];//ʣ�µļ�װ������

//��ʼ��ʣ�µļ�װ������
void InitRestWeight()
{
	RestWeight[N - 1] = Weights[N - 1];
	for (int i = N - 2; i >= 0; i--)
		RestWeight[i] = RestWeight[i + 1] + Weights[i];
}
//�ж��ܷ��Ҽ�֦ iΪ��ǰ��װ�� CurrentWeightΪ��ǰ��������
bool RightCut(int i, int CurrentWeight)
{
	return CurrentWeight + RestWeight[i] - Weights[i] < ShipWeitht;
}
//�ж��ܷ����֦ iΪ��ǰ��װ�� CurrentWeightΪ��ǰ��������
bool LeftCut(int i, int CurrentWeight)
{
	return CurrentWeight + Weights[i] > ShipWeitht;
}
//�ݹ���⣬iΪ������cwΪ��ǰ����
bool DFS(int i,int cw)
{
	if (cw == ShipWeitht) return true;
	if (i >= N) return false;
	
	bool LeftHasResult = false;//��¼��ڵ�ݹ��Ƿ��н�
	bool RightHasResult = false;//��¼�ҽڵ�ݹ��Ƿ��н�
	if (!LeftCut(i,cw))//���������֦����չ��ڵ�
	{
		LeftHasResult = DFS(i + 1, cw + Weights[i]);
		if (LeftHasResult) return true;
	}
	if (!RightCut(i, cw))//�������Ҽ�֦����չ�ҽڵ�
	{
		RightHasResult = LeftHasResult = DFS(i + 1, cw);
		if (RightHasResult) return true;
	}
	return false;//�ýڵ����ҵݹ鶼�޽�
}
