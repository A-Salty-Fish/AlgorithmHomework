#pragma once
const int INF = 1<<30;
const int MAXV = 10;
int dist[MAXV], path[MAXV];
struct MGraph
{
    int n;
    int edges[MAXV][MAXV];
};
MGraph g; int v;
void showPath(int s, int t){}
void BellmanFord(MGraph g, int v)
{
    int i, k, u;
    for (i = 0; i < g.n; i++)
    {
        dist[i] = g.edges[v][i];		// 对dist(1)[i]初始化
        if (i != v && dist[i] < INF)
            path[i] = v;			// 对path(1)[i]初始化
        else
            path[i] = -1;
    }
    for (k = 2; k < g.n; k++) // 松弛操作
    {
        for (u = 0; u < g.n; u++)
        {
            if (u != v)
            {
                for (i = 0; i < g.n; i++)
                {
                    if (g.edges[i][u] < INF &&
                        dist[u] < dist[i] + g.edges[i][u])
                    {
                        dist[u] = dist[i] + g.edges[i][u];
                        path[u] = i;
                    }
                }
            }
        }
    }
}
void maxPath(int s, int t) {
    BellmanFord(g, v);
    showPath(s, t);
}

