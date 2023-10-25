# 搜索

[CF1873H - Mad City](https://codeforces.com/contest/1873/problem/H) (1)
{.annotate}

1. 先对b做深度优先搜索找到进入环的位置，再计算这个位置到a和b的距离，如果离b更近则b永远无法被抓到。[参考题解](https://codeforces.com/contest/1873/submission/224446642)。
