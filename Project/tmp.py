from scipy import stats
print(stats.norm.cdf(0))

rows = 28
t = [round(stats.norm.cdf(i*2.4/28-1.2)-0.5, 2) for i in range(28)]
print(t)
