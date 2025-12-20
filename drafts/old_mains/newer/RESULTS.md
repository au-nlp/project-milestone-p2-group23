
Selected some hardcoded ideas which are specific to some events in the may-jure 2020 interval but not that related:

ideas = [  # Specific ideas
    "The FDA issued an Emergency Use Authorization (EUA) for remdesivir for severe COVID-19 cases, boosting antiviral focus.",  # 2020-05-01, "GILD", "MRNA", "XLV", "IBB"
    "At Berkshire Hathaway’s annual meeting, Warren Buffett said Berkshire fully exited its major U.S. airline stakes, signaling prolonged aviation stress.",  # 2020-05-02, "BRK.B", "AAL", "DAL", "UAL", "LUV"
    ...
]

Mean correlation between idea vectors:  0.12504032



Comparing the pooling method per episodes
Initially max vs mean
Pretty decen for retriving similar text.

### Using mean:

Our of the top 50 episode matching:

Mean of number of keyword Apple:  2.64
Episodes with number of keyword Apple = 0:  35
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  1.3
Episodes with number of keyword Google = 0:  27

So most of them don't even contain the keyword Apple, which definately means that meaning looses too much information, and is up to which has better noise to mean over.


Mean correlation between idea scores:  0.33882461906543054 (these represents how correlated the idea evolutions are using a simple .corr() on the df) (I guess they should be as different as possible (similar to the mean correlation between idea vectors at best))

### When using the mean max method initially seem much better:


Mean of number of keyword Apple:  8.9
Episodes with number of keyword Apple = 0:  10
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.2
Episodes with number of keyword Google = 0:  13

Most are actually related to apple, containg a good average of 9. The would be fine if max wouldn't be so sensible to the episode length. The longer the episode, the higher chance to conain random segment which matches more with the idea. Because of that, the expected score is a lot higher for longer episodes, which is not good.

Mean correlation between idea scores:  0.4568393364998916


The mean correlation between the idea and the length of the episode for the 20 preselected ideas is:

np.float64(0.34402598735067536)
Which is far to big. We don't want to corelate the length of the episodes with the stocks, but rather the content.




Similar to the max pooling is the topk pooling(max being a particular case with k=1)
Setting the K higher, be it 5 we get:

Mean of number of keyword Apple:  9.58
Episodes with number of keyword Apple = 0:  13
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.64
Episodes with number of keyword Google = 0:  16

Mean correlation between idea scores:  0.5330044839312418


The mean correlation between the idea and the length of the episode for the 20 preselected ideas is:

k = 1: np.float64(0.34402598735067536)
k = 3: np.float64(0.39258888104767675)
k = 5: np.float64(0.4174914789785271)
k = 7: np.float64(0.4299087297985579)
k = 10: np.float64(0.43650736886660607)
k = 15: np.float64(0.43101232296147274)
k = 20: np.float64(0.41040899967242195)





Mean of number of keyword Apple:  7.9
Episodes with number of keyword Apple = 0:  13
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.44
Episodes with number of keyword Google = 0:  14

logsumexp(lse):

LSE_TAU = 0.01:

Mean of number of keyword Apple:  8.9
Episodes with number of keyword Apple = 0:  10
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.2
Episodes with number of keyword Google = 0:  13

LSE_TAU = 0.03:

Mean of number of keyword Apple:  9.78
Episodes with number of keyword Apple = 0:  11
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.08
Episodes with number of keyword Google = 0:  15


LSE_TAU = 0.07:

Mean of number of keyword Apple:  8.76
Episodes with number of keyword Apple = 0:  12
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.64
Episodes with number of keyword Google = 0:  14


LSE_TAU = 0.10:

Mean of number of keyword Apple:  7.9
Episodes with number of keyword Apple = 0:  13
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  3.44
Episodes with number of keyword Google = 0:  14


LSE_TAU = 0.07:

Mean correlation between idea scores:  0.5545483182848722

Corelation between scores and length: np.float64(0.4402662831931461)


Pct_above:

SIM_THRESHOLD: 0.2:

Mean of number of keyword Apple:  2.4
Episodes with number of keyword Apple = 0:  36
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  0.98
Episodes with number of keyword Google = 0:  33


Mean correlation between idea scores:  0.2039022500955801

Corelation between scores and length: -0.04727895332137326

SIM_THRESHOLD: 0.3:

Mean of number of keyword Apple:  5.98
Episodes with number of keyword Apple = 0:  26
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  2.44
Episodes with number of keyword Google = 0:  28

Mean correlation between idea scores:  0.07263122520963884

Corelation between scores and length: -0.02834179152603842



SIM_THRESHOLD: 0.4:

Mean of number of keyword Apple:  7.5
Episodes with number of keyword Apple = 0:  22
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  2.68
Episodes with number of keyword Google = 0:  23


SIM_THRESHOLD: 0.35:

Mean of number of keyword Apple:  8.38
Episodes with number of keyword Apple = 0:  18
Mean of number of keyword AAPL:  0.0
Episodes with number of keyword AAPL = 0:  50
Mean of number of keyword Google:  2.18
Episodes with number of keyword Google = 0:  21


The threshold is too big and some ideas have 0 score (nan in the code)
