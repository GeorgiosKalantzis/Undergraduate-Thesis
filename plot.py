import matplotlib.pyplot as plt


c = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
rl = [0.846,0.823,0.786,0.734,0.775,0.707,0.709,0.728,0.694,0.663,0.66,0.663,0.623]

prr = [0.729,0.716,0.695,0.687,0.656,0.644,0.656,0.649,0.624,0.601,0.599,0.587,0.597]
rll = [0.878,0.851,0.832,0.819,0.794,0.755,0.773,0.784,0.739,0.723,0.723,0.700,0.706]
mj = [346,349,352,356,358,364,362,360,362,371,371,364,372]

ippr = [1.0,0.707,0.149,0.178,0.521,0.028,0.0097,0.0057,0.027,0.018,0.043,0.03,0.043]
irll = [1.0,0.698,0.141,0.185,0.484,0.026,0.016,0.0084,0.041,0.028,0.048,0.032,0.047]

mj = [(1-(i/491))*100 for  i in mj]
genetic = [18.7]*len(c)

plt.plot(c,ippr,marker = 'o', color = 'red',label = 'Improved')
plt.plot(c,prr,marker = 'o', color = 'black',label='Deteriorated')
plt.legend()
plt.ylim(0,1.1)

plt.title('Combarison with original graph')

plt.xlabel('Number of Changed Edges')

plt.ylabel('Precision(%)')


plt.show()

