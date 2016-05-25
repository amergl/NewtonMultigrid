# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


if __name__ == "__main__":

    rc('font', family='sans-serif', size=32)
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    fig = plt.subplots(figsize=(15, 8))

    ndofs = 256

    plotlist = [ (1.0, '1', 'red'), (2.0/3, '2/3', 'orange'), (1.0/2, '1/2', 'blue'), (1.0/3, '1/3', 'green') ]

    xvals = np.array( [k*np.pi/ndofs for k in range(ndofs)] )

    for w,l,color in plotlist:
        print(w,l,color)

        ev = np.array( [ 1 - 2*w*np.sin(xvals[k]/2)**2 for k in range(ndofs) ] )

        label = '$w=$'+l
        plt.plot(xvals, ev, lw=3, label=label, color=color)

    plt.plot(xvals,[0]*ndofs, lw=3, color='black')

    plt.axis([0,np.pi,-1.1,1.1])
    # plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi], ['$0$','$\pi/4$','$\pi/2$','$3\pi/4$','$\pi$'])
    plt.xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi], ['$0$', '$N/4$', '$N/2$', '$3N/4$', '$N-1$'])
    plt.yticks([-1,-0.5,0,0.5,1], ['$-1$', '$-1/2$', '$0$', '$1/2$', '$1$'])
    plt.xlabel('$k$')
    plt.ylabel('$\lambda_k$')

    plt.legend(loc=0)
    plt.grid()
    plt.tight_layout()

    plt.savefig('eigenvalues_jacobi.pdf', rasterized=True, transparent=True, bbox_inches='tight')

    # plt.show()
