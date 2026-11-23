"""Renderização vetorial respeitando o estado de caneta."""

def draw(ax,xy,pen,color,width=1.5,alpha=1):
    begin=0
    for i in range(len(xy)):
        if pen[i] or i==len(xy)-1:
            ax.plot(xy[begin:i+1,0],xy[begin:i+1,1],color=color,linewidth=width,alpha=alpha)
            begin=i+1
