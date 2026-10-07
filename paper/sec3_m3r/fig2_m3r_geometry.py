#!/usr/bin/env python
"""Fig. 2: schematic of the M3R map for a toy ensemble of equal-weight three-constituent jets.

The two-parameter family interpolates between one-, two- and three-prong jets: u opens a
second prong and v splits it, so the three references have different prongness. Right-hand
coordinates are exact W2 distances to the three reference jets; the left surface is only an
illustrative display of the family. Needs no data. Writes figures/m3r_geometry.pdf.
"""

from __future__ import annotations

import argparse
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon  # noqa: E402

from m3r.plotting.style import savefig  # noqa: E402


def draw() -> plt.Figure:
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral'],
                         'mathtext.fontset': 'stix', 'svg.fonttype': 'path', 'pdf.fonttype': 42, 'font.size': 14})
    fig,ax=plt.subplots(figsize=(14.4,7.0))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax.set(xlim=(0,14.4),ylim=(0,7.0),aspect='equal'); ax.axis('off')
    ink='#222B37'; grid='#9CA3AF'
    surface_face='#D4D7DC'; patch_face='#7B8594'
    colors=['#0072B2','#D55E00','#009E73']
    # Display coordinates t in [0,1]^2; r1 = (0,0) is a single-particle jet.
    refs=np.array([[0.,0.],[.76,.03],[.74,.76]])
    sample=np.array([.45,.4]); neighbor=np.array([.68,.56])
    perms=list(permutations(range(3)))
    
    def jet(uv):
        # u separates two prongs, v splits the second one: (0,0) is a single
        # particle (all three constituents coincide), (1,0) is 2-prong, and (1,1)
        # is 3-prong. Centered at the origin.
        a=.54*uv[0]; b=.57*uv[1]
        pts=np.array([[-a,0.],[a,-b],[a,b]])
        return pts-pts.mean(axis=0)
    def distance(a,b):
        ja,jb=jet(a),jet(b)
        return np.sqrt(min(np.mean(np.sum((ja-jb[list(p)])**2,axis=1)) for p in perms))
    def phi(uv):
        return np.array([distance(uv,r) for r in refs])
    def left(uv):
        u,v=np.asarray(uv)
        h=.42*np.sin(np.pi*u)*np.cos(np.pi*v)
        return np.array([3.52+3.1*(u-.5)-1.8*(v-.5),
                         3.5+.68*(u-.5)+1.42*(v-.5)+h])
    # Orthographic view of the three distance coordinates: rotate by AZIM about the
    # vertical d(., r3) axis, tilt by ELEV, then scale so that the mapped manifold
    # and the axes (just longer than the manifold) fill the panel.
    AZIM=np.deg2rad(25.); ELEV=np.deg2rad(35.)
    M0=np.array([[np.cos(AZIM),-np.sin(AZIM),0.],
                 [np.sin(AZIM)*np.sin(ELEV),np.cos(AZIM)*np.sin(ELEV),np.cos(ELEV)]])
    n=17; ticks=np.linspace(0,1,n)
    grid_d=np.array([phi([u,v]) for u in ticks for v in ticks])
    axis_lengths=grid_d.max(axis=0)*.9
    proj=M0@np.vstack([grid_d,np.diag(axis_lengths),np.zeros(3)]).T
    lo,hi=proj.min(axis=1),proj.max(axis=1)
    box_lo,box_hi=np.array([8.1,1.25]),np.array([13.5,5.95])
    proj_scale=np.min((box_hi-box_lo)/(hi-lo))
    proj_offset=box_lo+((box_hi-box_lo)-proj_scale*(hi-lo))/2-proj_scale*lo
    def right_from_d(d):
        return proj_offset+proj_scale*M0@d
    def right(uv): return right_from_d(phi(uv))
    
    ax.text(3.5,6.64,r'Event Manifold $(\mathcal{X},d_\alpha)$',ha='center',fontsize=20,color=ink)
    ax.text(10.85,6.64,'M3R Space',ha='center',fontsize=20,color=ink)
    
    # Same sampled family is drawn on both sides of the map.
    # The surface is shaded on a fine mesh for smoothness; grid lines are drawn
    # along every STEP-th fine line, i.e. on the same n x n grid as before.
    N_FINE=65; fine=np.linspace(0,1,N_FINE); STEP=(N_FINE-1)//(n-1)
    def surface(mapper, mapped=False):
        pts=np.array([[mapper([u,v]) for v in fine] for u in fine])
        cells=[[pts[i,j],pts[i+1,j],pts[i+1,j+1],pts[i,j+1]]
               for i in range(N_FINE-1) for j in range(N_FINE-1)]
        ax.add_collection(PolyCollection(cells,facecolors=surface_face,edgecolors='none',
                          antialiased=False,alpha=.5 if mapped else .85,zorder=2))
        lines=[pts[i,:] for i in range(0,N_FINE,STEP)]+[pts[:,j] for j in range(0,N_FINE,STEP)]
        ax.add_collection(LineCollection(lines,colors=grid,linewidths=.3 if mapped else .36,
                          alpha=.75,zorder=2))
        if mapped:
            return
        # Show a neighborhood around x, with the same parameter patch on both sides.
        ang=np.linspace(0,2*np.pi,80)
        patch=np.array([mapper(sample+.16*np.array([np.cos(t),np.sin(t)])) for t in ang])
        ax.add_patch(Polygon(patch,facecolor=patch_face,edgecolor='#8193AC',
                             alpha=.45,lw=.8,zorder=4))
    surface(left)
    # A transparent image of the same family.
    surface(right, mapped=True)
    
    # Distance-coordinate axes: reference events are NOT placed at axis tips.
    origin=right_from_d(np.zeros(3))
    for k,c in enumerate(colors):
        unit=np.zeros(3); unit[k]=axis_lengths[k]
        end=right_from_d(unit)
        ax.add_patch(FancyArrowPatch(origin,end,arrowstyle='-|>',mutation_scale=13,
                                    lw=1.3,color=c,zorder=5))
        shifts=[(0,-.28),(.1,.3),(.12,.24)]
        shift=np.array(shifts[k])
        ax.text(*(end+shift),rf'$d_\alpha(\,\cdot\,,r_{k+1})$',color=c,fontsize=17,
                ha='center',va='center',zorder=8)
    
    # Reference events remain on the original manifold only.
    for mapper in (left,):
        for k,c in enumerate(colors):
            p=mapper(refs[k])
            ax.scatter(*p,s=80,facecolor=c,edgecolor='white',lw=1,zorder=7)
        a=mapper(sample)
        ax.scatter(*a,s=58,c=ink,edgecolor='white',lw=.8,zorder=9)
    
    # Axis-aligned coordinate box with x at the opposite corner from the origin.
    d1,d2,d3=phi(sample)
    P=np.array([d1,d2,d3])
    A=np.array([d1,0.,0.]); B=np.array([0.,d2,0.]); C=np.array([0.,0.,d3])

    for bits in product((0,1),repeat=3):
        for k in range(3):
            if bits[k]:
                continue
            end_bits=list(bits); end_bits[k]=1
            a=right_from_d(P*np.array(bits))
            b=right_from_d(P*np.array(end_bits))
            ax.plot([a[0],b[0]],[a[1],b[1]],color='#858585',lw=1.05,
                    linestyle=(0,(4,3)),alpha=.85,zorder=4)
    for k,coord in enumerate((A,B,C)):
        tick=right_from_d(coord)
        ax.plot([origin[0],tick[0]],[origin[1],tick[1]],color=colors[k],lw=2.6,zorder=5)
        ax.scatter(*tick,s=30,color=colors[k],zorder=6)
        offsets=[(.16,-.33),(-.42,-.08),(-.45,.04)]
        ax.text(*(tick+offsets[k]),rf'$d_\alpha(x,r_{k+1})$',color=colors[k],fontsize=17,
                ha='center',va='center',zorder=9,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
    ax.scatter(*right(sample),s=80,c=ink,edgecolor='white',lw=1.1,zorder=10)
    # Images of the references: Phi(r_k) has d(r_k,r_k)=0, so the manifold touches
    # the coordinate plane spanned by the other two axes there.
    ref_labels=[r'$(0,\,\ast,\,\ast)$',r'$(\ast,\,0,\,\ast)$',r'$(\ast,\,\ast,\,0)$']
    ref_offsets=[((0,.26),'center'),((.3,-.1),'left'),((.18,-.22),'left')]
    for k,c in enumerate(colors):
        p=right(refs[k])
        ax.scatter(*p,s=70,facecolor=c,edgecolor='white',lw=1,zorder=10)
        (dx,dy),ha=ref_offsets[k]
        ax.text(p[0]+dx,p[1]+dy,ref_labels[k],color=c,fontsize=17,ha=ha,va='center',
                zorder=10,bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
    
    # Reference-distance comparisons from x on the event manifold.
    for k,c in enumerate(colors):
        pts=np.array([left((1-t)*sample+t*refs[k]) for t in np.linspace(0,1,70)])
        ax.plot(pts[:,0],pts[:,1],ls=(0,(3,3)),lw=1.4,color=c,zorder=5)
    
    ax.text(*(left(sample)+[-.17,.19]),'$x$',fontsize=20,color=ink)
    ax.text(*(right(sample)+[.15,.25]),r'$\Phi_{\mathcal{R}}^{(\alpha)}(x)$',fontsize=20,color=ink,
            zorder=10,bbox=dict(facecolor='white',edgecolor='none',pad=1.2))
    
    glyph_scale=.36/max(np.abs(jet(uv)).max() for uv in list(refs)+[sample])
    def glyph(center,uv,color,label,radius=.48):
        ax.add_patch(Circle(center,radius,facecolor='#FAFBFC',edgecolor=color,lw=1.3,zorder=10))
        for xy in jet(uv):
            ax.add_patch(Circle(np.array(center)+xy*glyph_scale,.075,facecolor=color,
                                edgecolor='white',lw=.5,zorder=11))
        ax.text(center[0],center[1]+radius+.13,label,ha='center',fontsize=20,color=color)
    
    # Jet-image insets identify event points without reusing the reference artwork.
    insets=[(1.15,1.57),(5.86,5.32),(.95,5.32)]
    for k,(center,c) in enumerate(zip(insets,colors)):
        target=left(refs[k]); start=np.array(center)
        direction=(target-start)/np.linalg.norm(target-start)
        ax.plot([start[0]+direction[0]*.49,target[0]],
                [start[1]+direction[1]*.49,target[1]],color=c,alpha=.6,lw=.9,zorder=1)
        glyph(center,refs[k],c,rf'$r_{k+1}$')
    center=np.array([5.7,1.48]); target=left(sample)
    direction=(target-center)/np.linalg.norm(target-center)
    ax.plot([center[0]+direction[0]*.49,target[0]],
            [center[1]+direction[1]*.49,target[1]],color=ink,alpha=.45,lw=.9,zorder=1)
    glyph(center,sample,ink,'$x$')
    
    ax.add_patch(FancyArrowPatch((6.2,3.68),(7.2,3.68),arrowstyle='-|>',
                                lw=1.8,mutation_scale=20,color=ink,zorder=10))
    ax.text(6.7,4.03,r'$\Phi_{\mathcal{R}}^{(\alpha)}$',ha='center',fontsize=25,color=ink)
    ax.text(10.9,.5,r'$x\ \mapsto\ \left(d_\alpha(x,r_1),\ d_\alpha(x,r_2),\ d_\alpha(x,r_3)\right)$',
            ha='center',fontsize=22,color=ink)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf"], help="Also write svg/png with the same stem.")
    args = parser.parse_args()
    cfg = load_cfg(args)
    out = args.out or cfg.figure_path(cfg["figures"]["fig2"])
    fig = draw()
    for ext in args.formats:
        path = out.with_suffix(f".{ext}")
        savefig(fig, path, dpi=240, facecolor="white", bbox_inches=None)
        print(f"wrote {describe(path, cfg)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
