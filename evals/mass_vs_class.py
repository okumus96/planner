"""M_cas KUTLESI vs L1 SINIFI -- "none ajanlar hak etmedikleri agirligi aliyor mu?"

NE ICIN: L_mass'in cozmesi gereken problemi OLCMEK ve GORMEK. Beklenti: yieldingTo/waitingFor
etiketli ajan, ayni sahnedeki 'none' etiketli ajandan DAHA COK kutle almali.

NASIL: yalniz EN AZ BIR non-none GT ajani olan sahneler sayilir -- hepsi none olan sahnede
"kutle none'da" demek totolojidir, oradan bilgi cikmaz.

METRIK:
  pay        : sahnedeki M_cas kutlesinin sinif basina yuzdesi (yuvalar sayisina gore normalize DEGIL)
  top1 dogru : sahnenin en yuksek M_cas'li ajani non-none mi
  ihlal      : bir none ajani, ayni sahnenin en oncelikli non-none ajanindan daha fazla kutle aliyor
"""
import argparse, os, sys, math, collections
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader, Subset
from GameFormer.predictor import GameFormer
from GameFormer.causal_graph import CausalPlanner
from GameFormer.train_utils import DrivingData
from GameFormer.decision_labels import NUM_LON4, NUM_LAT5V
from train_planner import read_batch, extract_neighbor_top1_futures, freeze_gameformer

CLS = ['none','follows','yieldingTo','waitingFor','mergesInFrontOf','overtakes']
PRIO = [2,3,4,5,1]                      # yield > wait > merge > overtake > follow
COL  = ['#9aa0aa','#4c78a8','#d1495b','#c9781f','#5a8f5a','#8a63a8']


def box(ax,x,y,th,L,W,**kw):
    c,s = math.cos(th), math.sin(th)
    ax.add_patch(Rectangle((x-(L/2)*c+(W/2)*s, y-(L/2)*s-(W/2)*c), L, W,
                           angle=math.degrees(th), **kw))


def main(a):
    dev=a.device
    gf=GameFormer(encoder_layers=3,decoder_levels=2,neighbors=10)
    gf.load_state_dict(torch.load(a.pretrained_path,map_location=dev)); gf=gf.to(dev); freeze_gameformer(gf)
    m=CausalPlanner(layers=1,modes=6,nbr_enrich=2,ego_residual=0,gate_channels=1,typed_kv=1,
                    dod_meta=1,lat_moe=1,num_lon=NUM_LON4,num_lat=NUM_LAT5V,
                    l1=1,l1_bottleneck=1,num_l1_ag=6).to(dev)
    m.load_state_dict(torch.load(a.causal_path,map_location=dev),strict=False); m.eval()
    ds=DrivingData(a.valid_set+"/*.npz",10,l1_labels=a.l1_labels)
    files=sorted(__import__('glob').glob(a.valid_set+"/*.npz"))
    n=min(a.limit or len(ds), len(ds))

    mass=collections.defaultdict(list); top1=[]; viol=[]; per_scene=[]
    idx=0
    for b in DataLoader(Subset(ds,list(range(n))),batch_size=64,num_workers=4):
        inp,ef,nf,rp=read_batch(b,dev); B=ef.shape[0]
        with torch.no_grad():
            enc=gf.encoder(inp); t1,ns,_=extract_neighbor_top1_futures(gf,enc,10)
            d=m.disentangler(enc['agent_tokens'][:,:11].detach(),~enc['mask'][:,:11],
                enc['actors'][:,:11,-1].detach(),torch.zeros(B,11,dtype=torch.long,device=dev),
                inp,neighbor_futures=t1,neighbor_states=ns)
        gv=d['gated_valid']; Mc=(gv.float()*d['M_cas']).cpu().numpy()
        y=inp['l1_agent'][:,:Mc.shape[1]].cpu().numpy(); gvn=gv.cpu().numpy()
        for i in range(B):
            si=idx+i
            if not gvn[i].any() or not ((y[i]>0)&gvn[i]).any():   # non-none ajani OLMAYAN sahne atlanir
                continue
            tot=Mc[i].sum()
            if tot<1e-6: continue
            p=Mc[i]/tot
            for c in range(6):
                s=p[(y[i]==c)&gvn[i]].sum()
                if ((y[i]==c)&gvn[i]).any(): mass[c].append(s)
            best_nn=max([c for c in PRIO if ((y[i]==c)&gvn[i]).any()],
                        key=lambda c: -PRIO.index(c))
            j_nn=int(np.argmax(np.where((y[i]==best_nn)&gvn[i],p,-1)))
            j_top=int(np.argmax(np.where(gvn[i],p,-1)))
            top1.append(y[i][j_top]>0)
            nones=np.where((y[i]==0)&gvn[i])[0]
            if len(nones) and p[nones].max()>p[j_nn]:
                viol.append(dict(scene=si,file=os.path.basename(files[si]),
                                 j_none=int(nones[np.argmax(p[nones])]), p_none=float(p[nones].max()),
                                 j_nn=j_nn, p_nn=float(p[j_nn]), cls_nn=CLS[best_nn]))
            per_scene.append((p[(y[i]==0)&gvn[i]].sum(), p[(y[i]>0)&gvn[i]].sum()))
        idx+=B

    print(f"\n=== M_cas KUTLESI vs L1 SINIFI ===")
    print(f"sayilan sahne (en az bir non-none ajani olan): {len(top1)}\n")
    print(f"{'sinif':<20s}{'sahne':>7s}{'ort. kutle payi':>18s}")
    print("-"*45)
    for c in range(6):
        if mass[c]: print(f"{CLS[c]:<20s}{len(mass[c]):>7d}{100*np.mean(mass[c]):>17.1f}%")
    ns=np.array(per_scene)
    print(f"\nsahne basina: none ajanlarda %{100*ns[:,0].mean():.1f}  |  non-none ajanlarda %{100*ns[:,1].mean():.1f}")
    print(f"en yuksek M_cas'li ajan NON-NONE mi: %{100*np.mean(top1):.1f}")
    print(f"IHLAL (bir none ajani, en oncelikli non-none'dan fazla kutle aliyor): "
          f"{len(viol)}/{len(top1)}  (%{100*len(viol)/max(len(top1),1):.1f})")

    # ---- figur ----
    fig=plt.figure(figsize=(16,11))
    gs=fig.add_gridspec(3,3,height_ratios=[1,1.35,1.35],hspace=.32,wspace=.22)
    ax=fig.add_subplot(gs[0,0])
    cs=[c for c in range(6) if mass[c]]
    ax.barh([CLS[c] for c in cs],[100*np.mean(mass[c]) for c in cs],
            color=[COL[c] for c in cs],edgecolor='k',lw=.6)
    ax.set_xlabel('ortalama M_cas kutle payi [%]'); ax.invert_yaxis()
    ax.set_title('Sinif basina kutle payi\n(non-none ajani OLAN sahneler)',fontsize=10)
    ax.grid(axis='x',alpha=.3)
    ax2=fig.add_subplot(gs[0,1])
    ax2.hist(ns[:,0]*100,bins=25,color='#9aa0aa',edgecolor='k',lw=.5)
    ax2.axvline(100*ns[:,0].mean(),color='#d1495b',lw=2,ls='--',
                label=f'ort %{100*ns[:,0].mean():.0f}')
    ax2.set_xlabel("'none' ajanlara giden kutle [%]"); ax2.set_ylabel('sahne')
    ax2.legend(fontsize=8); ax2.set_title('Sahne basina none-payi',fontsize=10)
    ax3=fig.add_subplot(gs[0,2]); ax3.axis('off')
    ax3.text(0,.95,"OKUMA",fontsize=11,weight='bold',va='top')
    ax3.text(0,.80,f"sahne: {len(top1)}\n"
                   f"top-1 ajan non-none: %{100*np.mean(top1):.1f}\n"
                   f"ihlal orani: %{100*len(viol)/max(len(top1),1):.1f}\n\n"
                   f"Beklenti: yieldingTo/waitingFor\ncubugu 'none'dan UZUN olmali.\n"
                   f"Degilse M_cas sinifla ilgisiz\ndagiliyor -> L_mass gerekli.",
             fontsize=9,va='top',family='monospace')

    viol.sort(key=lambda r:-(r['p_none']-r['p_nn']))
    fs={os.path.basename(f):f for f in files}
    for k,r in enumerate(viol[:6]):
        axp=fig.add_subplot(gs[1+k//3,k%3])
        d=np.load(fs[r['file']])
        for pl in d['lanes'][...,:2]:
            v=np.abs(pl).sum(-1)>1e-6
            if v.sum()>1: axp.plot(pl[v,0],pl[v,1],color='0.87',lw=.9,zorder=1)
        ef_=d['ego_agent_future'][:,:2]
        axp.plot(ef_[:,0],ef_[:,1],color='k',lw=1.4,zorder=5)
        box(axp,0,0,0,4.62,2.1,fc='k',ec='k',zorder=6)
        nb,nfz=d['neighbor_agents_past'][:10],d['neighbor_agents_future'][:10]
        for j in range(10):
            if np.abs(nb[j]).sum()==0: continue
            s=nb[j,-1]
            if j==r['j_none']: fc,lb='#d1495b',f"none  M={r['p_none']:.2f}"
            elif j==r['j_nn']: fc,lb='#2e7d32',f"{r['cls_nn']}  M={r['p_nn']:.2f}"
            else: fc,lb='0.78',None
            box(axp,s[0],s[1],s[2],max(s[6],1.),max(s[7],1.),fc=fc,ec='k',
                lw=1.1 if lb else .4,zorder=6 if lb else 4,alpha=1 if lb else .65)
            if lb: axp.annotate(lb,(s[0],s[1]),fontsize=7.5,weight='bold',color=fc,
                                xytext=(0,7),textcoords='offset points',ha='center')
            fj=nfz[j,:,:2]; ok=np.abs(fj).sum(-1)>1e-6
            if ok.sum()>1: axp.plot(fj[ok,0],fj[ok,1],color=fc,lw=1.5 if lb else .7,zorder=5 if lb else 3)
        axp.set_aspect('equal'); axp.set_xlim(-30,70); axp.set_ylim(-25,25)
        axp.tick_params(labelsize=6)
        axp.set_title(f"IHLAL: none ajani {r['p_none']:.2f} > {r['cls_nn']} {r['p_nn']:.2f}",fontsize=8.5)
    fig.suptitle("M_cas kutlesi L1 sinifini takip ediyor mu?  (kirmizi = kutleyi kapan 'none' ajani, "
                 "yesil = hak eden ajan)",fontsize=12,y=.985)
    plt.savefig(a.out,dpi=125,bbox_inches='tight')
    print(f"\n[saved] {a.out}")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--pretrained_path",required=True); p.add_argument("--causal_path",required=True)
    p.add_argument("--valid_set",required=True); p.add_argument("--l1_labels",required=True)
    p.add_argument("--limit",type=int,default=0); p.add_argument("--device",default="cuda:0")
    p.add_argument("--out",default="viz_out/mass_vs_class.png")
    main(p.parse_args())
