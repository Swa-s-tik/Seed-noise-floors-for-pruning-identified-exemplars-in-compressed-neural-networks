# ponytail: minimal TF tensor-bundle .index (leveldb SSTable) reader, no compression support
import struct,sys
def vint(b,i):
    r=s=0
    while True:
        x=b[i];i+=1;r|=(x&0x7f)<<s;s+=7
        if x<0x80:return r,i
def unsnappy(c):
    n,i=vint(c,0);o=bytearray()
    while i<len(c):
        t=c[i];i+=1;k=t&3
        if k==0:
            l=t>>2
            if l>=60: nb=l-59;l=int.from_bytes(c[i:i+nb],'little');i+=nb
            l+=1;o+=c[i:i+l];i+=l;continue
        if k==1: l=((t>>2)&7)+4;off=((t>>5)<<8)|c[i];i+=1
        elif k==2: l=(t>>2)+1;off=int.from_bytes(c[i:i+2],'little');i+=2
        else: l=(t>>2)+1;off=int.from_bytes(c[i:i+4],'little');i+=4
        for _ in range(l): o.append(o[-off])
    assert len(o)==n;return bytes(o)
def block(b,off,size):
    d=b[off:off+size]; ct=b[off+size]
    if ct==1: d=unsnappy(d)
    else: assert ct==0,ct
    nre=struct.unpack('<I',d[-4:])[0]; end=len(d)-4-4*nre; i=0; key=b''; out=[]
    while i<end:
        sh,i=vint(d,i);ns,i=vint(d,i);vl,i=vint(d,i)
        key=key[:sh]+d[i:i+ns];i+=ns;out.append((key,d[i:i+vl]));i+=vl
    return out
def pb(b):
    i=0;f={}
    while i<len(b):
        t,i=vint(b,i);fn,wt=t>>3,t&7
        if wt==0: v,i=vint(b,i)
        elif wt==2: l,i=vint(b,i);v=b[i:i+l];i+=l
        elif wt==5: v=b[i:i+4];i+=4
        elif wt==1: v=b[i:i+8];i+=8
        f.setdefault(fn,[]).append(v)
    return f
def read(path):
    b=open(path,'rb').read(); ft=b[-48:]; assert struct.unpack('<Q',ft[-8:])[0]==0xdb4775248b80fb57
    j=0;_,j=vint(ft,j);_,j=vint(ft,j);io,j=vint(ft,j);isz,j=vint(ft,j)
    ents=[]
    for _,h in block(b,io,isz):
        o,k=vint(h,0);s,_=vint(h,k);ents+=block(b,o,s)
    res=[]
    for k,v in ents:
        if k==b'':continue
        f=pb(v);dt=f.get(1,[0])[0];shape=[]
        if 2 in f:
            for dim in pb(f[2][0]).get(2,[]): shape.append(pb(dim).get(1,[0])[0])
        res.append((k.decode(),dt,shape,f.get(5,[0])[0],f.get(4,[0])[0]))
    return res
if __name__=='__main__':
    r=read(sys.argv[1]);import collections
    print(len(r),'tensors; bytes',sum(x[3] for x in r)); print('dtypes',collections.Counter(x[1] for x in r))
    for x in r[:int(sys.argv[2]) if len(sys.argv)>2 else 0]: print(x)
