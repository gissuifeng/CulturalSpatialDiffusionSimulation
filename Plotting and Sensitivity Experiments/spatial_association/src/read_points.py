"""Read the supplied dBASE/POINT files using standard-library binary readers."""
from pathlib import Path
import struct
import pandas as pd

def read_points(path):
    path=Path(path)
    raw=path.with_suffix('.dbf').read_bytes()
    n=struct.unpack_from('<I',raw,4)[0]
    header,length=struct.unpack_from('<HH',raw,8)
    fields=[]
    pos=32
    while raw[pos]!=13:
        name=raw[pos:pos+11].split(b'\0')[0].decode('ascii')
        fields.append((name,raw[pos+16]));pos+=32
    records=[]
    for i in range(n):
        row=raw[header+i*length:header+(i+1)*length]
        if row[:1]==b'*': raise ValueError('Deleted DBF record requires explicit mapping')
        offset=1;record={}
        for name,size in fields:
            record[name]=row[offset:offset+size].decode('gbk').strip();offset+=size
        records.append(record)
    shp=path.read_bytes()
    if struct.unpack_from('>I',shp,0)[0]!=9994: raise ValueError('Invalid shapefile')
    xy=[];pos=100
    while pos<len(shp):
        _,words=struct.unpack_from('>II',shp,pos);pos+=8
        shape=struct.unpack_from('<I',shp,pos)[0]
        if shape!=1: raise ValueError(f'Only POINT is supported: {shape}')
        xy.append(struct.unpack_from('<dd',shp,pos+4));pos+=2*words
    if len(records)!=len(xy): raise ValueError('DBF/geometry count mismatch')
    out=pd.DataFrame(records);out[['x','y']]=xy
    return out
