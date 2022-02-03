"""
 Modificado por @danielvallejo237
"""

import argparse
import time
from pathlib import Path
from tkinter import NE

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image, ImageFilter
import pytesseract
import unidecode
import re

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import loader

@torch.no_grad()
def get_coords(size,yoloformat):
    dh,dw,_=size #Including a chanel for RGB images
    x,y,w,h=yoloformat
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)
    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    return(l,t,r,b)

def ProcessLine(line,size=(640,640,3)):
    clase=line[0]
    coordenadas=line[1:-1]
    c=get_coords(size,coordenadas)
    clase=int(clase.item())
    return (clase,c,size)

def detect(source=None,imgsz=640,important=[3],augment=True) -> list:
    ROIS=[]
    save_conf=True
    device = loader.Device
    model =  loader.ModelToUse
    imgsz = check_img_size(imgsz, s=model.stride.max())
    dataset = LoadImages(source, img_size=imgsz)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, 0.5, 0.45, classes=important, agnostic=True)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p) 
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                areas={}
                for im in det[:,2:]:
                    if int(im[-1].item()) not in areas:
                        areas[int(im[-1].item())]=im[0].item()*im[1].item()
                    else:
                        areas[int(im[-1].item())]+=im[0].item()*im[1].item()
                for area in areas:
                    line=(area,areas[area])
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                    lproc=ProcessLine(line)
                    ROIS.append(lproc)
    return ROIS


def NewScaler(coords,width, height,size=(640,640))-> tuple:
    x1,y1,x2,y2=coords
    ow,oh=size 
    x1=int((x1/ow)*width)
    y1=int((y1/oh)*height)
    x2=int((x2/ow)*width)
    y2=int((y2/oh)*height)
    return x1,y1,x2,y2
    

               
def findNameInFile(source,regions,size=640,save_Img=True,padd=(0,0,0,0)):
    Name=""
    if len(regions)>0:
        #print("Nombre encontrado")
        img=Image.open(source)
        wi,hi=img.size
        #img = img.resize((size,size))
        nombre=regions[0]
        toCrop=tuple(map(lambda x, y: x + y, nombre[1], padd))
        toCrop=NewScaler(toCrop,wi,hi)
        im1 =img.crop(toCrop)
        #im1 =im1.convert('1')
        #im1=im1.filter(ImageFilter.MinFilter)
        if save_Img:
            nombre_imagen=source.split('/')[-1].split('.')[0]+'_NAME.jpg'
            im1.save(nombre_imagen)
        
        config=[("--oem 1 --psm 7"),("--oem 1 --psm 13"),("--oem 1 --psm 3"),("--oem 1 --psm 6")]
        PossibleNames=[]
        for c in config:
            Name=pytesseract.image_to_string(im1,config=c,lang='spa')
            Name=unidecode.unidecode(Name)
            Name=' '.join(Name.split())
            try:
                Name=re.findall(r"\b[a-zA-Z][\w ]*",Name)[-1].upper()
                try:
                    Name=re.findall(r'RE[\s\w]+',Name)[-1][2:]
                    Name=' '.join(Name.split())
                except:
                    pass
            except:
                Name=""
            PossibleNames.append(Name)
    return PossibleNames


if __name__ == '__main__':
    pass