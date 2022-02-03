from shutil import ExecError
import detect
import json
import os
"""
Creado por @danielvallejo237
"""

def GetName(source,padd=(0,0,0,0),save_Img=True,ToJson=True):
    regions=detect.detect(source)
    name=detect.findNameInFile(source,regions=regions,padd=padd,save_Img=save_Img)
    if ToJson:
        informacion={}
        informacion['fuente']=source.split('/')[-1]
        informacion['nombre']=name
        return json.dumps(informacion) 
    else:    
        return name

def ProcessPath(path,padd=(0,0,0,0),save_Img=True):
    terminaciones=['.png','.jpg']
    if not os.path.exists(path):
        raise ExecError("Non existent Path")
    directorios=os.listdir(path)
    directorios=[d for d in directorios if d.endswith(tuple(terminaciones))]
    MegaJson={}
    personas=[]
    for i,d in enumerate(directorios):
        fname=os.path.join(path,d)
        fjs=GetName(source=fname,padd=padd,save_Img=save_Img,ToJson=True)
        personas.append(fjs)
    MegaJson['personas']=personas
    return json.dumps(MegaJson)


if __name__=='__main__':
    pass