"""
Código para caragar el modelo una sola vez, creado por
@danielvallejo237

"""

from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized

ModelToUse=None
Device=None
Device =select_device('cpu')
ModelToUse=attempt_load('YellowSheetsModel.pt',Device)
if __name__=='__main__':
   pass