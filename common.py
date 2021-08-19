from lib import *
from lib.loss import *
from lib.utility.file import *
from lib.net.lookahead import *
from lib.net.radam import *
from lib.net.layer_np import *
from lib.net.rate import *


IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
COMMON_STRING = '@%s:  \n' % os.path.basename(__file__)
# print(COMMON_STRING)
if 1:
    seed = int(time.time())
    seed_py(seed)
    seed_torch(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

    COMMON_STRING += '\tpytorch\n'
    COMMON_STRING += '\t\tseed = %d\n' % seed
    COMMON_STRING += '\t\ttorch.__version__                  = %s\n' % torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda                 = %s\n' % torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version()     = %s\n' % torch.backends.cudnn.version()
    COMMON_STRING += '\t\ttorch.cuda.device_count()          = %d\n' % torch.cuda.device_count()

    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']         = %s\n' % os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']         = None\n'
        NUM_CUDA_DEVICES = 1
    COMMON_STRING += '\t\ttorch.cuda.get_device_properties() = %s\n' % str(torch.cuda.get_device_properties(0))[21:]
    COMMON_STRING += '\t\ttorch.cuda.device_count()          = %d\n' % torch.cuda.device_count()    

COMMON_STRING += '\n'

if __name__ == "__main__":
    print(COMMON_STRING)