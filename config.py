#  training specific variables

trainloader = None
testloader  = None
num_classes = None
args = None
PERFORMANCE_DETAILS = 1
INPUT_CHANNEL = 1
best_acc = 0.0 
report = {
        'Progress': {
            'test_acc': [],
            'train_acc': [],
            'test_loss': [],
            'train_loss': [],
        },
        'Final': {
            'target': [],
            'output_pre': [],
            'pred_prob': [],
        }
    }


# transformation specific variables 
AUGMENTED_TRANS_SET = ['rot', 'rot_ext', 'ref', 'scale', 'scale_ref']


# rest the configuration
def reset_report():
    report['Progress']['test_acc'] = []
    report['Progress']['train_acc'] = []
    report['Progress']['test_loss'] = []
    report['Progress']['train_loss'] = [] 

    report['Final']['target'] = []
    report['Final']['output_pre'] = [] 
    report['Final']['pred_prob'] = []