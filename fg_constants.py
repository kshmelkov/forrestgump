import platform
host = platform.node()

DIALOGS = 'german_dialog_20150211.json'
BLIND_ANNOTATIONS = 'german_audio_description_20150211.json'

SEGMENTS = [902, 882, 876, 976, 924, 878, 1084, 675]  # length in seconds

TOTAL_SEGMENTS = 8

num_word2vec_features = 300
num_glove_features = 300

SEGLIST = range(TOTAL_SEGMENTS)

TR = 2.0
SCANS = [451, 441, 438, 488, 462, 439, 542, 337]

SUBTITLES_DIR = './subtitles'
CQ_DIR = './cq'
MFCC_DIR = './mfcc'

if host == 'drago':
    AUDIO_DIR = '/storage/workspace/kshmelkov/stimuli'
    DATA_DIR = '/storage/data/openfmri/ds113'
    JOBLIB_DIR = '/storage/workspace/kshmelkov/joblib'
else:
    AUDIO_DIR = '/volatile/accounts/kshmelkov/fg'
    DATA_DIR = '/volatile/accounts/kshmelkov/data'
    JOBLIB_DIR = '/volatile/accounts/kshmelkov/joblib'

FREQ_MIN = 20
FREQ_MAX = 20480

PREP_DATA_DIR = './clean_data'
MAPS_DIR = './maps'

WARP_DIR = './warped_data'

PREDICTED_DIR = './predicted_bold'

MASK_FILE = './masks/template_mask_thick.nii.gz'

CONFOUNDS_DIR = './confounds'

SUBJECTS = [1,2,3,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]

ALPHA = 1000  # L_2 regularization
LAGS = [30, 50, 70]  # lags in 10Hz rate

SESSION_BOLD_OFFSET = 3  # skip first TRs in a session
