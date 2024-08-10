label_num2str = {0: 'glioma',
                    1: 'meningioma',
                    2: 'notumor',
                    3: 'pituitary',
                    }
label_str2num = {'glioma': 0,
                    'meningioma':1,
                    'notumor':2,
                    'pituitary':3,
                    }

TRAIN_DIR = "/mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/Training"
TEST_DIR = "/mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/Testing"

BRAIN_TUMOR_PRETRAINED = "/mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/new_cnn_brain_cass.pt"
batch_size = 32
device = "cuda:0"
checkpoint_path = "/mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/new_cnn_brain_cass.pt"