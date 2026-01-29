
# The configurations is prepare for C2PRI-Net


import argparse

class Config:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_subjects = self.get_num_subjects(dataset_name)
        self.samples_per_subject = self.get_samples_per_subject(dataset_name)
        self.batch_size = 128
        self.eeg_stat_channel_feat_dim = 7
        self.eeg_peri_channel_feat_dim = 1
        self.peri_feat_dim = self.get_peri_feat_dim(dataset_name)
        self.EEG_channel = self.get_EEG_channel(dataset_name)
        self.positional_encoding = True
        self.backbone_hidden = 256
        self.num_layer = 4
        # self.fusion_mothod = 'HF_ICMA'
        self.fusion_mothod = 'HKE_model'
        # self.fusion_mothod = 'concat'
        self.fusion_hidden = 512  
        # self.fusion_hidden = 256  
        self.label_type = 'valence'
        self.hidden = 256
        # self.lr = 1e-3  
        self.lr = 5e-5
        self.weight_decay = 1e-3
        self.device = 'cuda:0'
        self.split = True
        self.backbone_switch = [1,1,1]
        self.trial_cnt = self.get_trial_cnt(dataset_name)
        self.fusion = 'concat'
       

    def get_num_subjects(self, dataset_name):
        
        return {
            'DEAP': 32,
          # 'HCI': 24,
           # 'SEED-IV': 15,
           # 'SEED-V': 16
        }.get(dataset_name, 0)

    def get_samples_per_subject(self, dataset_name):
        
        return {
            'DEAP': 600,
            #'HCI': 528,
            #'SEED-IV': 2505,
            #'SEED-V': 1823
        }.get(dataset_name, 0)

    def get_peri_feat_dim(self, dataset_name):
        
        return {
            'DEAP': 55,
            # 'DEAP': 5, # HST, GSR
            # 'DEAP': 8, # BVP
            # 'DEAP': 20, # RES
            # 'DEAP':17, # EMOG
            #'HCI': 49,
            # 'HCI': 18, # ECG
            # 'HCI': 5, # GSR, TEMP
            # 'HCI': 21, # RESP
            #'SEED-IV': 22,
            #'SEED-V': 24
        }.get(dataset_name, 0)

    def get_EEG_channel(self, dataset_name):
        
        return {
            'DEAP': 32,
            #'HCI': 32,
           # 'SEED-IV': 62,
           # 'SEED-V': 62
        }.get(dataset_name, 0)
    
    def get_trial_cnt(self,dataset_name):
        
        return {
            'DEAP': 40,
           # 'HCI': 20,
            #'SEED-IV': 72,
           # 'SEED-V': 45
        }.get(dataset_name, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LOSO_training_set')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (DEAP)')
    args = parser.parse_args()
    config = Config(args.dataset)




