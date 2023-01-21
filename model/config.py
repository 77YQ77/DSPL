class Arguments():
    def __init__(self):
        self.batch_size = 32
        self.test_batch_size = 1000
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.epoch_em_train = 100 #250
        self.seed = 1
        self.log_interval = 5
        self.save_model = False
        self.image_size = 32
        self.epoch = 25 # 150 # 25
        self.ADHD_epoch = 20
        # self.lambda_1 = 0.25
        # self.lambda_2 = 0.25
        # self.lambda_3 = 0.5
        self.weight_decay = 5e-4
        self.n_lbl = 4000
        self.split_txt = 'run1'
        self.iterations = 20
        self.out = 'outputs'
        self.resume = ''
        self.epchs = 25
        self.batchsize = 32
        self.warmup=0
        self.epochs=50
        self.device='cuda'
        self.num_classes=1
        self.test_freq=10
        self.temp_nl = 2.0
        self.kappa_n = 0.1
        self.kappa_p = 0.2
        self.tau_n=0.1
        self.tau_p = 0.55
        self.num_classes_pseudo = 2
        self.class_blnc = 3
        self.generate_epoch =4
        self.FL_generate_epcoh=4
args = Arguments()



