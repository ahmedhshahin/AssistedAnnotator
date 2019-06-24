hyperparams = {
    'gpu_id' : 0,
    'data_parallel' : True,
    'gpu_ids' : [0,1,2,3],
    'nEpochs': 100,
    'resume_epoch' : 0,
    'train_batch' : 16,
    'relax_dynamic_or_static' : 'static',
    'relax_crop' : 20,
    'n_channels': 4,
    'lr' : 1e-7,
    'wd' : 5e-4,
    'optimizer' : 'sgd',
    'lambda': 0.005,
    'dataset_index': 8,
    'pretrained' : True,
    'pretrained_path' : '/raid/sha/ahmed/multitask/run_2/models/dextr_pascalbest_epoch.pth' 
}