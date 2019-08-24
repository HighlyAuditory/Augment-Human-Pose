
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PATN':
        assert opt.dataset_mode == 'keypoint'
        from .PATN import TransferModel
        model = TransferModel()

    elif opt.model == 'Augment':
        from .augment_model import AugmentModel
        model = AugmentModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

        
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
