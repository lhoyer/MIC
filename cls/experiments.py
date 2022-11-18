import itertools


def generate_experiment_cfgs(id):

    def config_from_vars():
        # The default parameters for the dataset/model configuration are based on
        # https://github.com/val-iisc/SDAT/blob/main/examples/run_office_home.sh and
        # https://github.com/val-iisc/SDAT/blob/main/examples/run_visda.sh
        data_path = {
            'OfficeHome': 'data/office-home',
            'VisDA2017': 'data/visda-2017',
        }[dataset]
        if 'vit' in architecture or 'beit' in architecture or 'mae' in architecture:
            specific_args = {
                'OfficeHome': '--epochs 30 -b 24 --no-pool',
                'VisDA2017': '--epochs 15 --per-class-eval --train-resizing cen.crop --no-pool',
            }[dataset]
            default_lr = 0.002
            default_seed = 0
        elif architecture == 'resnet101':
            specific_args = {
                'VisDA2017': '--epochs 30 --per-class-eval --train-resizing cen.crop --temperature 3.0',
            }[dataset]
            default_lr = 0.002
            default_seed = 2
        else:
            raise NotImplementedError(architecture)
        if lr is None:
            specific_args += f' --lr {default_lr}'
        else:
            specific_args += f' --lr {lr}'
        if seed is None:
            specific_args += f' --seed {default_seed}'
        else:
            specific_args += f' --seed {seed}'
        mask_args = ''
        if 'masking' in uda:
            mask_args = f'--alpha {alpha} --pseudo_label_weight {pseudo_label_weight} ' \
                        f'--mask_block_size {mask_block_size} --mask_ratio {mask_ratio} ' \
                        f'--mask_color_jitter_s {mask_color_jitter_s} --mask_color_jitter_p {mask_color_jitter_p} --mask_blur {mask_blur}'
        else:
            assert alpha == 0 and mask_block_size == 0 and mask_ratio == 0
        common_args = f'-a {architecture} --gpu 0 --rho 0.02'

        data_name_abrev = f'{dataset}_{source}2{target}'\
            .replace('VisDA2017', 'visda')\
            .replace('OfficeHome', 'office')\
            .replace('Synthetic', 'syn')\
            .lower()
        architecture_name = architecture.replace('_base_patch16_224', '')
        name = f'{data_name_abrev}_{uda}'
        if 'masking' in uda:
            name += f'_m{mask_block_size}-{mask_ratio}-a{alpha}'
            if pseudo_label_weight is None:
                pass
            elif pseudo_label_weight == 'prob':
                name += '-plw'
            else:
                name += f'-plw{pseudo_label_weight}'
            if mask_color_jitter_p > 0 and mask_color_jitter_s > 0:
                name += f'-cj{mask_color_jitter_p}-{mask_color_jitter_s}'
            if mask_blur:
                name += f'-b'
        name += f'_{architecture_name}'
        if lr is not None:
            name += f'_lr{lr}'
        if seed is not None:
            name += f'_s{seed}'
        name = name.replace('True', 'T').replace('.', '')
        log_args = f'--log logs/cdan_mcc_sdat_{architecture_name}/{data_name_abrev} ' \
                   f'--log_name {name} ' \
                   f'--log_results'

        cmd = f'python {uda}.py {data_path} -d {dataset} ' \
              f'-s {source} -t {target} {specific_args} {common_args} ' \
              f'{mask_args} {log_args}'

        cfg = dict(
            exp=id,
            name=name,
            subfolder='examples',
            NGPUS=n_gpus,
            NCPUS=n_cpus,
            gpu_model=gpu_model,
            EXEC_CMD=cmd,
        )

        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    n_cpus = 8
    gpu_model = 'NVIDIATITANRTX'
    dataset = 'VisDA2017'
    uda = 'cdan_mcc_sdat'
    architecture = 'vit_base_patch16_224'
    seed = None
    lr = None
    alpha = 0
    pseudo_label_weight = False
    mask_block_size = 0
    mask_ratio = 0
    mask_color_jitter_s = 0
    mask_color_jitter_p = 0
    mask_blur = False

    # -------------------------------------------------------------------------
    # MIC(SDAT) with ViT on VisDA and OfficeHome
    # -------------------------------------------------------------------------
    # yapf: disable
    if id == 1:
        datasets = [
            ('VisDA2017', 'Synthetic', 'Real'),
            ('OfficeHome', 'Ar', 'Cl'),
            ('OfficeHome', 'Ar', 'Pr'),
            ('OfficeHome', 'Ar', 'Rw'),
            ('OfficeHome', 'Cl', 'Ar'),
            ('OfficeHome', 'Cl', 'Pr'),
            ('OfficeHome', 'Cl', 'Rw'),
            ('OfficeHome', 'Pr', 'Ar'),
            ('OfficeHome', 'Pr', 'Cl'),
            ('OfficeHome', 'Pr', 'Rw'),
            ('OfficeHome', 'Rw', 'Ar'),
            ('OfficeHome', 'Rw', 'Cl'),
            ('OfficeHome', 'Rw', 'Pr'),
        ]
        udas = [
            # uda method, alpha, weight, patch, ratio, aug
            ('cdan_mcc_sdat_masking', 0.9, 'prob', 64, 0.7, True),
        ]
        for (dataset, source, target), (uda, alpha, pseudo_label_weight, mask_block_size, mask_ratio, caug) \
                in itertools.product(datasets, udas):
            if caug:
                # augmentation parameters from DAFormer
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0.2, 0.2, True
            else:
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0, 0, False
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # MIC(SDAT) with ResNet-101 on VisDA
    # -------------------------------------------------------------------------
    elif id == 2:
        architecture = 'resnet101'
        datasets = [
            ('VisDA2017', 'Synthetic', 'Real'),
        ]
        udas = [
            # uda method, alpha, q, patch, ratio, aug
            ('cdan_mcc_sdat_masking', 0.9, None, 64, 0.7, True),
        ]
        for (dataset, source, target), (uda, alpha, pseudo_label_weight, mask_block_size, mask_ratio, caug) \
                in itertools.product(datasets, udas):
            if caug:
                # augmentation parameters from DAFormer
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0.2, 0.2, True
            else:
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0, 0, False
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SDAT with MAE on VisDA (as baseline)
    # -------------------------------------------------------------------------
    elif id == 3:
        architecture = 'mae_base_patch16_224',  # MAE+ImageNet pretraining
        datasets = [
            ('VisDA2017', 'Synthetic', 'Real'),
        ]
        udas = [
            # uda method, alpha, q, patch, ratio, aug
            ('cdan_mcc_sdat', 0, False, 0, 0, False),
        ]
        seeds = [1]  # best seed from [0, 7]
        for (dataset, source, target), (uda, alpha, pseudo_label_weight, mask_block_size, mask_ratio, caug), seed \
                in itertools.product(datasets, udas, seeds):
            if caug:
                # augmentation parameters from DAFormer
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0.2, 0.2, True
            else:
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0, 0, False
            cfg = config_from_vars()
            cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
