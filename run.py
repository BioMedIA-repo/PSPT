import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'
PREPROCESS = ROOT / 'preprocessing'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def environment(gpu):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def execute(command, args, label):
    print(label + ': ' + ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=environment(args.gpu), check=True)


def resolve_split(args, config):
    if args.split_csv:
        path = args.split_csv.resolve()
    elif config['task'] == 'BRACS':
        path = ROOT / 'splits/bracs.csv'
    else:
        filename = (f'cptac_label_split{args.fold}.csv' if config['task'] == 'COAD'
                    else f'luad_survival_split43_fold{args.fold}.csv')
        directory = args.split_dir.resolve() if args.split_dir else ROOT/'splits'/config['task'].lower()
        path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_split(path, config):
    import pandas as pd
    frame = pd.read_csv(path)
    required = {'wsi_id', 'label', 'fold'}
    if config['task'] == 'LUAD':
        required |= {'case_id', 'event', 'censorship', 'survival_months', 'time_bin', 'split'}
    if not required <= set(frame):
        raise ValueError(f'Missing CSV columns: {sorted(required-set(frame))}')
    frame['wsi_id'] = frame.wsi_id.astype(str)
    if frame.empty or frame.wsi_id.duplicated().any() or frame.wsi_id.str.contains(r'[/\\]').any():
        raise ValueError('Empty split, duplicate WSI IDs, or path separators in WSI IDs')
    if not frame.fold.isin([-1, 0, 1]).all() or not set([-1, 0, 1]) <= set(frame.fold):
        raise ValueError('fold must contain train=1, val=0, test=-1')
    frame['is_test'] = (frame.fold < 0).astype(int)
    if config['task'] != 'BRACS':
        cases = frame.case_id.astype(str) if 'case_id' in frame else frame.wsi_id.str.split('-').str[0]
        if frame.assign(_case=cases).groupby('_case').fold.nunique().max() > 1:
            raise ValueError('Patient overlap across train/validation/test')
    if config['task'] == 'LUAD':
        if not frame.event.isin([0, 1]).all() or not (frame.event + frame.censorship == 1).all():
            raise ValueError('event and censorship must be complementary binary labels')
        if (frame.survival_months < 0).any() or frame.survival_months.isna().any():
            raise ValueError('Invalid survival time')
        expected = frame.fold.map({1:'train',0:'val',-1:'test'})
        if not (expected == frame['split']).all():
            raise ValueError('LUAD split strings disagree with fold roles')
        if frame.groupby('case_id')[['event','survival_months','fold']].nunique().to_numpy().max() > 1:
            raise ValueError('Inconsistent patient clinical labels')
        import numpy as np
        if not np.isfinite(frame.survival_months.to_numpy(dtype=float)).all():
            raise ValueError('Invalid survival time')
        if not frame.time_bin.isin(range(4)).all() or not (frame.label == frame.time_bin).all():
            raise ValueError('Expected matching time_bin and label values in [0, 3]')
        if frame.groupby('case_id').time_bin.nunique().max() > 1:
            raise ValueError('Inconsistent patient survival bins')
    else:
        classes = 3 if config['task'] == 'BRACS' else 2
        if not frame.label.isin(range(classes)).all():
            raise ValueError('Classification label outside task range')
    return frame


def paths(args, config):
    if args.data_dir is None:
        raise ValueError('--data-dir is required')
    data = args.data_dir.resolve()
    output = (args.output_dir or ROOT/'outputs'/f'{config["task"]}_{config["encoder"]}'.lower()).resolve()
    pcps = (args.pcps_dir or output/'pcps'/f'fold{args.fold}').resolve()
    return {'data': data, 'png': (args.png_dir.resolve() if args.png_dir else data/'scatter_pngs'),
            'coords':data/'coordinates/patches', 'features':data/'features'/{'UNI':'uni_v1','CONCH':'conch_v1','PLIP':'plip'}[config['encoder']]/'pt_files',
            'output':output, 'pcps':pcps, 'cohort':data/'cohorts'/f'{config["task"]}_fold{args.fold}.csv'}


def prepare_cohort(frame, locations):
    import h5py
    rows = frame.copy()
    counts=[]
    for wsi in rows.wsi_id:
        path=locations['coords']/f'{wsi}.h5'
        with h5py.File(path) as handle:
            if 'coords' not in handle or len(handle['coords']) == 0:
                raise ValueError(f'Empty/missing coordinates: {path}')
            counts.append(len(handle['coords']))
    rows['len_img']=counts
    locations['cohort'].parent.mkdir(parents=True,exist_ok=True)
    rows.to_csv(locations['cohort'],index=False)
    return rows


def check_inputs(frame, locations, config, verify_features=False):
    import h5py
    if verify_features:
        import numpy as np
        import torch
    for wsi in frame.wsi_id:
        coord=locations['coords']/f'{wsi}.h5'
        feature=locations['features']/f'{wsi}.pt'
        with h5py.File(coord) as handle:
            n=len(handle['coords'])
            xy=handle['coords'][:] if verify_features else None
        png=locations['png']/wsi
        if not feature.is_file() or not (png/f'{wsi}_patch_0.png').is_file() or not (png/f'{wsi}_patch_{n-1}.png').is_file():
            raise FileNotFoundError(f'Missing feature or boundary patch: {wsi}')
        if verify_features:
            value=torch.load(feature,map_location='cpu',weights_only=True)
            if tuple(value.shape) != (n,config['feature_dim']) or not torch.isfinite(value).all():
                raise ValueError(f'Feature shape/value mismatch: {feature}')
            h5=feature.parent.parent/'h5_files'/f'{wsi}.h5'
            with h5py.File(h5) as handle:
                if not np.array_equal(handle['coords'][:],xy) or not np.array_equal(handle['features'][:],value.numpy()):
                    raise ValueError(f'PT/H5/coordinate mismatch: {wsi}')
    print(f'PASS: {len(frame)} WSI; partitions, coordinates, patches and {config["encoder"]} features checked.',flush=True)


def export_patches(args, config, frame, locations):
    if args.wsi_dir is None:
        raise ValueError('patches requires --wsi-dir')
    import pandas as pd
    data=locations['data']; data.mkdir(parents=True,exist_ok=True)
    coord_root=locations['coords'].parent; coord_root.mkdir(parents=True,exist_ok=True)
    wsi_dir=data/'wsi'; wsi_dir.mkdir(exist_ok=True)
    supplied={}
    for path in args.wsi_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.svs','.ndpi','.tif','.tiff'}:
            if path.stem in supplied:
                raise ValueError(f'Duplicate WSI stem: {path.stem}')
            supplied[path.stem]=path.resolve()
    for wsi in frame.wsi_id:
        if wsi not in supplied:
            raise FileNotFoundError(f'WSI missing in --wsi-dir: {wsi}')
        source=supplied[wsi]; link=wsi_dir/source.name
        if link.exists() and link.resolve()!=source:
            raise ValueError(f'WSI destination conflict: {link}')
        if not link.exists():link.symlink_to(source)
    settings=dict(patch_level=0 if config['task']=='LUAD' else 1, patch_size=256, step_size=256,
                  segmentation={'a_h': 16, 'a_t': 100, 'close': 4, 'contour_fn': 'four_pt', 'exclude_ids': 'none', 'keep_ids': 'none', 'line_thickness': 500, 'max_n_holes': 8, 'mthresh': 7, 'seg_level': -1, 'sthresh': 8, 'use_otsu': False, 'use_padding': True, 'vis_level': -1})
    process=data/'process.csv'
    rows=pd.DataFrame({'slide_id':[supplied[w].name for w in frame.wsi_id],'process':1})
    for key,value in settings['segmentation'].items():rows[key]=value
    rows.to_csv(process,index=False)
    preset=coord_root/'process.csv'; preset.write_bytes(process.read_bytes())
    csv_path=data/'features.csv';pd.DataFrame({'slide_id':frame.wsi_id}).to_csv(csv_path,index=False)
    contract={'preprocessing':settings,'wsi':[{'id':w,'size':supplied[w].stat().st_size} for w in frame.wsi_id]}
    guard=data/'preprocessing_contract.json'
    if guard.exists() and json.loads(guard.read_text())!=contract:
        raise ValueError('Preprocessing input/settings changed; use a new --data-dir')
    write_json(guard,contract)
    command=[sys.executable,'-B',PREPROCESS/'create_patches.py','--source',wsi_dir,'--save_dir',coord_root,
             '--process_list','process.csv','--patch_level',settings['patch_level'],
             '--patch_size',settings['patch_size'],'--step_size',settings['step_size'],'--seg','--patch']
    execute(command,args,'COORDINATES')
    prepare_cohort(frame,locations)
    command=[sys.executable,'-B',PREPROCESS/'export_pngs.py','--svs_dir',wsi_dir,'--h5_dir',locations['coords'],
             '--output_dir',locations['png'],'--patch_size',settings['patch_size'],'--num_workers',args.workers]
    execute(command,args,'PNG')


def extract_features(args, config, frame, locations):
    if args.encoder_weights is None:
        raise ValueError('features requires --encoder-weights')
    import pandas as pd
    data=locations['data']
    coord_root=locations['coords'].parent
    guard=data/'preprocessing_contract.json'
    if not guard.is_file():
        raise FileNotFoundError('Run patches before features: ' + str(guard))
    prepare_cohort(frame,locations)
    encoder_path=args.encoder_weights.resolve()
    weight_file=encoder_path/'pytorch_model.bin' if encoder_path.is_dir() else encoder_path
    if not weight_file.is_file():raise FileNotFoundError(weight_file)
    csv_path=data/'features.csv'
    pd.DataFrame({'slide_id':frame.wsi_id}).to_csv(csv_path,index=False)
    feature_root=locations['features'].parent
    feature_contract={'encoder':config['encoder'],'weights_sha256':sha(weight_file),'input_size':224,
                      'preprocessing_sha256':sha(guard)}
    feature_guard=feature_root/'feature_contract.json'
    if feature_guard.exists() and json.loads(feature_guard.read_text())!=feature_contract:
        raise ValueError('Encoder/input changed; use a new --data-dir')
    write_json(feature_guard,feature_contract)
    command=[sys.executable,'-B',PREPROCESS/'extract_features.py','--scatter_png_dir',locations['png'],
             '--coord_h5_dir',coord_root,'--csv_path',csv_path,'--feat_dir',feature_root,
             '--model_name',{'UNI':'uni_v1','CONCH':'conch_v1','PLIP':'plip'}[config['encoder']],'--target_patch_size','224',
             '--batch_size',args.feature_batch_size,'--num_workers',args.workers,
             '--encoder-weights',encoder_path]
    execute(command,args,'FEATURES')
    check_inputs(frame,locations,config,verify_features=True)


def preprocess(args, config, frame, locations):
    export_patches(args, config, frame, locations)
    extract_features(args, config, frame, locations)


def pcps_command(args,config,locations):
    task=config['task']
    command=[sys.executable,'-B',CODE/('train_pcps_survival.py' if task=='LUAD' else 'train_pcps.py')]
    if task=='LUAD':
        command+=['--split_dir',locations['cohort'].parent,'--split-pattern',locations['cohort'].name,'--n-bins','4']
    else:
        command+=['--split_csv',locations['cohort'],'--num_classes',3 if task=='BRACS' else 2]
    command+=['--feature_dir',locations['features'],'--output_dir',locations['pcps'],'--fold',args.fold]
    return command


def online_config(args,config,locations):
    sys.path.insert(0,str(CODE))
    from options import add_common_arguments
    parser=add_common_arguments(argparse.ArgumentParser())
    settings=vars(parser.parse_args([str(locations['cohort'])]))
    if settings['pcps_context_temperature']==float('inf'):
        settings['pcps_context_temperature']='inf'
    settings['clam_size']=[1024,256,128]
    if config['encoder'] in ['CONCH','PLIP']:
        settings.update(network='conch_v1' if config['encoder']=='CONCH' else 'plip',
                        backbone_image_size=256 if config['encoder']=='CONCH' else 224,
                        clam_size=[512,256,128],
                        data_mean=[0.48145466,0.4578275,0.40821073],
                        data_std=[0.26862954,0.26130258,0.27577711])
    if config['task']=='COAD':
        seed=1 if config['encoder']=='UNI' else 2
        settings.update(dataset_name='coad-msi',weighted_sample=True,seed=seed,pcps_eval_sampling_seed=seed)
    elif config['task']=='LUAD':
        settings.update(dataset_name='luad-survival',num_workers=4,seed=4,pcps_eval_sampling_seed=42,
                        n_bins=4,survival_alpha=0.0,survival_checkpoint_metric='cindex',
                        survival_disable_last_checkpoint=True,survival_eval_level='case',
                        survival_split_index=args.fold,survival_test_both_checkpoints=False)
    if config['task']=='BRACS':
        settings['seed']=args.fold if args.seed is None else args.seed
        settings['pcps_eval_sampling_seed']=settings['seed']
    elif args.seed is not None:
        settings['seed']=args.seed
    if args.eval_sampling_seed is not None:settings['pcps_eval_sampling_seed']=args.eval_sampling_seed
    if args.encoder_weights is None:raise ValueError('train/evaluate require --encoder-weights (pretrained backbone input)')
    weights=args.encoder_weights.resolve()
    if config['encoder']=='PLIP' and weights.is_file():weights=weights.parent
    elif config['encoder']!='PLIP' and weights.is_dir():weights=weights/'pytorch_model.bin'
    if not weights.exists():raise FileNotFoundError(weights)
    settings.update(dataset_csv=str(locations['cohort']),dataset_root='',scatter_png_dir=str(locations['png']),
                    output_dir=str(locations['output']),run_name=f'{config["task"]}_{config["encoder"]}_{args.variant}',
                    tag=f'fold{args.fold}',load_backbone_weight=str(weights),load_weights=None,gpu_id=[0],val_fold=0,
                    disable_scpm=args.variant=='wo_scpm',enable_fprd=args.variant!='wo_fprd')
    settings['pcps_selection_path']=str(locations['pcps']/f'pcps_selection_fold{args.fold}.json')
    if args.variant=='wo_pcps_uniform':
        settings.update(pcps_selection_path=None,pcps_sampling_mode='uniform',pcps_eval_mode='matched')
    else:
        for path in [Path(settings['pcps_selection_path']),locations['pcps']/f'pcps_patch_diagnostics_fold{args.fold}.pt']:
            if not path.is_file():raise FileNotFoundError(path)
    if config['task']=='LUAD':settings['survival_split_index']=args.fold
    return settings


def train(args,config,locations):
    settings=online_config(args,config,locations)
    directory=locations['output']/'configs';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{settings["run_name"]}_fold{args.fold}.json'
    contract={'configuration':settings,'split_sha256':sha(locations['cohort']),
              'source_sha256':{str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py')}}
    contract_path=path.with_suffix('.contract.json')
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
        raise ValueError('Training contract changed; use a new --output-dir')
    if contract_path.exists() and not args.restart:
        raise ValueError('This training output already exists. Use a new output, or --restart to explicitly rerun an incomplete trial.')
    write_json(contract_path,contract);write_json(path,settings)
    execute([sys.executable,'-B',ROOT/'run.py','worker','--worker-config',path],args,'ONLINE TRAIN')


def worker(path,evaluate_checkpoint=None):
    sys.path.insert(0,str(CODE))
    settings=argparse.Namespace(**json.loads(path.read_text()))
    if evaluate_checkpoint:
        import pytorch_lightning as pl
        if settings.dataset_name=='luad-survival':
            from main_survival import get_model,get_network
            from dataset.survival_wsi_dataset import SurvivalPatchWsiDataModule as DataModule
            from pl_model.mil_survival_trainer import MilSurvivalModule as Module
            extra={}
        else:
            from main import get_model,get_network,get_loss_weight
            from dataset import get_class_names
            from dataset.merge_patch_wsi_dataset import PatchWsiDataModule as DataModule
            from pl_model.mil_e2e_trainer import MilE2EModule as Module
            extra={'classes_names':get_class_names(settings.dataset_name)}
        pl.seed_everything(settings.seed,workers=True)
        data=DataModule(settings.dataset_root,settings.dataset_csv,val_fold=0,num_workers=0,num_workers_eval=0,
            scatter_png_dir=settings.scatter_png_dir,pcps_selection_path=settings.pcps_selection_path,
            pcps_selected_count=settings.pcps_selected_count,pcps_random_count=settings.pcps_random_count,
            pcps_sampling_mode=settings.pcps_sampling_mode,pcps_total_count=settings.pcps_total_count,
            pcps_evidence_concentration=settings.pcps_evidence_concentration,pcps_context_temperature=settings.pcps_context_temperature,
            pcps_distribution_concentration=settings.pcps_distribution_concentration,pcps_eval_mode=settings.pcps_eval_mode,
            pcps_eval_sampling_seed=settings.pcps_eval_sampling_seed,**extra)
        backbone,dim=get_network(settings)
        if settings.dataset_name=='luad-survival':model=get_model(settings,backbone,dim)
        else:model=get_model(settings,backbone,dim,len(get_class_names(settings.dataset_name)[0]),get_loss_weight(settings,data))
        trainer=pl.Trainer(accelerator='gpu',devices=[0],logger=False,num_sanity_val_steps=0)
        trainer.test(model,datamodule=data,ckpt_path=str(evaluate_checkpoint))
        return
    from utils import save_parameters
    save_parameters(settings)
    if settings.dataset_name=='luad-survival':from main_survival import main as entry
    else:from main import main as entry
    entry(settings)


def summarize(output):
    import pandas as pd
    files=sorted(output.rglob('all_experiments_results.csv'))+sorted(output.rglob('all_survival_experiments_results.csv'))
    if not files:raise FileNotFoundError('No exported experiment result CSVs')
    frame=pd.concat([pd.read_csv(p).assign(source_file=str(p.relative_to(output))) for p in files],ignore_index=True)
    frame.to_csv(output/'per_run_results.csv',index=False)
    keys=[k for k in ['dataset','network','transfer_type','run_name'] if k in frame]
    numeric=[k for k in frame.select_dtypes(include='number').columns if k not in ['fold','seed']]
    summary=frame.groupby(keys,dropna=False)[numeric].agg(['count','mean','std'])
    summary.to_csv(output/'summary_mean_sample_sd.csv')
    print(summary.to_string())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['patches','features','check','preprocess','pcps','train','evaluate','all','summarize','worker'])
    parser.add_argument('--task',type=str.upper,choices=['BRACS','COAD','LUAD'],default='BRACS')
    parser.add_argument('--encoder',type=str.upper,choices=['UNI','CONCH','PLIP'],default='UNI')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--wsi-dir',type=Path)
    parser.add_argument('--png-dir',type=Path,help='Optional existing PNG root (e.g. historical COAD PNGs)')
    parser.add_argument('--split-dir',type=Path)
    parser.add_argument('--split-csv',type=Path)
    parser.add_argument('--encoder-weights',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--pcps-dir',type=Path,help='Optional existing selector score/export directory')
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--eval-sampling-seed',type=int)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--feature-batch-size',type=int,default=256)
    parser.add_argument('--variant',choices=['full','wo_scpm','wo_fprd','wo_pcps_uniform'],default='full')
    parser.add_argument('--verify-features',action='store_true')
    parser.add_argument('--restart',action='store_true',help='Explicitly rerun an incomplete online output, not an epoch resume')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--worker-config',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.action=='worker':return worker(args.worker_config,args.checkpoint)
    if args.action=='summarize':
        if args.output_dir is None:parser.error('summarize requires --output-dir')
        return summarize(args.output_dir.resolve())
    config=dict(task=args.task,encoder=args.encoder,feature_dim=1024 if args.encoder=='UNI' else 512)
    split=resolve_split(args,config);frame=read_split(split,config);locations=paths(args,config)
    if args.action=='patches':return export_patches(args,config,frame,locations)
    if args.action=='features':return extract_features(args,config,frame,locations)
    if args.action in ['preprocess','all']:preprocess(args,config,frame,locations)
    prepare_cohort(frame,locations)
    check_inputs(frame,locations,config,args.verify_features)
    if args.action in ['pcps','all'] and args.variant!='wo_pcps_uniform':
        command=pcps_command(args,config,locations)
        if locations['pcps'].exists() and any(locations['pcps'].iterdir()) and not args.restart:
            raise ValueError('PCPS output already exists; use --pcps-dir with train/evaluate or choose a new output.')
        execute(command,args,'PCPS TRAIN')
    if args.action in ['train','all']:train(args,config,locations)
    if args.action=='evaluate':
        if args.checkpoint is None:parser.error('evaluate requires --checkpoint')
        settings=online_config(args,config,locations)
        path=locations['output']/'evaluation_config.json';write_json(path,settings)
        execute([sys.executable,'-B',ROOT/'run.py','worker','--worker-config',path,'--checkpoint',args.checkpoint.resolve()],args,'EVALUATE')


if __name__=='__main__':main()
