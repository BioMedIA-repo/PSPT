"""Verify actual optimizer updates, independently of Lightning step counters."""
import json
from pathlib import Path

import torch


def precision_options(args):
    if str(args.precision) in ('16', '16-mixed'):
        from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
        scale = float(getattr(args, 'amp_init_scale', 512.))
        if scale <= 0:
            raise ValueError('AMP initial scale must be positive')
        scaler = torch.amp.GradScaler('cuda', init_scale=scale)
        return {'plugins': [MixedPrecisionPlugin('16-mixed', 'cuda', scaler)]}
    return {'precision': args.precision}


def verify_optimizer_updates(trainer, model, args):
    steps = [float(state['step']) for optimizer in trainer.optimizers
             for state in optimizer.state.values() if 'step' in state]
    if not steps or max(steps) <= 0:
        raise RuntimeError('All optimizer updates skipped; this run is not a completed training trial')
    by_component = {'backbone': 0, 'head': 0}
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for optimizer in trainer.optimizers:
        for parameter, state in optimizer.state.items():
            if 'step' in state and float(state['step']) > 0:
                key = 'backbone' if names[id(parameter)].startswith('backbone.') else 'head'
                by_component[key] += 1
    if not by_component['head'] or (not model.freeze_backbone and not by_component['backbone']):
        raise RuntimeError('Missing required backbone/head optimizer updates')
    result = {'optimizer_max_step': max(steps), 'updated_parameter_tensors': by_component,
              'precision': args.precision, 'amp_initial_scale': getattr(args, 'amp_init_scale', 512.)}
    path = Path(args.output_dir) / args.run_name / args.tag / 'optimizer_update_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Optimizer updates verified: {result}', flush=True)
