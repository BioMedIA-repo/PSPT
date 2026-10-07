import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import h5py
import openslide


def export_one(task):
    wsi,h5,output,patch_size=task
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with h5py.File(h5) as handle:
        coords=handle['coords'][:];level=int(handle['coords'].attrs['patch_level'])
        stored_size=int(handle['coords'].attrs.get('patch_size',patch_size))
    if not len(coords) or stored_size!=patch_size:raise ValueError(f'Invalid coordinates/patch size: {h5}')
    slide=openslide.OpenSlide(str(wsi))
    try:
        for index,coord in enumerate(coords):
            destination=output/f'{Path(h5).stem}_patch_{index}.png'
            if destination.exists():continue
            image=slide.read_region(tuple(map(int,coord)),level,(patch_size,patch_size)).convert('RGB')
            image.info.pop('icc_profile',None)
            temporary=destination.with_suffix('.png.partial')
            image.save(temporary,format='PNG');temporary.replace(destination)
    finally:slide.close()
    return f'{Path(h5).stem}: {len(coords)} PNGs'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--svs_dir',type=Path,required=True)
    parser.add_argument('--h5_dir',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--patch_size',type=int,default=256)
    parser.add_argument('--num_workers',type=int,default=8)
    args=parser.parse_args()
    inputs={}
    for path in args.svs_dir.iterdir():
        if path.suffix.lower() in {'.svs','.ndpi','.tif','.tiff'}:
            if path.stem in inputs:raise ValueError(f'Duplicate WSI ID: {path.stem}')
            inputs[path.stem]=path
    tasks=[]
    for path in sorted(args.h5_dir.glob('*.h5')):
        if path.stem not in inputs:raise FileNotFoundError(f'Missing WSI for {path.stem}')
        tasks.append((inputs[path.stem],path,args.output_dir/path.stem,args.patch_size))
    if not tasks:raise ValueError('No H5 coordinate inputs')
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for result in executor.map(export_one,tasks):print(result,flush=True)


if __name__=='__main__':main()
