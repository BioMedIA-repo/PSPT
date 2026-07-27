import os
import h5py
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import argparse

def process_wsi(args_tuple):
    slide_id, svs_path, h5_path, output_dir, patch_size = args_tuple
    
    # Create a dedicated scatter PNG folder per WSI for easier management and selection
    wsi_scatter_dir = os.path.join(output_dir, slide_id)
    os.makedirs(wsi_scatter_dir, exist_ok=True)
    
    try:
        slide = openslide.OpenSlide(svs_path)
        
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            patch_level = int(f['coords'].attrs.get('patch_level', 1)) # inherit the 10x setting
            
        num_patches = len(coords)
        if num_patches == 0:
            return f"FAILED {slide_id}: empty H5 coordinates"
            
        # Export strictly following the coordinate order in H5
        for idx, coord in enumerate(coords):
            # 100% replica of CLAM native read_region and lossless channel conversion
            patch = slide.read_region(tuple(coord), patch_level, (patch_size, patch_size)).convert('RGB')
            
            # Save as lossless PNG to lock pixel color values
            # Naming: WSI_ID_patch_index.png for precise tracking and reordering in phase 2
            save_path = os.path.join(wsi_scatter_dir, f"{slide_id}_patch_{idx}.png")
            patch.save(save_path, format='PNG')
            
        return f"SUCCESS {slide_id}: exported {num_patches} lossless square patches"
        
    except Exception as e:
        return f"FAILED {slide_id}: {str(e)}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1: lossless export of WSI scatter patches")
    parser.add_argument('--svs_dir', type=str, required=True, help='SVS slide directory')
    parser.add_argument('--h5_dir', type=str, required=True, help='CLAM H5 coordinate directory')
    parser.add_argument('--output_dir', type=str, required=True, help='scatter PNG output directory')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    h5_files = [f for f in os.listdir(args.h5_dir) if f.endswith('.h5')]
    tasks = []
    for h5_file in h5_files:
        slide_id = h5_file.replace('.h5', '')
        svs_path = os.path.join(args.svs_dir, f"{slide_id}.svs")
        if not os.path.exists(svs_path):
            svs_path = os.path.join(args.svs_dir, f"{slide_id}.ndpi")
        if os.path.exists(svs_path):
            tasks.append((slide_id, svs_path, os.path.join(args.h5_dir, h5_file), args.output_dir, args.patch_size))

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_wsi, tasks), total=len(tasks)))