import os
from diffusers import *
import torch
from accelerate import PartialState
from PIL import Image
from tqdm import tqdm 
from random import randint, randrange

torch.cuda.empty_cache()

class_list = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church",
          "circular_farmland", "cloud", "commercial_area", "dense_residential", "desert", "forest", "freeway",
          "golf_course", "ground_track_field", "harbor", "industrial_area", "intersection", "island", "lake",
          "meadow", "medium_residential", "mobile_home_park", "mountain", "overpass", "palace", "parking_lot",
          "railway", "railway_station", "rectangular_farmland", "river", "roundabout", "runway", "sea_ice",
          "ship", "snowberg", "sparse_residential", "stadium", "storage_tank", "tennis_court", "terrace",
          "thermal_power_station", "wetland"]
output_dir = '/nfs/bigiris.cs.stonybrook.edu/rraina/resisc45_sdxl/'
os.makedirs(output_dir, exist_ok=True)
for k in class_list:
    os.makedirs(os.path.join(output_dir, k), exist_ok=True)

distributed_state = PartialState()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo").to(distributed_state.device)
pipe.set_progress_bar_config(disable=True)
pipe.enable_xformers_memory_efficient_attention()

with distributed_state.split_between_processes(list(range(0, len(class_list), 1))) as indexes:
    for i in tqdm(indexes):
        cls = class_list[i]

        prompt = '{}, aerial view, satellite, high resolution, 4K, photorealistic'.format(cls.replace("_", " "))

        if cls == 'runway':
            prompt = 'airport runway, satellite, high resolution, 4K, photorealistic'

        else if cls == 'overpass':
            prompt = 'overpass bridge(s), satellite, high resolution, 4K, photorealistic'
        
        else if cls == 'ground_track_field':
            prompt = 'ground track and field, satellite, high resolution, 4K, photorealistic'
        
        if cls == 'commercial_area':
            prompt = 'metropolitan, commerical area, satellite, high resolution, 4K, photorealistic'
            
        if cls == 'chaparral':
            prompt = 'brushland, satellite, high resolution, 4K, photorealistic'
                  
        formatted_numbers = [f"{num:03}" for num in range(1, 701)]
        for j in tqdm(formatted_numbers):
            generator = torch.Generator(device="cuda").manual_seed(randint(100000, 9999999))
            output_path = os.path.join(output_dir, cls, '{}_{}.jpg'.format(cls, j))
            out = pipe(prompt=prompt, 
                       negative_prompt='low resolutiom, blurry, unrealistic', 
                       safety_checker=None, 
                       num_inference_steps=2, 
                       guidance_scale=0.0,
                       generator=generator).images[0].resize((256,256))
            out.save(output_path) 
