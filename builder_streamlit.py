import json

def process_streamlit_notebook():
    with open("Streamlit_submission_BFGAI_Abdul_Latif.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            
            # Basic replace
            if "generator = ________" in src and "image = ________" in src and "def generate_image" in src and "set_scheduler" not in src:
                src = src.replace("generator = ________", "generator = torch.Generator(pipe.device).manual_seed(seed)")
                src = src.replace("image = ________", "image = pipe(prompt=prompt, negative_prompt=neg_prompt, guidance_scale=cfg, num_inference_steps=steps, generator=generator).images[0]")
                cell['source'] = [line + '\n' for line in src.split('\n')]
                if len(cell['source']) > 0: cell['source'][-1] = cell['source'][-1].rstrip('\n')
                
            # Skilled replace
            if "def flush_memory():" in src and "set_scheduler" in src:
                src = src.replace("________\n    ________", "gc.collect()\n    if torch.cuda.is_available():\n        torch.cuda.empty_cache()")
                src = src.replace("pipe.scheduler = ________", "EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)", 1)
                src = src.replace("pipe.scheduler = ________", "DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)", 1)
                src = src.replace("pipe.scheduler = ________", "DDIMScheduler.from_config(pipe.scheduler.config)", 1)
                
                src = src.replace("pipe = ________", "set_scheduler(pipe, scheduler_name)")
                src = src.replace("generator = ________", "torch.Generator(pipe.device).manual_seed(seed)")
                src = src.replace("result = ________", "pipe(prompt=[prompt]*num_images, negative_prompt=[neg_prompt]*num_images, guidance_scale=cfg, num_inference_steps=steps, generator=generator).images")
                
                cell['source'] = [line + '\n' for line in src.split('\n')]
                if len(cell['source']) > 0: cell['source'][-1] = cell['source'][-1].rstrip('\n')
                
            # Advanced replace
            if "def run_inpainting" in src and "prepare_outpainting" in src:
                src = src.replace("result = ________", "pipe(prompt=prompt, image=image, mask_image=mask, strength=strength).images[0]")
                src = src.replace("w, h = ________", "image.size")
                src = src.replace("new_w = ________", "w + (expand_pixels * 2)")
                src = src.replace("new_h = ________", "h + (expand_pixels * 2)")
                src = src.replace("new_w -= (________)", "new_w % 8")
                src = src.replace("new_h -= (________)", "new_h % 8")
                src = src.replace("mask.paste(________, (________, ________))", "inner_box, (paste_x, paste_y)")
                
                cell['source'] = [line + '\n' for line in src.split('\n')]
                if len(cell['source']) > 0: cell['source'][-1] = cell['source'][-1].rstrip('\n')
                
    with open("Streamlit_submission_BFGAI_Abdul_Latif.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

process_streamlit_notebook()
