import json

def create_notebook():
    with open("Pipeline_submission_BFGAI_Abdul_Latif.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_snippets = [
        # Dependancies
        "!pip install --upgrade diffusers transformers accelerate scipy safetensors torch torchvision",
        
        # Load Base Pipeline Model
        "import torch\nfrom diffusers import StableDiffusionPipeline\nfrom IPython.display import display\n\nmodel_id = 'runwayml/stable-diffusion-v1-5'\npipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)\npipe = pipe.to('cuda')\npipe.enable_attention_slicing()",
        
        # Generate Image (simple)
        "def generate_simple_image(prompt, negative_prompt, seed):\n    generator = torch.Generator('cuda').manual_seed(seed)\n    image = pipe(\n        prompt=prompt,\n        negative_prompt=negative_prompt,\n        generator=generator,\n    ).images[0]\n    return image\n\nprompt = 'a broken satellite floating in space, highly detailed digital art, trending on artstation'\nnegative_prompt = 'photorealistic, realistic, photograph, 3d render, messy, blurry, low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration'\nseed = 222\n\nimage_simple = generate_simple_image(prompt, negative_prompt, seed)\ndisplay(image_simple)",
        
        # Generate Image with Hyperparameter (advanced)
        "def generate_advanced_image(prompt, negative_prompt, seed, guidance_scale, num_inference_steps):\n    generator = torch.Generator('cuda').manual_seed(seed)\n    image = pipe(\n        prompt=prompt,\n        negative_prompt=negative_prompt,\n        guidance_scale=guidance_scale,\n        num_inference_steps=num_inference_steps,\n        generator=generator,\n    ).images[0]\n    return image\n\nimage_advanced = generate_advanced_image(\n    prompt=prompt, \n    negative_prompt=negative_prompt, \n    seed=seed, \n    guidance_scale=7.5, \n    num_inference_steps=50\n)\ndisplay(image_advanced)",
        
        # Guidance Scale Comparison
        "# Low Guidance Scale\nimg_low_gs = generate_advanced_image(prompt, negative_prompt, seed, guidance_scale=2.0, num_inference_steps=30)\nprint('Low Guidance Scale (2.0):')\ndisplay(img_low_gs)\n\n# High Guidance Scale\nimg_high_gs = generate_advanced_image(prompt, negative_prompt, seed, guidance_scale=15.0, num_inference_steps=30)\nprint('High Guidance Scale (15.0):')\ndisplay(img_high_gs)",
        
        # Inference Steps Comparison
        "# Low Inference Steps (10)\nimg_low_steps = generate_advanced_image(prompt, negative_prompt, seed, guidance_scale=7.5, num_inference_steps=10)\nprint('Low Inference Steps (10):')\ndisplay(img_low_steps)\n\n# High Inference Steps (50)\nimg_high_steps = generate_advanced_image(prompt, negative_prompt, seed, guidance_scale=7.5, num_inference_steps=50)\nprint('High Inference Steps (50):')\ndisplay(img_high_steps)",
        
        # Batch Inference
        "import matplotlib.pyplot as plt\n\ndef generate_batch_images(prompt, negative_prompt, seed, guidance_scale, num_inference_steps, num_images=4):\n    generator = torch.Generator('cuda').manual_seed(seed)\n    images = pipe(\n        prompt=[prompt] * num_images,\n        negative_prompt=[negative_prompt] * num_images,\n        guidance_scale=guidance_scale,\n        num_inference_steps=num_inference_steps,\n        generator=generator,\n    ).images\n    \n    fig, axes = plt.subplots(2, 2, figsize=(10, 10))\n    for ax, img in zip(axes.flatten(), images):\n        ax.imshow(img)\n        ax.axis('off')\n    plt.tight_layout()\n    plt.show()\n\ngenerate_batch_images(prompt, negative_prompt, seed, 7.5, 30)",
        
        # Load Scheduler
        "from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler\n\ndef load_scheduler(pipe, scheduler_name):\n    if scheduler_name == 'Euler A':\n        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)\n    elif scheduler_name == 'DPM++':\n        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)\n    elif scheduler_name == 'DDIM':\n        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)\n    return pipe",
        
        # Scheduler Comparison
        "for sched in ['Euler A', 'DPM++', 'DDIM']:\n    print(f'Generating with {sched}')\n    pipe = load_scheduler(pipe, sched)\n    img = generate_advanced_image(prompt, negative_prompt, seed, 7.5, 30)\n    display(img)",
        
        # Load Model Inpainting
        "from diffusers import StableDiffusionInpaintPipeline\n\ninpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(\n    'runwayml/stable-diffusion-inpainting', \n    torch_dtype=torch.float16\n)\ninpaint_pipe = inpaint_pipe.to('cuda')\ninpaint_pipe.enable_attention_slicing()\n\n# Prepare base image from previous task\nbase_image = image_advanced.copy().resize((512, 512))",
        
        # Manual Masking
        "from PIL import Image, ImageDraw\n\nmask_image = Image.new('L', base_image.size, 0)\ndraw = ImageDraw.Draw(mask_image)\n# Draw a mask block in the center\ndraw.rectangle((150, 150, 362, 362), fill=255)\ndisplay(mask_image)",
        
        # Generate (Inpainting)
        "def inpaint_engine(image, mask, prompt, seed=9):\n    generator = torch.Generator('cuda').manual_seed(seed)\n    result = inpaint_pipe(\n        prompt=prompt,\n        image=image,\n        mask_image=mask,\n        generator=generator\n    ).images[0]\n    return result\n\ninpaint_prompt = 'a detailed broken satellite part with sparking wires'\ninpainted_image = inpaint_engine(base_image, mask_image, inpaint_prompt)\ndisplay(inpainted_image)",
        
        # Load Model Segmentation (Auto masking)
        "from transformers import pipeline\nimport numpy as np\n\nsegmenter = pipeline('image-segmentation', model='facebook/detr-resnet-50-panoptic')",
        
        # Masking with Segmentation
        "def create_auto_mask(image):\n    # Using a simple threshold for demonstration since panoptic might split too much\n    # Let's just create a circular mask as a fallback if segmentation doesn't find object\n    results = segmenter(image)\n    if len(results) > 0:\n        # Get the mask of the most prominent object\n        mask = results[0]['mask']\n        return mask\n    else:\n        m = Image.new('L', image.size, 0)\n        ImageDraw.Draw(m).ellipse((100,100,400,400), fill=255)\n        return m\n\nauto_mask = create_auto_mask(base_image)\ndisplay(auto_mask)",
        
        # Generate (Auto Mask Inpainting)
        "auto_inpainted_image = inpaint_engine(base_image, auto_mask, inpaint_prompt)\ndisplay(auto_inpainted_image)",
        
        # Prepare Canvas (Outpainting)
        "def prepare_outpainting(image, direction='right', expand_pixels=256):\n    width, height = image.size\n    if direction == 'right':\n        new_size = (width + expand_pixels, height)\n        box, mask_box = (0, 0), (width, 0, width + expand_pixels, height)\n    elif direction == 'left':\n        new_size = (width + expand_pixels, height)\n        box, mask_box = (expand_pixels, 0), (0, 0, expand_pixels, height)\n    elif direction == 'bottom':\n        new_size = (width, height + expand_pixels)\n        box, mask_box = (0, 0), (0, height, width, height + expand_pixels)\n    elif direction == 'top':\n        new_size = (width, height + expand_pixels)\n        box, mask_box = (0, expand_pixels), (0, 0, width, expand_pixels)\n        \n    new_image = Image.new('RGB', new_size, (255, 255, 255))\n    new_image.paste(image, box)\n    \n    mask = Image.new('L', new_size, 0)\n    draw = ImageDraw.Draw(mask)\n    draw.rectangle(mask_box, fill=255)\n    return new_image, mask\n\noutpaint_base, outpaint_mask = prepare_outpainting(inpainted_image, 'right', 256)\ndisplay(outpaint_base)",
        
        # Generate (Outpainting)
        "outpaint_prompt = 'space background, stars, dark universe, nebula'\noutpainted_result = inpaint_engine(outpaint_base, outpaint_mask, outpaint_prompt)\ndisplay(outpainted_result)",
        
        # Prepare Canvas for Zoom Out
        "def prepare_zoom_out(image, expand_pixels=128):\n    width, height = image.size\n    new_size = (width + 2*expand_pixels, height + 2*expand_pixels)\n    new_image = Image.new('RGB', new_size, (255, 255, 255))\n    new_image.paste(image, (expand_pixels, expand_pixels))\n    \n    mask = Image.new('L', new_size, 255)\n    draw = ImageDraw.Draw(mask)\n    draw.rectangle((expand_pixels, expand_pixels, expand_pixels+width, expand_pixels+height), fill=0)\n    return new_image, mask\n\nzoom_base, zoom_mask = prepare_zoom_out(inpainted_image, 128)\ndisplay(zoom_base)",
        
        # Generate (Zoom Out)
        "zoom_result = inpaint_engine(zoom_base, zoom_mask, outpaint_prompt)\ndisplay(zoom_result)",
        
        # Base + Refiner Image Generation
        "from diffusers import StableDiffusionImg2ImgPipeline\n\n# Stage 1: Base (Txt2Img)\ngenerator = torch.Generator('cuda').manual_seed(9)\n# Generate initial latent with pipe\nbase_output = pipe(\n    prompt=prompt,\n    generator=generator,\n    output_type='latent',\n)\nlatents = base_output.images\n\n# Load Refiner (Img2Img pipeline with same model for SD1.5)\nrefiner_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(\n    'runwayml/stable-diffusion-v1-5', \n    torch_dtype=torch.float16\n)\nrefiner_pipe = refiner_pipe.to('cuda')\nrefiner_pipe.enable_attention_slicing()\n\n# Stage 2: Refiner (using strength/denoising)\nrefiner_result = refiner_pipe(\n    prompt=prompt,\n    image=latents,\n    strength=0.8, # Equivalent to denoising_start\n    generator=generator,\n).images[0]\ndisplay(refiner_result)"
    ]
    
    code_idx = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if code_idx < len(code_snippets):
                cell['source'] = [line + '\n' for line in code_snippets[code_idx].split('\n')]
                if len(cell['source']) > 0:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                code_idx += 1
                
    with open("Pipeline_submission_BFGAI_Abdul_Latif.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

create_notebook()
