import gradio as gr
import os
hf_token = os.environ.get("HF_TOKEN")
import spaces
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch
import time

class Dummy():
    pass

# Ng
default_negative_prompt= "Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"

# Load pipeline
model_id = "briaai/BRIA-2.2"
scheduler = EulerAncestralDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                steps_offset=1
            )
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16,scheduler=scheduler).to("cuda")

print("Optimizing BRIA-2.2 - this could take a while")
t=time.time()
pipe.unet = torch.compile(
    pipe.unet, mode="reduce-overhead", fullgraph=True # 600 secs compilation
)
with torch.no_grad():
    outputs = pipe(
        prompt="an apple",
        num_inference_steps=30,
    )

    # This will avoid future compilations on different shapes
    unet_compiled = torch._dynamo.run(pipe.unet)
    unet_compiled.config=pipe.unet.config
    unet_compiled.add_embedding = Dummy()
    unet_compiled.add_embedding.linear_1 = Dummy()
    unet_compiled.add_embedding.linear_1.in_features = pipe.unet.add_embedding.linear_1.in_features
    pipe.unet = unet_compiled

print(f"Optimizing finished successfully after {time.time()-t} secs")

@spaces.GPU(enable_queue=True)
def infer(prompt):
    print(f"""
    —/n
    {prompt}
    """)
    
    # generator = torch.Generator("cuda").manual_seed(555)
    t=time.time()
    image = pipe(prompt,num_inference_steps=30, negative_prompt=default_negative_prompt).images[0]
    print(f'gen time is {time.time()-t} secs')
    
    # Future
    # Add amound of steps
    # if nsfw:
    #     raise gr.Error("Generated image is NSFW")
    
    return image

css = """
#col-container{
    margin: 0 auto;
    max-width: 580px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <h2 style="text-align: center;">
            BRIA-2.2
        </h2>
        """)
        with gr.Group():
            with gr.Column():
                prompt_in = gr.Textbox(label="Prompt", value="A red colored sports car")
                submit_btn = gr.Button("Generate")
        result = gr.Image(label="BRIA-2.2 Result")

        # gr.Examples(
        #     examples = [ 
        #         "Dragon, digital art, by Greg Rutkowski",
        #         "Armored knight holding sword",
        #         "A flat roof villa near a river with black walls and huge windows",
        #         "A calm and peaceful office",
        #         "Pirate guinea pig"
        #     ],
        #     fn = infer, 
        #     inputs = [
        #         prompt_in
        #     ],
        #     outputs = [
        #         result
        #     ]
        # )

    submit_btn.click(
        fn = infer,
        inputs = [
            prompt_in
        ],
        outputs = [
            result
        ]
    )

demo.queue().launch(show_api=False)