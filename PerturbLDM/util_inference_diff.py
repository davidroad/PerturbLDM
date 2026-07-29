
import torch

def make_batch(example: dict, num_sample: int) -> dict:
    ctrl_expr = example['ctrl_expr'].unsqueeze(0).repeat(num_sample, 1)

    # Repeat scalar conditions
    drug_id       = torch.full((num_sample,), example['drug_id'], dtype=torch.long)
    dose_uM_level = torch.full((num_sample,), example['dose_uM_level'], dtype=torch.float)

    return {
        'ctrl_expr': ctrl_expr,
        'drug_id': drug_id,
        'dose_uM_level': dose_uM_level
    }

# g_step = torch.Generator()
# g_step.manual_seed(seed1)

def inference_process_batch(batch, noise_scheduler, denoising_model, generator = None, latent_dim = 1024):
    denoising_model.eval()
    device = next(denoising_model.parameters()).device
    # print(device)
    all_latents_result = []
    with torch.no_grad():
        noise_scheduler.set_timesteps(1000)
        latents = torch.randn(batch['ctrl_expr'].size(0), latent_dim, device=batch['ctrl_expr'].device, dtype=batch['ctrl_expr'].dtype, generator=generator) * noise_scheduler.init_noise_sigma
        for t in noise_scheduler.timesteps:
            model_output = denoising_model(latents = latents.to(device), **{kk:vv.to(device) for kk,vv in batch.items() if kk not in ["latents", 'expr_mean']}, timesteps = t.unsqueeze(0).repeat(latents.size(0)).to(device))['predict_output']
            latents = noise_scheduler.step(model_output.to(device), t, latents.to(device), generator=generator).prev_sample
        all_latents_result.append(latents)
    
    if len(all_latents_result) == 1:
        latent_result_final = all_latents_result[0]
    else:
        latent_result_final = torch.concat(all_latents_result)
    return(latent_result_final)



def inference_process_batch_logstep(batch, noise_scheduler, denoising_model, generator = None, latent_dim = 1024):
    denoising_model.eval()
    device = next(denoising_model.parameters()).device
    # print(device)
    all_latents_result = []
    latent_steps = []
    with torch.no_grad():
        noise_scheduler.set_timesteps(1000)
        latents = torch.randn(batch['ctrl_expr'].size(0), latent_dim, device=batch['ctrl_expr'].device, dtype=batch['ctrl_expr'].dtype, generator=generator) * noise_scheduler.init_noise_sigma
        for t in noise_scheduler.timesteps:
            if (t+1) % 50 == 0:
                # print(f"timestep {t+1} / 1000")
                latent_steps.append(latents.cpu())

            model_output = denoising_model(latents = latents.to(device), **{kk:vv.to(device) for kk,vv in batch.items() if kk not in ["latents", 'expr_mean']}, timesteps = t.unsqueeze(0).repeat(latents.size(0)).to(device))['predict_output']
            latents = noise_scheduler.step(model_output.cpu(), t, latents, generator=generator).prev_sample
        
        latent_steps.append(latents.cpu())
        all_latents_result.append(latents)
    
    if len(all_latents_result) == 1:
        latent_result_final = all_latents_result[0]
    else:
        latent_result_final = torch.concat(all_latents_result)

    latent_steps = torch.stack(latent_steps)  # shape: (num_steps, batch_size, latent_dim)
    return(latent_result_final, latent_steps)



def inference_process_batch_conditions(batch, noise_scheduler, denoising_model, generator = None, latent_dim = 1024, output_latent_steps = False):
    denoising_model.eval()
    device = next(denoising_model.parameters()).device
    # print(device)
    all_latents_result = []
    for k,v in batch.items():
        batch_len = v.size(0)
        break
    if output_latent_steps:
        latent_steps = []
    with torch.no_grad():
        noise_scheduler.set_timesteps(1000)
        latents = torch.randn(batch_len, latent_dim, dtype=torch.float32, generator=generator) * noise_scheduler.init_noise_sigma
        for t in noise_scheduler.timesteps:
            if output_latent_steps and (t+1) % 50 == 0:
                latent_steps.append(latents.cpu())

            model_output = denoising_model(latents = latents.to(device), **{kk:vv.to(device) for kk,vv in batch.items() if kk not in ["latents"]}, timesteps = t.unsqueeze(0).repeat(latents.size(0)).to(device))['predict_output']
            latents = noise_scheduler.step(model_output.to(device), t, latents.to(device), generator=generator).prev_sample

        if output_latent_steps:
            latent_steps.append(latents.cpu())
        all_latents_result.append(latents)
    
    if len(all_latents_result) == 1:
        latent_result_final = all_latents_result[0]
    else:
        latent_result_final = torch.concat(all_latents_result)

    if output_latent_steps:
        latent_steps = torch.stack(latent_steps)
        return latent_result_final, latent_steps
    else:
        return latent_result_final
    

import torch

def inference_process_strength(
    batch,
    control_input,
    noise_scheduler,
    denoising_model,
    generator=None,
    latent_dim=1024,
    num_inference_steps=1000,
    output_latent_steps=False,
    strength=0.5,
):
    """
    Perform diffusion inference with strength control (like img2img).

    strength:
        0.0 -> almost no change (very small noise added)
        1.0 -> pure noise (full generation)
    """

    device = next(denoising_model.parameters()).device
    denoising_model.eval()

    # Batch size
    batch_len = next(iter(batch.values())).size(0)

    # Set inference timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    # --------------------------------------------------
    # 1️⃣ Compute starting timestep from strength
    # --------------------------------------------------
    init_timestep = int(num_inference_steps * strength)
    init_timestep = min(init_timestep, num_inference_steps)
    
    if init_timestep == 0:
        if output_latent_steps:
            return control_input, control_input
        return control_input
    

    # The timestep we will start denoising from
    t_start = num_inference_steps - init_timestep
    timesteps = noise_scheduler.timesteps[t_start:]

    # --------------------------------------------------
    # 2️⃣ Add noise to control_input
    # --------------------------------------------------
    control_input = control_input.to(device)

    noise = torch.randn(
        control_input.shape,
        dtype=control_input.dtype,
        device=control_input.device,
        generator=generator,
    )

    t_init = timesteps[0]
    print('t_init:', t_init)
    latents = noise_scheduler.add_noise(control_input, noise, t_init)

    # --------------------------------------------------
    # 3️⃣ Denoising loop (only remaining steps)
    # --------------------------------------------------
    latent_steps = []

    with torch.no_grad():
        for i, t in enumerate(timesteps):

            if output_latent_steps and (i % 50 == 0):
                latent_steps.append(latents.detach().cpu())

            model_output = denoising_model(
                latents=latents,
                timesteps=t.unsqueeze(0).repeat(batch_len).to(device),
                **{k: v.to(device) for k, v in batch.items() if k != "latents"},
            )["predict_output"]

            latents = noise_scheduler.step(
                model_output, t, latents, generator=generator
            ).prev_sample

    if output_latent_steps:
        latent_steps.append(latents.detach().cpu())
        return latents, torch.stack(latent_steps)

    return latents