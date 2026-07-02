from train import train_pipeline


if __name__ == "__main__":
    args_list = [
        "-opt", r"D:\codes\SimpleSR\configs\diffusion_sr\ddpm_unet_x4.yml",
        "--debug",
    ]
    train_pipeline(args_list)
