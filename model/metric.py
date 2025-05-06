            # inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
            # video_recon = rearrange(
            #     video_recon, "b c t h w -> (b t) c h w"
            # ).contiguous()

            # # Calculate PSNR
            # mse = torch.mean(torch.square(inputs - video_recon), dim=(1, 2, 3))
            # psnr = 20 * torch.log10(1 / torch.sqrt(mse))
            # psnr = psnr.mean().detach().cpu().item()

            # # Calculate LPIPS
            # if args.eval_lpips:
            #     lpips_score = (
            #         lpips_model.forward(inputs, video_recon)
            #         .mean()
            #         .detach()
            #         .cpu()
            #         .item()
            #     )
            #     lpips_list.append(lpips_score)