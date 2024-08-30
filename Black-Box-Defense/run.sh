# recontruction training
python main.py --mode train --dataset Brain_Tumor --encoder_arch Custom_Encoder --decoder_arch Custom_Decoder --out_dir Brain_Recontruction --mode train
# inference
python main.py --mode inference --encoder_arch Encoder_768 --pretrained_encoder /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/Brain_Recontruction_768/best_encoder.pth --decoder_arch Decoder_768 --pretrained_decoder /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/Brain_Recontruction_768/best_decoder.pth --out_dir Brain_Recontruction_768 --img_path /mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/Training/notumor/Tr-no_0010.jpg
# inference
python main.py --mode inference --pretrained_encoder /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/Brain_Recontruction_1000/best_encoder.pth --decoder_arch Decoder_1000 --pretrained_decoder /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/Brain_Recontruction_1000/best_decoder.pth --out_dir Brain_Recontruction_1000 --img_path //mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/Training/glioma/Tr-gl_0010.jpg


# test defense
python test.py --mode infer_DS --pretrained_encoder /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/Brain_Recontruction_1000/best_encoder.pth --decoder_arch Decoder_1000 --pretrained_decoder /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/Brain_Recontruction_1000/best_decoder.pth --out_dir /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/DS_FO_1000 --img_path /mlcv2/WorkingSpace/Personal/baotg/brain/Brain-Tumor-Classfication/Training/glioma/Tr-gl_0010.jpg --pretrained_denoiser /mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/DS_FO_1000/best_denoiser.pth.tar



python test.py --mode DS --split FGSM --dataset SIPADMEK_Noise --classifier vit_sipadmek --out_dir D:\Medical-Robustness-with-Classification-model\Medical-Blackbox-defense\Black-Box-Defense\experiment\Brain_Recon_MSE_0.25 --pretrained_denoiser D:\Medical-Robustness-with-Classification-model\Medical-Blackbox-defense\Black-Box-Defense\experiment\SIPADMEK_CE_0.25\best_denoiser.pth.tar --img_path D:\Medical-Robustness-with-Classification-model\Medical-Blackbox-defense\Black-Box-Defense\Dataset\Brain_Tumor\AT_FGSM\1\Te-me_0018.jpg
