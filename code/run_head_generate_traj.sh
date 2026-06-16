python head_app.py --data_path /mnt/dataset/track-videos/video_32/video_32.avi \
	--tracker CenterTrack --track_thresh 0.4 \
        --gpus 1 \
        --load_model CenterTrack/exp/tracking/detrac_train/model_epoch_15.pth \
        --save_dir results/save_traj/ \
        --save_video  \
        --track_only  
#        --load_model CenterTrack/exp/tracking/detrac_train/model_epoch_15.pth
#	--load_routes /home/debugger/code/traffic-UAC/results/s3-center/routes.bin
