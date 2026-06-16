python head_app.py --data_path /mnt/dataset/track-videos/video_demo/video_demo.mp4 \
	--tracker CenterTrack --track_thresh 0.4 \
        --gpus 0 \
        --load_model CenterTrack/exp/tracking/detrac_train/model_15.pth \
        --save_dir results/save_dir/ \
        --save_video  \
        --track_only   
        #--load_routes /home/debugger/code/xieliang/traffic-UAC-01/traffic-UAC/results/s13-center/video_32/routes.bin
        #--load_model models_test/model_last.pth
        #--load_model CenterTrack/exp/tracking/detrac_train/model_epoch_15.pth
#	--load_routes /home/debugger/code/traffic-UAC/results/s3-center/routes.bin
