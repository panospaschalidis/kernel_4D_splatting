workdir=$1
video_path="${workdir}.mp4"
cd colmap_params
python video2im_seq.py --video_path ../videos/$video_path --out_path $workdir
CUDA_VISIBLE_DEVICES=0 python colmap_runner.py $workdir
cd ../
mkdir -p data/custom/$workdir/
mkdir -p data/custom/$workdir/rgb/
mkdir -p data/custom/$workdir/rgb/1x
cp -fr colmap_params/$workdir/color_full/* data/custom/$workdir/rgb/1x
cp -fr colmap_params/$workdir/colmap_dense data/custom/$workdir/
mv data/custom/$workdir/colmap_dense data/custom/$workdir/colmap
cp -fr colmap_params/$workdir/frames.txt data/custom/$workdir/
python scripts/create_camera_dir.py --video_instance data/custom/$workdir 
python scripts/create_json_files.py --video_instance data/custom/$workdir
cd data/custom/$workdir/
mv colmap/dense/0 colmap/dense/workspace
cd colmap/dense/workspace/
rm -rf !\(fused.ply\)
cd ../../../../../../
python scripts/downsample_point.py data/custom/$workdir/colmap/dense/workspace/fused.ply data/custom/$workdir/points3D_downsample2.ply
