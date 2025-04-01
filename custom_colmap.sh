workdir=$1
mkdir data/custom/$workdir/
mkdir data/custom/$workdir/rgb/
mkdir data/custom/$workdir/rgb/1x
cp -fr ~/workspace/internship_repos/CVD/output/$workdir/color_full/* data/custom/$workdir/rgb/1x
cp -fr ~/workspace/internship_repos/CVD/output/$workdir/colmap_dense data/custom/$workdir/
mv data/custom/$workdir/colmap_dense data/custom/$workdir/colmap
cp -fr ~/workspace/internship_repos/CVD/output/$workdir/frames.txt data/custom/$workdir/
cp -fr data/custom/cretan/*.py data/custom/$workdir
cd data/custom/$workdir/
python create_camera_dir.py 
python create_json_files.py 
mv colmap/dense/0 colmap/dense/workspace
cd colmap/dense/workspace/
rm -rf !\(fused.ply\)
cd ../../../
cp -fr ../cretan/scene.json .
cd ../../../
python scripts/downsample_point.py data/custom/$workdir/colmap/dense/workspace/fused.ply data/custom/$workdir/points3D_downsample2.ply
