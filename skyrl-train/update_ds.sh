pushd ../../DeepSpeed
DS_BUILD_STRING=" " python -m build  --wheel --no-isolation
popd
cp ../../DeepSpeed/dist/deepspeed-0.17.6-cp312-cp312-linux_x86_64.whl .
uv install deepspeed-0.17.6-cp312-cp312-linux_x86_64.whl

