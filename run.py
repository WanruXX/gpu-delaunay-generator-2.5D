import os
import subprocess
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path='conf/', config_name='config.yaml')
def my_app(cfg: DictConfig):
    PointTypeHeaderPath = os.path.join(os.getcwd(), "app/inc/PointType.h")
    with open(PointTypeHeaderPath, "r") as f:
        Lines = f.readlines()
    f.close()
    if(cfg.DisablePCL):
        Lines[2] = "#define DISABLE_PCL_INPUT\n"
    else:
        Lines[2] = "\n"
    with open(PointTypeHeaderPath, "w") as f:
        f.writelines(Lines)
    f.close()

    BuildDir = os.path.join(os.getcwd(), "app/build")
    if(os.path.isdir(BuildDir)):
        subprocess.check_call(["rm", "-rf", BuildDir])
    subprocess.check_call(["mkdir", BuildDir])
    os.chdir(BuildDir)
    subprocess.check_call(["cmake", ".."])
    subprocess.check_call(["make"])
    subprocess.check_call(["make"])


if __name__ == '__main__':
    my_app()
