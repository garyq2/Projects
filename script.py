import glob
import os

# Used to unpackage lib* so we can insert into Conda environment

directory = '/home/q/Downloads/cuda/files/var/cuda-repo-10-0-local-10.0.130-410.48/'
g = glob.glob(directory+'*.deb')

dpkg ='dpkg -x '

# dpkg -x libcudnn7_7.6.0.64-1+cuda10.0_amd64.deb cudann
# sudo chmod 777 -R cudann

for file in g:
    folder = file[:-4]
    if not os.path.exists(folder):
        os.makedirs(folder)
    command = dpkg + file + ' ' + folder
    os.system(command)

    command = 'sudo chmod 777 -R ' + folder
    os.system(command)

