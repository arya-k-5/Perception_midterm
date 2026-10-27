#!/bin/bash
#SBATCH --job-name=orbslam3-build
#SBATCH --account=rob_gy6203-a-2026fa
#SBATCH --partition=n2c48m24          # CPU only (per the Open OnDemand notes)
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                     # 3 GB locally was not enough to compile ORB-SLAM3
#SBATCH --time=02:00:00
#SBATCH --output=orbslam3_build_%j.log

bash "$(dirname "$0")/setup_orbslam3.sh"
