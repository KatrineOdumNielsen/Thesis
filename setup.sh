#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=my_job.out
#SBATCH --error=my_job.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1

# Optionally load required modules (if your HPC uses a module system)
# module load python/3.13
cd Thesis
# Create a virtual environment if not already present
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip (optional)
#pip install --upgrade pip



# Install dependencies from requirements.txt
pip install -r requirements.txt

# Now run your Python script
# python py_code/Date_file.py