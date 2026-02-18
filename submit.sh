#!/bin/bash -l

if [ -z "$1" ]; then
    echo "provide the python script to run as an argument"
    exit 1
fi

PYTHON_SCRIPT=$1
SCRIPT_PATH="/home/RUS_CIP/st196114/dl-lab-25w-team03/diabetic_retinopathy/$PYTHON_SCRIPT"

sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name=dr_job
#SBATCH --output=dr_job-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

cd /home/RUS_CIP/st196114/dl-lab-25w-team03/diabetic_retinopathy
source venv/bin/activate

which python
python --version

python $SCRIPT_PATH
EOT
