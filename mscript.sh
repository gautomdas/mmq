python multi_sbatch.py --env trial \
                              --nhrs 8 \
                              --qos scav \
                              --partition nexus \
                              --gpu 8 --gpu-type a5000 a6000 \
                              --cores 1 \
                              --mem 128 \
                              --base-dir ./ \
                              --output-dirname slurm_files \
			                  --offset 2 \
							  --batchsize 2

