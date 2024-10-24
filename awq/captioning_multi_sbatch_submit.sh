python captioning_multi_sbatch.py --env slurm_files \
                                  --nhrs 2 \
                                  --qos scav \
                                  --partition nexus \
                                  --gpu 1 --gpu-type a5000 a6000 \
                                  --cores 1 \
                                  --mem 64 \
                                  --output-dirname captioning_output \
                                #  --dryrun
                                #  --base-dir awq/ \
                                

