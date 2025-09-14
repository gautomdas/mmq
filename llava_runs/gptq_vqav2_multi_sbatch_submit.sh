python multi_sbatch_gptq_vqav2.py --env slurm_files \
                                  --nhrs 4 \
                                  --qos scav \
                                  --partition vulcan \
                                  --gpu 1 \
                                  --gpu-type a5000 a6000 \
                                  --cores 1 \
                                  --mem 48 \
                                  --output-dirname vqav2_subset_gptq_output \
                                  # --dryrun
                                

